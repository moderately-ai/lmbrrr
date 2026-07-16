use std::{
    collections::HashMap,
    fmt,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result};
use candle::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Module, Tensor,
};
use candle_nn::Linear;
use serde::Deserialize;

#[derive(Clone)]
pub enum MixedLinear {
    Dense(Linear),
    QMatMul {
        matmul: Arc<QMatMul>,
        force_f32_input: bool,
        // The fork's Metal kernels take BF16 activations directly for these
        // block dtypes, skipping the input-cast dispatch per call.
        bf16_direct: bool,
        // Tensor-op planes (mm2d verify route); shared across clones so the
        // repack happens once per weight. Exactly one of these is populated,
        // keyed by the weight dtype (q4_K vs the ternary Q2_0 verify path).
        mm2d: Arc<std::sync::OnceLock<Option<crate::mm2d::Mm2dPlanes>>>,
        mm2d_q2: Arc<std::sync::OnceLock<Option<crate::mm2d::Mm2dQ2Planes>>>,
        // Route configuration, resolved at the entrypoint and passed down
        // at construction — modules never read the environment.
        mm2d_cfg: Arc<crate::mm2d::Mm2dConfig>,
    },
    /// Planar-only Q2_0 weight (cfg.planar_only): ONLY the mm2d planes are
    /// resident — the raw ggml copy was dropped at load, halving the target's
    /// footprint so the full plane set fits the M3's GPU budget. Every m runs
    /// on the tensor-op kernel (m > 8 tiles by 8); there is no GEMV fallback,
    /// so a dispatch failure here is an error, not a reroute.
    Mm2dOnly {
        planes: Arc<crate::mm2d::Mm2dQ2Planes>,
        mm2d_cfg: Arc<crate::mm2d::Mm2dConfig>,
    },
}

impl MixedLinear {
    pub fn dense(linear: Linear) -> Self {
        Self::Dense(linear)
    }

    pub fn from_qtensor(
        weight: QTensor,
        mm2d_cfg: Arc<crate::mm2d::Mm2dConfig>,
    ) -> candle::Result<Self> {
        use candle::quantized::GgmlDType;
        let bf16_direct = matches!(
            weight.dtype(),
            GgmlDType::Q8_0 | GgmlDType::Q4K | GgmlDType::Q6K | GgmlDType::Q2_0
        );
        // Tensor-op planes build EAGERLY at load when the route is enabled:
        // the CPU repack must never land inside the decode window (it
        // contaminated the first verify's timing when lazy). Load-time cost
        // is reported via load_seconds.
        let mm2d = std::sync::OnceLock::new();
        let mm2d_q2 = std::sync::OnceLock::new();
        // min_n gates the BUILD, not just the route: planes are a second
        // device-resident copy of the weight, and building them for weights
        // the forward will never route (n < min_n) blows the M3's ~13 GB GPU
        // working-set budget — which does not error, it silently corrupts
        // resident buffers (measured: the spec prefill read f32 1.0 bit
        // patterns as token ids with all ~5.3 GB of Q2_0 planes resident).
        let n_rows = weight.shape().dims().first().copied().unwrap_or(0);
        // Planar-only: build the planes and DROP the raw weight — return the
        // planes-only variant. A build failure falls through to the normal
        // QMatMul construction below (raw stays, GEMV route).
        if mm2d_cfg.enabled
            && mm2d_cfg.planar_only
            && n_rows >= mm2d_cfg.min_n.max(crate::mm2d::PLANAR_ONLY_MIN_N)
            && crate::mm2d::mm2d_q2_plane_eligible(&weight)
        {
            if let candle::Device::Metal(dev) = weight.device() {
                match crate::mm2d::Mm2dQ2Planes::from_qtensor(
                    &weight,
                    &dev,
                    mm2d_cfg.plane_cache_dir.as_deref(),
                ) {
                    Ok(p) => {
                        return Ok(Self::Mm2dOnly {
                            planes: Arc::new(p),
                            mm2d_cfg,
                        });
                    }
                    Err(err) => {
                        eprintln!(
                            "warning: planar-only q2 repack failed, keeping the raw weight ({err})"
                        );
                    }
                }
            }
        }
        if mm2d_cfg.enabled && !mm2d_cfg.planar_only && n_rows >= mm2d_cfg.min_n {
            if let candle::Device::Metal(dev) = weight.device() {
                // Non-planar planes DUPLICATE the weight on-device. Exceeding
                // the GPU working-set budget does not error — it silently
                // corrupts resident buffers (twice documented on the 18 GB
                // M3) — so stop building planes when the next one would come
                // within 10% of the budget. Planar mode replaces the raw copy
                // and does not need this guard.
                let projected = dev.current_allocated_size()
                    + weight.storage_size_in_bytes();
                if projected as f64 > dev.recommended_max_working_set_size() as f64 * 0.9 {
                    static BUDGET_WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                    BUDGET_WARNED.get_or_init(|| {
                        eprintln!(
                            "warning: skipping further mm2d plane builds — GPU allocation would approach the working-set budget (use LMBRRR_MM2D_PLANAR=1 or LMBRRR_MM2D_MIN_N)"
                        );
                    });
                    let _ = mm2d.set(None);
                    let _ = mm2d_q2.set(None);
                    return Ok(Self::QMatMul {
                        matmul: Arc::new(QMatMul::from_qtensor(weight)?),
                        force_f32_input: true,
                        bf16_direct,
                        mm2d: Arc::new(mm2d),
                        mm2d_q2: Arc::new(mm2d_q2),
                        mm2d_cfg,
                    });
                }
                match weight.dtype() {
                    GgmlDType::Q4K => match crate::mm2d::Mm2dPlanes::from_qtensor(
                        &weight,
                        &dev,
                        mm2d_cfg.plane_cache_dir.as_deref(),
                    ) {
                        Ok(p) => {
                            let _ = mm2d.set(Some(p));
                        }
                        Err(err) => {
                            eprintln!("warning: mm2d repack failed, wide route stays ({err})");
                            let _ = mm2d.set(None);
                        }
                    },
                    // Ineligible weights (ffn_down: k=17408 > the kernel's
                    // 8192 limit) stay on the GEMV route by design — a silent
                    // skip, not a failure. The warning is reserved for real
                    // build errors on eligible weights.
                    GgmlDType::Q2_0 if crate::mm2d::mm2d_q2_plane_eligible(&weight) => {
                        match crate::mm2d::Mm2dQ2Planes::from_qtensor(
                            &weight,
                            &dev,
                            mm2d_cfg.plane_cache_dir.as_deref(),
                        ) {
                            Ok(p) => {
                                let _ = mm2d_q2.set(Some(p));
                            }
                            Err(err) => {
                                eprintln!(
                                    "warning: mm2d q2 repack failed, wide route stays ({err})"
                                );
                                let _ = mm2d_q2.set(None);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(Self::QMatMul {
            matmul: Arc::new(QMatMul::from_qtensor(weight)?),
            force_f32_input: true,
            bf16_direct,
            mm2d: Arc::new(mm2d),
            mm2d_q2: Arc::new(mm2d_q2),
            mm2d_cfg,
        })
    }

    /// The underlying quantized weight, when this linear wraps a QTensor —
    /// fused kernels (the Metal Markov chain) read its ggml blocks directly.
    pub fn qtensor(&self) -> Option<&QTensor> {
        match self {
            Self::QMatMul { matmul, .. } => match matmul.as_ref() {
                QMatMul::QTensor(q) => Some(q),
                _ => None,
            },
            Self::Dense(_) | Self::Mm2dOnly { .. } => None,
        }
    }

    /// The tensor-op planes, when built (eager at load under LMBRRR_MM2D).
    pub fn mm2d_planes(&self) -> Option<&crate::mm2d::Mm2dPlanes> {
        match self {
            Self::QMatMul { mm2d, .. } => mm2d.get().and_then(|p| p.as_ref()),
            Self::Dense(_) | Self::Mm2dOnly { .. } => None,
        }
    }

    /// The dense weight tensor, when unquantized (or dequantized-at-load).
    pub fn dense_weight(&self) -> Option<&Tensor> {
        match self {
            Self::Dense(linear) => Some(linear.weight()),
            Self::QMatMul { matmul, .. } => match matmul.as_ref() {
                QMatMul::Tensor(t) => Some(t),
                _ => None,
            },
            Self::Mm2dOnly { .. } => None,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(xs),
            Self::Mm2dOnly { planes, mm2d_cfg } => {
                // The tensor-op kernel reads BF16 activations; F32 callers
                // pay the cast both ways (margin-class either way — the
                // planar route is not bit-compatible with the GEMV one).
                let x = if xs.dtype() == DType::BF16 {
                    xs.clone()
                } else {
                    xs.to_dtype(DType::BF16)?
                };
                let out = crate::mm2d::mm2d_q2_0_forward(&x, planes, mm2d_cfg)
                    .map_err(|e| candle::Error::Msg(format!("mm2d planar-only: {e:#}")))?;
                if xs.dtype() == DType::BF16 {
                    Ok(out)
                } else {
                    out.to_dtype(xs.dtype())
                }
            }
            Self::QMatMul {
                matmul,
                force_f32_input,
                bf16_direct,
                mm2d,
                mm2d_q2,
                mm2d_cfg,
            } => {
                if *force_f32_input && xs.dtype() != DType::F32 {
                    if *bf16_direct && xs.dtype() == DType::BF16 && xs.device().is_metal() {
                        // Tensor-op route for verify-chunk shapes (m in
                        // [2,8]): the matrix units run the whole 8-row tile
                        // at ~1.25x one GEMV read. Margin-class (not
                        // bit-compatible with the wide kernels), config-
                        // gated; any dispatch failure disables it
                        // process-wide and falls through to the wide route.
                        if crate::mm2d::mm2d_eligible(xs, mm2d_cfg) {
                            if let Some(Some(planes)) = mm2d.get() {
                                if let Ok(out) =
                                    crate::mm2d::mm2d_q4k_forward(xs, planes, mm2d_cfg)
                                {
                                    return Ok(out);
                                }
                            }
                            if let Some(Some(planes)) = mm2d_q2.get() {
                                if let Ok(out) =
                                    crate::mm2d::mm2d_q2_0_forward(xs, planes, mm2d_cfg)
                                {
                                    return Ok(out);
                                }
                            }
                        }
                        // mv/mc routes write BF16 directly (fork bf16-dst
                        // kernels; bit-identical to F32 + cast), making the
                        // to_dtype a no-op; the m>=8 tile-mm route still
                        // returns F32 and pays the cast here.
                        matmul.forward(xs)?.to_dtype(xs.dtype())
                    } else {
                        matmul
                            .forward(&xs.to_dtype(DType::F32)?)?
                            .to_dtype(xs.dtype())
                    }
                } else {
                    matmul.forward(xs)
                }
            }
        }
    }
}

impl fmt::Debug for MixedLinear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense(_) => f.write_str("MixedLinear::Dense"),
            Self::Mm2dOnly { .. } => f.write_str("MixedLinear::Mm2dOnly"),
            Self::QMatMul {
                force_f32_input, ..
            } => f
                .debug_struct("MixedLinear::QMatMul")
                .field("force_f32_input", force_f32_input)
                .finish(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct QuantizedTextArtifact {
    manifest_path: PathBuf,
    data_path: Option<PathBuf>,
    tensors: HashMap<String, QuantizedTensor>,
    device: Device,
    dtype: DType,
    pack: Option<Arc<crate::pack::PackStore>>,
    /// Construction context injected at build; every MixedLinear this
    /// artifact produces is built through its factory.
    ctx: crate::model_ctx::ModelCtx,
}

impl QuantizedTextArtifact {
    pub fn from_manifest(
        path: &Path,
        device: &Device,
        dtype: DType,
        ctx: crate::model_ctx::ModelCtx,
    ) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("open quantized manifest {}", path.display()))?;
        let manifest: QuantizedManifest = serde_json::from_reader(std::io::BufReader::new(file))
            .with_context(|| format!("parse quantized manifest {}", path.display()))?;
        if manifest.kind != "lmbrrr_mixed_precision_weights" {
            anyhow::bail!(
                "quantized manifest kind is {}, expected lmbrrr_mixed_precision_weights",
                manifest.kind
            );
        }
        let manifest_dir = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let data_path = manifest
            .output
            .data_file
            .as_ref()
            .map(|file| manifest_dir.join(file));
        let tensors = manifest
            .tensors
            .into_iter()
            .filter_map(|tensor| {
                if tensor.format == "source" {
                    return None;
                }
                // From-source formats carry no data blob; lmbq formats must.
                let from_source = tensor.format.ends_with("_from_source");
                if !from_source && tensor.data.is_none() {
                    return None;
                }
                Some((
                    tensor.name,
                    QuantizedTensor {
                        format: tensor.format,
                        shape: tensor.shape,
                        num_elements: tensor.num_elements,
                        expected_weight_bytes: tensor.expected_weight_bytes,
                        source_file: tensor.source_file,
                        data: tensor.data,
                    },
                ))
            })
            .collect::<HashMap<_, _>>();
        if tensors.values().any(|t| t.data.is_some()) && data_path.is_none() {
            anyhow::bail!("quantized manifest has tensor data entries but no output.data_file");
        }
        Ok(Self {
            manifest_path: path.to_path_buf(),
            data_path,
            tensors,
            device: device.clone(),
            dtype,
            pack: None,
            ctx,
        })
    }

    /// Attaches the GGML-ready weight pack: subsequent loads consume packed
    /// tensors instead of decode+requantize, and misses are recorded for the
    /// pack write (see crate::pack).
    pub fn set_pack(&mut self, pack: Arc<crate::pack::PackStore>) {
        self.pack = Some(pack);
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn quantized_tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn backend(&self) -> &'static str {
        match &self.pack {
            Some(pack) if pack.hit() => "candle_qtensor_packed",
            _ => "candle_qtensor_requantized",
        }
    }

    pub fn quantized_data_bytes(&self) -> u64 {
        self.tensors
            .values()
            .map(|tensor| match &tensor.data {
                Some(data) => data.length,
                // From-source tensors: on-device GGML footprint from the
                // manifest's accounting.
                None => tensor.expected_weight_bytes.unwrap_or(0) as u64,
            })
            .sum::<u64>()
    }

    pub fn dense_equivalent_bytes(&self) -> usize {
        self.tensors
            .values()
            .map(|tensor| tensor.num_elements * dtype_size_bytes(self.dtype))
            .sum()
    }

    pub fn load_linear(&self, name: &str) -> Result<Option<MixedLinear>> {
        let Some(tensor) = self.tensors.get(name) else {
            return Ok(None);
        };
        if let Some(pack) = &self.pack {
            if let Some(qweight) = pack.take(name) {
                return Ok(Some(self.ctx.quantized_linear(qweight)?));
            }
        }
        let values = self.load_values(name, tensor)?;
        let cpu_weight = Tensor::from_vec(values, tensor.shape.clone(), &Device::Cpu)?;
        let dtype = tensor
            .ggml_dtype()
            .with_context(|| format!("map quantized tensor format for {name}"))?;
        let qweight = self
            .quantize_for_device(name, &cpu_weight, dtype)
            .with_context(|| format!("requantize {name} into Candle {dtype:?} QTensor"))?;
        Ok(Some(self.ctx.quantized_linear(qweight)?))
    }

    /// Slow-path quantization, routed through the pack when attached so the
    /// bytes get recorded for the pack write (bit-identical either way: the
    /// pack runs the same CPU quantizer as quantize_onto).
    fn quantize_for_device(
        &self,
        key: &str,
        cpu_weight: &Tensor,
        dtype: GgmlDType,
    ) -> Result<QTensor> {
        match &self.pack {
            Some(pack) => pack.quantize_and_record(key, cpu_weight, dtype),
            None => QTensor::quantize_onto(cpu_weight, dtype, &self.device)
                .context("quantize_onto"),
        }
    }

    /// Loads several [out, in] weight tensors as ONE row-concatenated
    /// quantized linear (width fusion: one wide GEMV instead of N skinny
    /// ones). Bitwise-identical quantization to loading them separately:
    /// GGML blocks span the in-dimension within a row, so concatenating
    /// along out-rows never crosses a block boundary. Returns None if any
    /// tensor is absent (caller falls back to separate loads); errors if
    /// present tensors disagree on format or in-dim.
    pub fn load_linear_fused(&self, names: &[&str]) -> Result<Option<MixedLinear>> {
        let mut tensors = Vec::with_capacity(names.len());
        for name in names {
            match self.tensors.get(*name) {
                Some(tensor) => tensors.push((*name, tensor)),
                None => return Ok(None),
            }
        }
        let key = names.join("+");
        if let Some(pack) = &self.pack {
            if let Some(qweight) = pack.take(&key) {
                return Ok(Some(self.ctx.quantized_linear(qweight)?));
            }
        }
        let in_dim = tensors[0].1.shape[1];
        let dtype = tensors[0]
            .1
            .ggml_dtype()
            .with_context(|| format!("map quantized tensor format for {}", tensors[0].0))?;
        let mut values = Vec::new();
        let mut out_dim = 0usize;
        for (name, tensor) in &tensors {
            if tensor.shape.len() != 2 || tensor.shape[1] != in_dim {
                anyhow::bail!(
                    "fused linear {name} has shape {:?}, expected [*, {in_dim}]",
                    tensor.shape
                );
            }
            let fmt = tensor
                .ggml_dtype()
                .with_context(|| format!("map quantized tensor format for {name}"))?;
            if fmt != dtype {
                anyhow::bail!(
                    "fused linear {name} format {fmt:?} differs from {dtype:?}; \
                     refusing to mix quantization rungs in one fused weight"
                );
            }
            out_dim += tensor.shape[0];
            values.extend(self.load_values(name, tensor)?);
        }
        let cpu_weight = Tensor::from_vec(values, (out_dim, in_dim), &Device::Cpu)?;
        let qweight = self
            .quantize_for_device(&key, &cpu_weight, dtype)
            .with_context(|| format!("requantize fused {names:?} into {dtype:?} QTensor"))?;
        Ok(Some(self.ctx.quantized_linear(qweight)?))
    }

    fn load_values(&self, name: &str, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
        // From-source formats read the original safetensors tensor directly,
        // so the only quantization applied is the final GGML one.
        if tensor.format.ends_with("_from_source") {
            return self.load_source_values(name, tensor);
        }
        let data = tensor
            .data
            .as_ref()
            .with_context(|| format!("quantized tensor {name} has no data blob"))?;
        let data_path = self
            .data_path
            .as_ref()
            .context("quantized artifact has no data file")?;
        let mut file = File::open(data_path)
            .with_context(|| format!("open quantized data {}", data_path.display()))?;
        file.seek(SeekFrom::Start(data.offset))
            .with_context(|| format!("seek quantized tensor {name}"))?;
        let mut bytes = vec![0u8; data.length as usize];
        file.read_exact(&mut bytes)
            .with_context(|| format!("read quantized tensor {name}"))?;
        match tensor.format.as_str() {
            "q8_symmetric" => decode_q8(name, &bytes, tensor.num_elements),
            "q4k_block64_symmetric" => decode_q4_block64(name, &bytes, tensor.num_elements),
            other => anyhow::bail!("unsupported quantized tensor format {other} for {name}"),
        }
    }

    fn load_source_values(&self, name: &str, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
        let recorded = tensor
            .source_file
            .as_ref()
            .with_context(|| format!("from-source tensor {name} has no source_file"))?;
        let recorded_path = PathBuf::from(recorded);
        let path = if recorded_path.exists() {
            recorded_path
        } else {
            // Fall back to the file name next to the manifest (artifact
            // moved between machines).
            let manifest_dir = self
                .manifest_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."));
            let file_name = recorded_path
                .file_name()
                .with_context(|| format!("source_file for {name} has no file name"))?;
            let fallback = manifest_dir.join(file_name);
            if !fallback.exists() {
                anyhow::bail!(
                    "source weights for {name} not found at {} or {}",
                    recorded_path.display(),
                    fallback.display()
                );
            }
            fallback
        };
        let file =
            File::open(&path).with_context(|| format!("open source weights {}", path.display()))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
            .with_context(|| format!("mmap {}", path.display()))?;
        let safetensors = safetensors::SafeTensors::deserialize(&mmap)
            .with_context(|| format!("read safetensors {}", path.display()))?;
        let view = safetensors
            .tensor(name)
            .with_context(|| format!("source tensor {name} missing from {}", path.display()))?;
        crate::quant_sensitivity::load_tensor_values(&view)
    }
}

#[derive(Clone, Debug)]
struct QuantizedTensor {
    format: String,
    shape: Vec<usize>,
    num_elements: usize,
    expected_weight_bytes: Option<usize>,
    source_file: Option<String>,
    data: Option<QuantizedData>,
}

impl QuantizedTensor {
    fn ggml_dtype(&self) -> Result<GgmlDType> {
        match self.format.as_str() {
            "q8_symmetric" | "q8_0_from_source" => Ok(GgmlDType::Q8_0),
            "q4k_block64_symmetric" | "q4k_from_source" => Ok(GgmlDType::Q4K),
            "q6k_from_source" => Ok(GgmlDType::Q6K),
            other => anyhow::bail!("unsupported quantized tensor format {other}"),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct QuantizedManifest {
    kind: String,
    output: QuantizedOutput,
    tensors: Vec<QuantizedTensorManifest>,
}

#[derive(Clone, Debug, Deserialize)]
struct QuantizedOutput {
    data_file: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct QuantizedTensorManifest {
    name: String,
    format: String,
    shape: Vec<usize>,
    num_elements: usize,
    #[serde(default)]
    expected_weight_bytes: Option<usize>,
    #[serde(default)]
    source_file: Option<String>,
    data: Option<QuantizedData>,
}

#[derive(Clone, Debug, Deserialize)]
struct QuantizedData {
    offset: u64,
    length: u64,
}

fn decode_q8(name: &str, bytes: &[u8], num_elements: usize) -> Result<Vec<f32>> {
    if bytes.len() != 4 + num_elements {
        anyhow::bail!(
            "q8 tensor {name} has {} bytes, expected {}",
            bytes.len(),
            4 + num_elements
        );
    }
    let scale = f32::from_le_bytes(bytes[..4].try_into().expect("slice has length 4"));
    Ok(bytes[4..]
        .iter()
        .map(|byte| (*byte as i8) as f32 * scale)
        .collect())
}

fn decode_q4_block64(name: &str, bytes: &[u8], num_elements: usize) -> Result<Vec<f32>> {
    const BLOCK: usize = 64;
    let expected = num_elements.div_ceil(BLOCK) * 4 + num_elements.div_ceil(2);
    if bytes.len() != expected {
        anyhow::bail!(
            "q4k tensor {name} has {} bytes, expected {expected}",
            bytes.len()
        );
    }
    let mut out = Vec::with_capacity(num_elements);
    let mut cursor = 0usize;
    for _ in 0..num_elements.div_ceil(BLOCK) {
        let scale = f32::from_le_bytes(
            bytes[cursor..cursor + 4]
                .try_into()
                .expect("slice has length 4"),
        );
        cursor += 4;
        let values_in_block = (num_elements - out.len()).min(BLOCK);
        for _ in 0..values_in_block.div_ceil(2) {
            let packed = bytes[cursor];
            cursor += 1;
            let left = ((packed & 0x0f) as i8) - 8;
            out.push(left as f32 * scale);
            if out.len() < num_elements && out.len() % BLOCK != 0 {
                let right = ((packed >> 4) as i8) - 8;
                out.push(right as f32 * scale);
            }
        }
    }
    out.truncate(num_elements);
    Ok(out)
}

fn dtype_size_bytes(dtype: DType) -> usize {
    match dtype {
        DType::U8 => 1,
        DType::U32 | DType::F32 => 4,
        DType::I64 | DType::F64 => 8,
        DType::F16 | DType::BF16 => 2,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn decodes_q8_values() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        bytes.extend_from_slice(&[254u8, 0u8, 2u8]);
        let values = decode_q8("test", &bytes, 3).unwrap();
        assert_eq!(values, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn loads_q8_as_qtensor() {
        let tempdir = tempfile::tempdir().unwrap();
        let data_path = tempdir.path().join("weights.lmbq");
        let manifest_path = tempdir.path().join("manifest.json");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        bytes.extend(std::iter::repeat_n(2u8, 32));
        let mut data_file = std::fs::File::create(&data_path).unwrap();
        data_file.write_all(&bytes).unwrap();

        std::fs::write(
            &manifest_path,
            r#"{
  "kind": "lmbrrr_mixed_precision_weights",
  "output": {"data_file": "weights.lmbq"},
  "tensors": [{
    "name": "linear.weight",
    "format": "q8_symmetric",
    "shape": [1, 32],
    "num_elements": 32,
    "data": {"offset": 0, "length": 36}
  }]
}"#,
        )
        .unwrap();

        let artifact = QuantizedTextArtifact::from_manifest(
            &manifest_path,
            &Device::Cpu,
            DType::F32,
            crate::model_ctx::ModelCtx::default(),
        )
        .unwrap();
        let linear = artifact.load_linear("linear.weight").unwrap().unwrap();
        assert_eq!(artifact.backend(), "candle_qtensor_requantized");
        assert_eq!(artifact.quantized_data_bytes(), 36);
        assert_eq!(artifact.dense_equivalent_bytes(), 32 * 4);
        assert!(matches!(
            linear,
            MixedLinear::QMatMul {
                force_f32_input: true,
                ..
            }
        ));
    }
}

/// A 248k-vocab token embedding kept packed: the packed block bytes are cached
/// host-side ONCE, and each `gather` slices only the requested rows (rows are
/// contiguous packed blocks — blocks run along the hidden dim) and dequantizes
/// just those. This keeps the table off the accelerator's resident budget (a
/// dense bf16 expansion is ~2.5 GB) without the per-call whole-table `qt.data()`
/// device->host copy a naive gather would pay.
pub struct PackedEmbed {
    bytes: Arc<Vec<u8>>,
    gd: GgmlDType,
    hidden: usize,
    row_bytes: usize,
}

impl PackedEmbed {
    pub fn new(qt: &QTensor) -> Result<Self> {
        let (_vocab, hidden) = qt.shape().dims2()?;
        let gd = qt.dtype();
        let row_bytes = (hidden / gd.block_size()) * gd.type_size();
        Ok(Self {
            bytes: Arc::new(qt.data()?.into_owned()),
            gd,
            hidden,
            row_bytes,
        })
    }

    /// Gather rows `ids` -> `[ids.len(), hidden]` in `dtype`.
    pub fn gather(&self, ids: &[u32], device: &Device, dtype: DType) -> Result<Tensor> {
        use candle::quantized::ggml_file::qtensor_from_ggml;
        let mut bytes = Vec::with_capacity(ids.len() * self.row_bytes);
        for &id in ids {
            let off = id as usize * self.row_bytes;
            bytes.extend_from_slice(&self.bytes[off..off + self.row_bytes]);
        }
        Ok(qtensor_from_ggml(self.gd, &bytes, vec![ids.len(), self.hidden], device)?
            .dequantize(device)?
            .to_dtype(dtype)?)
    }
}

impl Clone for PackedEmbed {
    fn clone(&self) -> Self {
        Self {
            bytes: self.bytes.clone(),
            gd: self.gd,
            hidden: self.hidden,
            row_bytes: self.row_bytes,
        }
    }
}
