use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use candle::{safetensors::Load, DType, Device};
use memmap2::MmapOptions;
use safetensors::{tensor::Dtype as SafeDtype, SafeTensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::quant_sensitivity::{is_protected_tensor, tensor_family, tensor_protection_reason};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize)]
pub enum MixedPrecisionPolicy {
    Q8TextLinears,
    Q4KMlpOnly,
    Q4KTextSafe,
}

impl MixedPrecisionPolicy {
    pub fn name(self) -> &'static str {
        match self {
            Self::Q8TextLinears => "q8-text-linears",
            Self::Q4KMlpOnly => "q4k-mlp-only",
            Self::Q4KTextSafe => "q4k-text-safe",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConversionOptions {
    pub model_id: String,
    pub revision: String,
    pub policy: MixedPrecisionPolicy,
    pub source_weights: Vec<PathBuf>,
    pub sensitivity_artifact: PathBuf,
    pub output_dir: PathBuf,
    pub max_tensors: Option<usize>,
    pub manifest_only: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct SensitivityArtifact {
    kind: String,
    schema_version: usize,
    #[serde(default)]
    candidate_quants: Vec<String>,
    weights: SensitivityWeights,
}

#[derive(Clone, Debug, Deserialize)]
struct SensitivityWeights {
    #[serde(default)]
    modules: Vec<SensitivityModule>,
}

#[derive(Clone, Debug, Deserialize)]
struct SensitivityModule {
    name: String,
    family: String,
    candidate_quant: String,
    recommended_policy: String,
}

#[derive(Clone, Debug, Serialize)]
struct SourceFileManifest {
    path: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct TensorManifest {
    name: String,
    family: String,
    source_file: String,
    source_dtype: String,
    shape: Vec<usize>,
    num_elements: usize,
    source_bytes: usize,
    format: String,
    protected: bool,
    protection_reason: Option<String>,
    expected_weight_bytes: usize,
    data: Option<QuantizedDataManifest>,
}

#[derive(Clone, Debug, Serialize)]
struct QuantizedDataManifest {
    file: String,
    offset: u64,
    length: u64,
    block_size: Option<usize>,
    scale_dtype: String,
    quantized_dtype: String,
}

#[derive(Clone, Debug, Serialize)]
struct ConversionSummary {
    tensors_total: usize,
    tensors_quantized: usize,
    tensors_preserved: usize,
    source_bytes_total: usize,
    expected_weight_bytes_total: usize,
    quantized_data_bytes: u64,
    by_format: BTreeMap<String, usize>,
    by_family: BTreeMap<String, usize>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum TensorFormat {
    PreserveSource,
    Q8Symmetric,
    Q4KBlock64,
}

impl TensorFormat {
    fn name(self) -> &'static str {
        match self {
            Self::PreserveSource => "source",
            Self::Q8Symmetric => "q8_symmetric",
            Self::Q4KBlock64 => "q4k_block64_symmetric",
        }
    }
}

pub fn convert_mixed_precision(options: ConversionOptions) -> Result<serde_json::Value> {
    if options.source_weights.is_empty() {
        anyhow::bail!("no source weights supplied");
    }
    fs::create_dir_all(&options.output_dir)
        .with_context(|| format!("create output dir {}", options.output_dir.display()))?;

    let sensitivity = read_sensitivity(&options.sensitivity_artifact)?;
    validate_sensitivity(&sensitivity)?;
    let sensitivity_sha256 = sha256_file(&options.sensitivity_artifact)?;
    let sensitivity_modules = sensitivity_module_set(&sensitivity);
    let source_files = options
        .source_weights
        .iter()
        .map(|path| {
            let metadata = fs::metadata(path)
                .with_context(|| format!("stat source weight {}", path.display()))?;
            Ok(SourceFileManifest {
                path: path.display().to_string(),
                size_bytes: metadata.len(),
                sha256: sha256_file(path)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let data_filename = "weights.lmbq";
    let data_path = options.output_dir.join(data_filename);
    let mut data_writer = if options.manifest_only {
        None
    } else {
        Some(BufWriter::new(
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&data_path)
                .with_context(|| format!("open quantized data {}", data_path.display()))?,
        ))
    };
    let mut data_offset = 0u64;
    let mut tensors = Vec::new();
    let mut summary = ConversionSummary {
        tensors_total: 0,
        tensors_quantized: 0,
        tensors_preserved: 0,
        source_bytes_total: 0,
        expected_weight_bytes_total: 0,
        quantized_data_bytes: 0,
        by_format: BTreeMap::new(),
        by_family: BTreeMap::new(),
    };
    let mut quantized_count = 0usize;

    for path in &options.source_weights {
        let file = File::open(path).with_context(|| format!("open weights {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("mmap {}", path.display()))?;
        let safetensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("read safetensors {}", path.display()))?;
        let mut names = safetensors.names();
        names.sort_unstable();

        for name in names {
            let view = safetensors
                .tensor(name)
                .with_context(|| format!("read tensor metadata {name}"))?;
            let shape = view.shape().to_vec();
            let num_elements = shape_num_elements(&shape);
            let source_bytes = num_elements * dtype_size_bytes(view.dtype());
            let family = tensor_family(name).to_string();
            let protected = is_protected_tensor(name);
            let protection_reason = tensor_protection_reason(name).map(str::to_string);
            let planned_format = planned_tensor_format(
                options.policy,
                name,
                &family,
                &shape,
                protected,
                &sensitivity_modules,
            );
            let format = if options
                .max_tensors
                .is_some_and(|limit| quantized_count >= limit)
                && planned_format != TensorFormat::PreserveSource
            {
                TensorFormat::PreserveSource
            } else {
                planned_format
            };
            let expected_weight_bytes = expected_weight_bytes(num_elements, source_bytes, format);
            let data = match format {
                TensorFormat::PreserveSource => None,
                TensorFormat::Q8Symmetric | TensorFormat::Q4KBlock64 => {
                    if options.manifest_only {
                        data_offset += expected_weight_bytes as u64;
                        quantized_count += 1;
                        None
                    } else {
                        let values = load_tensor_values(&view)
                            .with_context(|| format!("load tensor values for conversion {name}"))?;
                        let encoded = encode_values(&values, format);
                        let length = encoded.len() as u64;
                        let writer = data_writer
                            .as_mut()
                            .context("quantized data writer missing outside manifest-only mode")?;
                        writer.write_all(&encoded).with_context(|| {
                            format!("write quantized tensor {name} to {}", data_path.display())
                        })?;
                        let manifest = QuantizedDataManifest {
                            file: data_filename.to_string(),
                            offset: data_offset,
                            length,
                            block_size: (format == TensorFormat::Q4KBlock64).then_some(64),
                            scale_dtype: "f32_le".to_string(),
                            quantized_dtype: match format {
                                TensorFormat::Q8Symmetric => "i8".to_string(),
                                TensorFormat::Q4KBlock64 => "packed_i4_plus_8".to_string(),
                                TensorFormat::PreserveSource => unreachable!(),
                            },
                        };
                        data_offset += length;
                        quantized_count += 1;
                        Some(manifest)
                    }
                }
            };

            summary.tensors_total += 1;
            summary.source_bytes_total += source_bytes;
            summary.expected_weight_bytes_total += expected_weight_bytes;
            summary.quantized_data_bytes = data_offset;
            if format == TensorFormat::PreserveSource {
                summary.tensors_preserved += 1;
            } else {
                summary.tensors_quantized += 1;
            }
            *summary
                .by_format
                .entry(format.name().to_string())
                .or_insert(0) += 1;
            *summary.by_family.entry(family.clone()).or_insert(0) += 1;

            tensors.push(TensorManifest {
                name: name.to_string(),
                family,
                source_file: path.display().to_string(),
                source_dtype: format!("{:?}", view.dtype()),
                shape,
                num_elements,
                source_bytes,
                format: format.name().to_string(),
                protected,
                protection_reason,
                expected_weight_bytes,
                data,
            });
        }
    }

    if let Some(writer) = data_writer.as_mut() {
        writer.flush().context("flush quantized data")?;
    }

    let manifest_path = options.output_dir.join("manifest.json");
    let manifest = serde_json::json!({
        "kind": "lmbrrr_mixed_precision_weights",
        "schema_version": 1,
        "model_id": options.model_id,
        "revision": options.revision,
        "policy": options.policy.name(),
        "format_note": "custom lmbrrr manifest plus packed quantized tensor data; source-format tensors are exact references to original safetensors",
        "source_files": source_files,
        "sensitivity_artifact": {
            "path": options.sensitivity_artifact,
            "sha256": sensitivity_sha256,
            "kind": sensitivity.kind,
            "schema_version": sensitivity.schema_version,
            "candidate_quants": sensitivity.candidate_quants,
        },
        "output": {
            "directory": options.output_dir,
            "data_file": (!options.manifest_only).then_some(data_filename),
            "manifest_only": options.manifest_only,
        },
        "summary": summary,
        "tensors": tensors,
    });
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&manifest_path)
        .with_context(|| format!("open manifest {}", manifest_path.display()))?;
    serde_json::to_writer_pretty(&mut file, &manifest)?;
    file.write_all(b"\n")?;
    Ok(manifest)
}

fn read_sensitivity(path: &Path) -> Result<SensitivityArtifact> {
    let file = File::open(path).with_context(|| format!("open sensitivity {}", path.display()))?;
    serde_json::from_reader(file).with_context(|| format!("parse sensitivity {}", path.display()))
}

fn validate_sensitivity(sensitivity: &SensitivityArtifact) -> Result<()> {
    if sensitivity.kind != "lmbrrr_quantization_sensitivity" {
        anyhow::bail!(
            "sensitivity artifact kind is {}, expected lmbrrr_quantization_sensitivity",
            sensitivity.kind
        );
    }
    if sensitivity.weights.modules.is_empty() {
        anyhow::bail!("sensitivity artifact contains no module scores");
    }
    Ok(())
}

fn sensitivity_module_set(sensitivity: &SensitivityArtifact) -> BTreeSet<String> {
    sensitivity
        .weights
        .modules
        .iter()
        .filter(|module| {
            module.family.starts_with("text.")
                && (module.recommended_policy.starts_with("candidate")
                    || module.candidate_quant == "q8_symmetric")
        })
        .map(|module| module.name.clone())
        .collect()
}

fn planned_tensor_format(
    policy: MixedPrecisionPolicy,
    name: &str,
    family: &str,
    shape: &[usize],
    protected: bool,
    sensitivity_modules: &BTreeSet<String>,
) -> TensorFormat {
    if protected || shape.len() < 2 || !name.ends_with(".weight") {
        return TensorFormat::PreserveSource;
    }
    if !sensitivity_modules.contains(name) {
        return TensorFormat::PreserveSource;
    }

    match policy {
        MixedPrecisionPolicy::Q8TextLinears => {
            if matches!(family, "text.mlp" | "text.full_attention" | "text.deltanet") {
                TensorFormat::Q8Symmetric
            } else {
                TensorFormat::PreserveSource
            }
        }
        MixedPrecisionPolicy::Q4KMlpOnly => {
            if family == "text.mlp" {
                TensorFormat::Q4KBlock64
            } else {
                TensorFormat::PreserveSource
            }
        }
        MixedPrecisionPolicy::Q4KTextSafe => {
            if family == "text.mlp"
                || (family == "text.full_attention" && attention_projection(name))
            {
                TensorFormat::Q4KBlock64
            } else {
                TensorFormat::PreserveSource
            }
        }
    }
}

fn attention_projection(name: &str) -> bool {
    name.ends_with(".self_attn.q_proj.weight")
        || name.ends_with(".self_attn.k_proj.weight")
        || name.ends_with(".self_attn.v_proj.weight")
        || name.ends_with(".self_attn.o_proj.weight")
}

fn encode_values(values: &[f32], format: TensorFormat) -> Vec<u8> {
    match format {
        TensorFormat::PreserveSource => Vec::new(),
        TensorFormat::Q8Symmetric => encode_q8(values),
        TensorFormat::Q4KBlock64 => encode_q4_block64(values),
    }
}

fn encode_q8(values: &[f32]) -> Vec<u8> {
    let scale = symmetric_scale(values, 127.0);
    let mut out = Vec::with_capacity(4 + values.len());
    out.extend_from_slice(&scale.to_le_bytes());
    for value in values {
        let quantized = quantize_symmetric(*value, scale, 127.0) as i8;
        out.push(quantized as u8);
    }
    out
}

fn encode_q4_block64(values: &[f32]) -> Vec<u8> {
    const BLOCK: usize = 64;
    let mut out = Vec::with_capacity(values.len().div_ceil(BLOCK) * (4 + BLOCK / 2));
    for chunk in values.chunks(BLOCK) {
        let scale = symmetric_scale(chunk, 7.0);
        out.extend_from_slice(&scale.to_le_bytes());
        let mut iter = chunk.iter();
        while let Some(left) = iter.next() {
            let left = (quantize_symmetric(*left, scale, 7.0) + 8) as u8 & 0x0f;
            let right = iter
                .next()
                .map(|value| (quantize_symmetric(*value, scale, 7.0) + 8) as u8 & 0x0f)
                .unwrap_or(8);
            out.push(left | (right << 4));
        }
    }
    out
}

fn symmetric_scale(values: &[f32], qmax: f32) -> f32 {
    let max_abs = values.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if max_abs > 0.0 {
        max_abs / qmax
    } else {
        1.0
    }
}

fn quantize_symmetric(value: f32, scale: f32, qmax: f32) -> i32 {
    if scale == 0.0 {
        0
    } else {
        (value / scale).round().clamp(-qmax, qmax) as i32
    }
}

fn load_tensor_values(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    let tensor = view.load(&Device::Cpu)?;
    Ok(tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

fn expected_weight_bytes(num_elements: usize, source_bytes: usize, format: TensorFormat) -> usize {
    match format {
        TensorFormat::PreserveSource => source_bytes,
        TensorFormat::Q8Symmetric => 4 + num_elements,
        TensorFormat::Q4KBlock64 => {
            let blocks = num_elements.div_ceil(64);
            blocks * 4 + num_elements.div_ceil(2)
        }
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn shape_num_elements(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn dtype_size_bytes(dtype: SafeDtype) -> usize {
    match dtype {
        SafeDtype::BOOL
        | SafeDtype::U8
        | SafeDtype::I8
        | SafeDtype::F8_E5M2
        | SafeDtype::F8_E4M3 => 1,
        SafeDtype::I16 | SafeDtype::U16 | SafeDtype::F16 | SafeDtype::BF16 => 2,
        SafeDtype::I32 | SafeDtype::U32 | SafeDtype::F32 => 4,
        SafeDtype::I64 | SafeDtype::U64 | SafeDtype::F64 => 8,
        SafeDtype::F6_E2M3 | SafeDtype::F6_E3M2 | SafeDtype::F4 | SafeDtype::F8_E8M0 => 1,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_block64_size_accounts_for_scales_and_packed_nibbles() {
        assert_eq!(expected_weight_bytes(64, 128, TensorFormat::Q4KBlock64), 36);
        assert_eq!(expected_weight_bytes(65, 130, TensorFormat::Q4KBlock64), 41);
    }

    #[test]
    fn q8_encoding_has_one_scale_and_one_byte_per_value() {
        let encoded = encode_q8(&[-1.0, 0.0, 1.0]);
        assert_eq!(encoded.len(), 7);
        assert_eq!(encoded[4] as i8, -127);
        assert_eq!(encoded[5] as i8, 0);
        assert_eq!(encoded[6] as i8, 127);
    }

    #[test]
    fn policy_keeps_deltanet_out_of_q4_text_safe() {
        let mut sensitivity = BTreeSet::new();
        sensitivity
            .insert("model.language_model.layers.0.linear_attn.in_proj_qkv.weight".to_string());
        sensitivity.insert("model.language_model.layers.11.self_attn.q_proj.weight".to_string());
        assert_eq!(
            planned_tensor_format(
                MixedPrecisionPolicy::Q4KTextSafe,
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                "text.deltanet",
                &[64, 64],
                false,
                &sensitivity,
            ),
            TensorFormat::PreserveSource
        );
        assert_eq!(
            planned_tensor_format(
                MixedPrecisionPolicy::Q4KTextSafe,
                "model.language_model.layers.11.self_attn.q_proj.weight",
                "text.full_attention",
                &[64, 64],
                false,
                &sensitivity,
            ),
            TensorFormat::Q4KBlock64
        );
    }
}
