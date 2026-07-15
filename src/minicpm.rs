use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{
    conv2d, layer_norm, linear, Conv2dConfig, LayerNorm, LayerNormConfig, Linear, VarBuilder,
};

use crate::{
    config::{MiniCpmConfig, VisionConfig},
    image_processor::ProcessedImages,
    quantized_linear::QuantizedTextArtifact,
    qwen35::{Qwen35Profiler, Qwen35TextModel, Qwen35TraceRecorder},
};

#[derive(Clone, Debug)]
struct VisionEmbeddings {
    patch_embedding: candle_nn::Conv2d,
    position_embedding: candle_nn::Embedding,
    num_patches_per_side: usize,
}

impl VisionEmbeddings {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        Ok(Self {
            patch_embedding: conv2d(
                cfg.num_channels,
                cfg.hidden_size,
                cfg.patch_size,
                conv_cfg,
                vb.pp("patch_embedding"),
            )?,
            position_embedding: candle_nn::embedding(
                (cfg.image_size / cfg.patch_size).pow(2),
                cfg.hidden_size,
                vb.pp("position_embedding"),
            )?,
            num_patches_per_side: cfg.image_size / cfg.patch_size,
        })
    }

    fn forward(&self, pixel_values: &Tensor, target_sizes: &[(usize, usize)]) -> Result<Tensor> {
        let patch_embeds = self.patch_embedding.forward(pixel_values)?;
        let embeddings = patch_embeds.flatten(2, 3)?.transpose(1, 2)?;
        let pos_ids = nearest_position_ids(target_sizes, self.num_patches_per_side);
        let pos_ids = Tensor::from_vec(
            pos_ids,
            target_sizes.iter().map(|(h, w)| h * w).sum::<usize>(),
            pixel_values.device(),
        )?;
        let position_embeddings = self.position_embedding.forward(&pos_ids)?.unsqueeze(0)?;
        embeddings.broadcast_add(&position_embeddings)
    }
}

#[derive(Clone, Debug)]
struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
    act: candle_nn::Activation,
}

impl VisionMlp {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?,
            fc2: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = xs.apply(&self.act)?;
        self.fc2.forward(&xs)
    }
}

#[derive(Clone, Debug)]
struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size;
        Ok(Self {
            q_proj: linear(dim, dim, vb.pp("q_proj"))?,
            k_proj: linear(dim, dim, vb.pp("k_proj"))?,
            v_proj: linear(dim, dim, vb.pp("v_proj"))?,
            out_proj: linear(dim, dim, vb.pp("out_proj"))?,
            num_heads: cfg.num_attention_heads,
            head_dim: dim / cfg.num_attention_heads,
        })
    }

    fn forward(&self, xs: &Tensor, cu_seqlens: &[usize]) -> Result<Tensor> {
        let (b, seq, dim) = xs.dims3()?;
        if b != 1 {
            candle::bail!("MiniCPM vision path expects packed batch size 1, got {b}");
        }
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut chunks = Vec::with_capacity(cu_seqlens.len().saturating_sub(1));
        for window in cu_seqlens.windows(2) {
            let start = window[0];
            let len = window[1] - start;
            if len == 0 {
                continue;
            }
            let q = q.narrow(2, start, len)?.contiguous()?;
            let k = k.narrow(2, start, len)?.contiguous()?;
            let v = v.narrow(2, start, len)?.contiguous()?;
            let k_t = k.transpose(2, 3)?.contiguous()?;
            let attn = (q.matmul(&k_t)? * scale)?;
            let attn = candle_nn::ops::softmax_last_dim(&attn.to_dtype(DType::F32)?)?
                .to_dtype(xs.dtype())?
                .contiguous()?;
            chunks.push(attn.matmul(&v)?.transpose(1, 2)?.reshape((b, len, dim))?);
        }
        let refs = chunks.iter().collect::<Vec<_>>();
        self.out_proj.forward(&Tensor::cat(&refs, 1)?)
    }
}

#[derive(Clone, Debug)]
struct VisionLayer {
    layer_norm1: LayerNorm,
    self_attn: VisionAttention,
    layer_norm2: LayerNorm,
    mlp: VisionMlp,
}

impl VisionLayer {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let ln_cfg = LayerNormConfig {
            eps: cfg.layer_norm_eps,
            ..Default::default()
        };
        Ok(Self {
            layer_norm1: layer_norm(cfg.hidden_size, ln_cfg, vb.pp("layer_norm1"))?,
            self_attn: VisionAttention::new(cfg, vb.pp("self_attn"))?,
            layer_norm2: layer_norm(cfg.hidden_size, ln_cfg, vb.pp("layer_norm2"))?,
            mlp: VisionMlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor, cu_seqlens: &[usize]) -> Result<Tensor> {
        let residual = xs;
        let hidden = self.layer_norm1.forward(xs)?;
        let hidden = self.self_attn.forward(&hidden, cu_seqlens)?;
        let xs = (residual + hidden)?;
        let residual = &xs;
        let hidden = self.layer_norm2.forward(&xs)?;
        let hidden = self.mlp.forward(&hidden)?;
        residual + hidden
    }
}

#[derive(Clone, Debug)]
struct WindowAttentionMerger {
    self_attn: VisionAttention,
    layer_norm1: LayerNorm,
    pre_norm: LayerNorm,
    linear_1: Linear,
    act: candle_nn::Activation,
    linear_2: Linear,
    window_kernel_size: [usize; 2],
    embed_dim: usize,
}

impl WindowAttentionMerger {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let ln_cfg = LayerNormConfig {
            eps: cfg.layer_norm_eps,
            ..Default::default()
        };
        let window_hidden = cfg.hidden_size * cfg.window_kernel_size[0] * cfg.window_kernel_size[1];
        let window_intermediate =
            cfg.intermediate_size * cfg.window_kernel_size[0] * cfg.window_kernel_size[1];
        Ok(Self {
            self_attn: VisionAttention::new(cfg, vb.pp("self_attn"))?,
            layer_norm1: layer_norm(cfg.hidden_size, ln_cfg, vb.pp("layer_norm1"))?,
            pre_norm: layer_norm(window_hidden, ln_cfg, vb.pp("pre_norm"))?,
            linear_1: linear(window_hidden, window_intermediate, vb.pp("linear_1"))?,
            act: candle_nn::Activation::GeluPytorchTanh,
            linear_2: linear(window_intermediate, cfg.hidden_size, vb.pp("linear_2"))?,
            window_kernel_size: cfg.window_kernel_size,
            embed_dim: cfg.hidden_size,
        })
    }

    fn forward(&self, xs: &Tensor, target_sizes: &[(usize, usize)]) -> Result<Tensor> {
        let residual = xs;
        let hidden = self.layer_norm1.forward(xs)?;
        let (window_index, cu_window_seqlens) =
            vision_window_index(target_sizes, 1, self.window_kernel_size[0], 1);
        let idx = Tensor::from_vec(
            window_index.iter().map(|v| *v as u32).collect::<Vec<_>>(),
            window_index.len(),
            xs.device(),
        )?;
        let hidden = hidden.index_select(&idx, 1)?;
        let hidden = self.self_attn.forward(&hidden, &cu_window_seqlens)?;
        let mut inverse = vec![0u32; window_index.len()];
        for (rank, &source) in window_index.iter().enumerate() {
            inverse[source] = rank as u32;
        }
        let inverse = Tensor::from_vec(inverse, window_index.len(), xs.device())?;
        let hidden = hidden.index_select(&inverse, 1)?;
        let hidden = (residual + hidden)?;

        let mut chunks = Vec::with_capacity(target_sizes.len());
        let mut start = 0usize;
        let window_h = self.window_kernel_size[0];
        let window_w = self.window_kernel_size[1];
        for &(height, width) in target_sizes {
            let num_patches = height * width;
            let merged_h = height / window_h;
            let merged_w = width / window_w;
            let patch = hidden.i((0, start..start + num_patches, ..))?;
            let patch_5d = patch
                .reshape((merged_h, window_h, merged_w, window_w, self.embed_dim))?
                .permute((0, 2, 1, 3, 4))?;
            let flat =
                patch_5d.reshape((merged_h * merged_w, window_h * window_w * self.embed_dim))?;
            let residual = patch_5d
                .reshape((merged_h * merged_w, window_h * window_w, self.embed_dim))?
                .mean(1)?;
            let merged = self.linear_2.forward(
                &self
                    .linear_1
                    .forward(&self.pre_norm.forward(&flat)?)?
                    .apply(&self.act)?,
            )?;
            chunks.push((merged + residual)?);
            start += num_patches;
        }
        let refs = chunks.iter().collect::<Vec<_>>();
        Tensor::cat(&refs, 0)?.unsqueeze(0)
    }
}

#[derive(Clone, Debug)]
struct VisionModel {
    embeddings: VisionEmbeddings,
    layers: Vec<VisionLayer>,
    post_layernorm: LayerNorm,
    vit_merger: WindowAttentionMerger,
    insert_layer_id: usize,
}

impl VisionModel {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("encoder").pp("layers");
        for idx in 0..cfg.num_hidden_layers {
            layers.push(VisionLayer::new(cfg, vb_layers.pp(idx))?);
        }
        let ln_cfg = LayerNormConfig {
            eps: cfg.layer_norm_eps,
            ..Default::default()
        };
        Ok(Self {
            embeddings: VisionEmbeddings::new(cfg, vb.pp("embeddings"))?,
            layers,
            post_layernorm: layer_norm(cfg.hidden_size, ln_cfg, vb.pp("post_layernorm"))?,
            vit_merger: WindowAttentionMerger::new(cfg, vb.pp("vit_merger"))?,
            insert_layer_id: cfg.insert_layer_id,
        })
    }

    fn forward(
        &self,
        pixel_values: &Tensor,
        target_sizes: &[(usize, usize)],
        use_vit_merger: bool,
    ) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let mut hidden = self.embeddings.forward(pixel_values, target_sizes)?;
        let mut sizes = target_sizes.to_vec();
        let mut cu_seqlens = cumulative_seqlens(&sizes);

        for (idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cu_seqlens)?;
            if use_vit_merger && idx == self.insert_layer_id {
                hidden = self.vit_merger.forward(&hidden, &sizes)?;
                sizes = sizes.iter().map(|(h, w)| (h / 2, w / 2)).collect();
                cu_seqlens = cumulative_seqlens(&sizes);
            }
        }
        Ok((self.post_layernorm.forward(&hidden)?, sizes))
    }
}

#[derive(Clone, Debug)]
struct DownsampleMlp {
    pre_norm: LayerNorm,
    linear_1: Linear,
    linear_2: Linear,
}

impl DownsampleMlp {
    fn new(input_dim: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        let merged_dim = input_dim * 4;
        let ln_cfg = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        Ok(Self {
            pre_norm: layer_norm(merged_dim, ln_cfg, vb.pp("pre_norm"))?,
            linear_1: linear(merged_dim, merged_dim, vb.pp("linear_1"))?,
            linear_2: linear(merged_dim, output_dim, vb.pp("linear_2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.pre_norm.forward(xs)?;
        let xs = self.linear_1.forward(&xs)?.gelu_erf()?;
        self.linear_2.forward(&xs)
    }
}

#[derive(Clone, Debug)]
struct MiniCpmMerger {
    mlps: Vec<DownsampleMlp>,
    merge_kernel_size: [usize; 2],
    merger_times: usize,
}

impl MiniCpmMerger {
    fn new(cfg: &MiniCpmConfig, vb: VarBuilder) -> Result<Self> {
        let mut mlps = Vec::with_capacity(cfg.merger_times);
        let vb_mlp = vb.pp("mlp");
        for idx in 0..cfg.merger_times {
            let out_dim = if idx + 1 == cfg.merger_times {
                cfg.text_config.hidden_size
            } else {
                cfg.vision_config.hidden_size
            };
            mlps.push(DownsampleMlp::new(
                cfg.vision_config.hidden_size,
                out_dim,
                vb_mlp.pp(idx),
            )?);
        }
        Ok(Self {
            mlps,
            merge_kernel_size: cfg.merge_kernel_size,
            merger_times: cfg.merger_times,
        })
    }

    fn forward(&self, hidden: &Tensor, target_sizes: &[(usize, usize)]) -> Result<Vec<Tensor>> {
        let mut start = 0usize;
        let mut processed = Vec::with_capacity(target_sizes.len());
        for &(mut height, mut width) in target_sizes {
            let num_patches = height * width;
            let mut state = hidden.i((0, start..start + num_patches, ..))?;
            let mut inner_dim = state.dim(D::Minus1)?;
            for idx in 0..self.merger_times {
                let merge_h = self.merge_kernel_size[0];
                let merge_w = self.merge_kernel_size[1];
                if height % merge_h != 0 || width % merge_w != 0 {
                    candle::bail!(
                        "vision target size ({height}, {width}) is not divisible by {:?}",
                        self.merge_kernel_size
                    );
                }
                let merged_h = height / merge_h;
                let merged_w = width / merge_w;
                state = state
                    .reshape((merged_h, merge_h, merged_w, merge_w, inner_dim))?
                    .permute((0, 2, 1, 3, 4))?
                    .reshape((merged_h * merged_w, merge_h * merge_w * inner_dim))?;
                state = self.mlps[idx].forward(&state)?;
                height = merged_h;
                width = merged_w;
                inner_dim = state.dim(D::Minus1)?;
            }
            processed.push(state);
            start += num_patches;
        }
        Ok(processed)
    }
}

#[derive(Clone, Debug)]
pub struct MiniCpmModel {
    vision_tower: VisionModel,
    pub language_model: Qwen35TextModel,
    merger: MiniCpmMerger,
}

impl MiniCpmModel {
    fn new(cfg: &MiniCpmConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            vision_tower: VisionModel::new(&cfg.vision_config, vb.pp("vision_tower"))?,
            language_model: Qwen35TextModel::new(&cfg.text_config, vb.pp("language_model"))?,
            merger: MiniCpmMerger::new(cfg, vb.pp("merger"))?,
        })
    }

    fn get_image_features(
        &self,
        images: &ProcessedImages,
        downsample_mode: &str,
        dtype: DType,
    ) -> Result<Vec<Tensor>> {
        let use_vit_merger = downsample_mode != "4x";
        let pixel_values = images.pixel_values.to_dtype(dtype)?;
        let (hidden, target_sizes) =
            self.vision_tower
                .forward(&pixel_values, &images.target_sizes, use_vit_merger)?;
        self.merger.forward(&hidden, &target_sizes)
    }

    fn apply_quantized_text_artifact(&mut self, artifact: &QuantizedTextArtifact) -> Result<usize> {
        self.language_model.apply_quantized_text_artifact(artifact)
    }
}

#[derive(Clone, Debug)]
pub struct MiniCpmForConditionalGeneration {
    model: MiniCpmModel,
    lm_head: crate::quantized_linear::MixedLinear,
    image_token_id: u32,
    device: Device,
    // EXPERIMENT: when the target head is restricted to a token subset, the
    // sliced head's argmax is an index into this table of global ids; None =
    // full 248k head (argmax is already a global id). See restrict_lm_head_vocab.
    head_vocab_ids: Option<Tensor>,
    /// FR-Spec draft head for the MTP path: (sliced+quantized head, global
    /// id map). Drafting argmaxes over the slice only; the target verifies
    /// full-vocab, so committed output is lossless — only draft cost and
    /// acceptance move.
    mtp_draft_head: Option<(crate::quantized_linear::MixedLinear, Tensor)>,
    // Transplanted Qwen3.5 MTP head (self-speculative drafting from
    // verify-pass hiddens); loaded on demand from the base checkpoint's
    // mtp.* tensors via load_mtp_head.
    mtp: Option<crate::qwen35::MtpHead>,
}

impl MiniCpmForConditionalGeneration {
    pub fn new(cfg: &MiniCpmConfig, vb: VarBuilder) -> Result<Self> {
        let model = MiniCpmModel::new(cfg, vb.pp("model"))?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            candle_nn::linear_no_bias(
                cfg.text_config.hidden_size,
                cfg.text_config.vocab_size,
                vb.pp("lm_head"),
            )?
        } else {
            Linear::new(model.language_model.embeddings().clone(), None)
        };
        Ok(Self {
            model,
            lm_head: crate::quantized_linear::MixedLinear::dense(lm_head),
            image_token_id: cfg.image_token_id,
            device: vb.device().clone(),
            head_vocab_ids: None,
            mtp: None,
            mtp_draft_head: None,
        })
    }

    /// Loads the transplanted MTP head from a base-checkpoint safetensors
    /// file carrying `mtp.*` tensors (the MiniCPM finetune ships none; the
    /// architecturally identical Qwen/Qwen3.5-0.8B does).
    pub fn load_mtp_head(&mut self, cfg: &MiniCpmConfig, weights: &std::path::Path) -> Result<()> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights.to_path_buf()],
                self.model.language_model.dtype(),
                &self.device,
            )?
        };
        self.mtp = Some(crate::qwen35::MtpHead::new(&cfg.text_config, vb)?);
        Ok(())
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Quantizes the loaded MTP head's dense linears (draft-side only; see
    /// MtpHead::quantize_with_pack).
    pub fn quantize_mtp_head_with_pack(
        &mut self,
        ggml: candle::quantized::GgmlDType,
        pack: Option<&crate::pack::PackStore>,
        key_prefix: &str,
    ) -> Result<()> {
        match self.mtp.as_mut() {
            Some(m) => m.quantize_with_pack(ggml, pack, key_prefix),
            None => Err(candle::Error::Msg("mtp head not loaded".to_string())),
        }
    }

    /// One MTP step over S (hidden, successor-token) pairs; returns the
    /// post-norm hidden [1, S, H]. The draft head runs separately via
    /// [`Self::mtp_draft_next`] on ONLY the rows a caller consumes — the
    /// old step ran the head over every row while both the chain and the
    /// catch-up read a single row's argmax (the m-row head pass was ~75%
    /// wasted at catch-up).
    pub fn mtp_step(&mut self, hidden: &Tensor, tokens: &Tensor) -> Result<Tensor> {
        let embeds = self.model.language_model.embed(tokens)?;
        let mtp = self
            .mtp
            .as_mut()
            .ok_or_else(|| candle::Error::Msg("mtp head not loaded".to_string()))?;
        mtp.step(hidden, &embeds)
    }

    /// Draft prediction for one post-norm row [1, 1, H]: sliced (or full)
    /// head forward + device argmax + global-id remap. Returns a device
    /// U32 [1, 1] token — no readback.
    pub fn mtp_draft_next(&mut self, post_row: &Tensor) -> Result<Tensor> {
        let logits = match &self.mtp_draft_head {
            Some((head, _)) => head.forward(post_row)?,
            None => self.lm_head.forward(post_row)?,
        };
        // Identical op chain to the pre-restructure path (byte-identity
        // gate depends on it): squeeze(1) -> [1, V] -> argmax -> [1].
        let local = logits.squeeze(1)?.argmax(candle::D::Minus1)?;
        let global = self.remap_mtp_draft_id(&local)?;
        global.reshape((1, 1))
    }

    /// FR-Spec slice for MTP drafting (mirror of restrict_lm_head_vocab, but
    /// as a separate head so verification stays full-vocab/lossless).
    pub fn restrict_mtp_draft_vocab(
        &mut self,
        ids: &[u32],
        ggml: Option<candle::quantized::GgmlDType>,
    ) -> Result<()> {
        let idx = Tensor::from_slice(ids, ids.len(), &self.device)?;
        let weight = self.model.language_model.embeddings().clone();
        let sliced = weight.index_select(&idx, 0)?.contiguous()?;
        let head = match ggml {
            Some(ggml) => {
                let cpu_f32 = sliced.to_dtype(candle::DType::F32)?.to_device(&Device::Cpu)?;
                let q = candle::quantized::QTensor::quantize_onto(&cpu_f32, ggml, &self.device)?;
                crate::quantized_linear::MixedLinear::from_qtensor(q)?
            }
            None => crate::quantized_linear::MixedLinear::dense(Linear::new(sliced, None)),
        };
        self.mtp_draft_head = Some((head, idx));
        Ok(())
    }

    /// Map an MTP draft-head argmax (slice-local, rank-1) to global token
    /// ids via a device gather; identity when no slice is configured.
    pub fn remap_mtp_draft_id(&self, sliced: &Tensor) -> Result<Tensor> {
        match &self.mtp_draft_head {
            Some((_, table)) => Ok(table.index_select(sliced, 0)?),
            None => Ok(sliced.clone()),
        }
    }

    /// Pairs currently in the MTP cache (its rope offset).
    pub fn mtp_kv_len(&self) -> usize {
        self.mtp.as_ref().map_or(0, |m| m.kv_len())
    }

    /// MTP rollback: keep the first `len` pairs.
    pub fn mtp_truncate(&mut self, len: usize) -> Result<()> {
        match self.mtp.as_mut() {
            Some(m) => m.truncate(len),
            None => Ok(()),
        }
    }

    pub fn mtp_clear(&mut self) {
        if let Some(m) = self.mtp.as_mut() {
            m.clear();
        }
    }

    /// Merged per-image vision features (vision tower + merger), for parity
    /// gates against the Transformers oracle.
    pub fn image_features(
        &self,
        images: &ProcessedImages,
        downsample_mode: &str,
        dtype: DType,
    ) -> Result<Vec<Tensor>> {
        self.model.get_image_features(images, downsample_mode, dtype)
    }

    /// Post-hoc lm_head quantization: replaces the tied-embedding dense head
    /// with a quantized copy (the BF16 embedding table stays for the
    /// gather). The 508 MB/token head read is 34% of all decode weight
    /// bytes; quality is advisory per campaign policy.
    pub fn quantize_lm_head(&mut self, ggml: candle::quantized::GgmlDType) -> Result<()> {
        self.quantize_lm_head_with_pack(ggml, None)
    }

    /// Pack-aware head quantization: a pack hit uploads the stored q blocks
    /// (bit-identical; same CPU quantizer); a miss quantizes and records the
    /// bytes for the pack write.
    pub fn quantize_lm_head_with_pack(
        &mut self,
        ggml: candle::quantized::GgmlDType,
        pack: Option<&crate::pack::PackStore>,
    ) -> Result<()> {
        if let Some(pack) = pack {
            if let Some(q) = pack.take("lm_head") {
                self.lm_head = crate::quantized_linear::MixedLinear::from_qtensor(q)?;
                return Ok(());
            }
        }
        let weight = self.model.language_model.embeddings().clone();
        let cpu_f32 = weight
            .to_dtype(candle::DType::F32)?
            .to_device(&Device::Cpu)?;
        let q = match pack {
            Some(pack) => pack
                .quantize_and_record("lm_head", &cpu_f32, ggml)
                .map_err(|e| candle::Error::Msg(format!("pack lm_head: {e:#}")))?,
            None => candle::quantized::QTensor::quantize_onto(&cpu_f32, ggml, &self.device)?,
        };
        self.lm_head = crate::quantized_linear::MixedLinear::from_qtensor(q)?;
        Ok(())
    }

    /// EXPERIMENT (quality trade): restrict the target head to `ids` (global
    /// token ids, most-frequent first with control tokens pinned). The head
    /// weight is sliced to those rows and optionally quantized, so it reads
    /// ~len(ids)/vocab of the bytes; the forward then emits logits over
    /// len(ids) columns and argmax gives a sliced index, remapped back to a
    /// global id by [`Self::remap_head_id`]. Out-of-set argmaxes become a
    /// different in-set token — this changes committed outputs.
    pub fn restrict_lm_head_vocab(
        &mut self,
        ids: &[u32],
        ggml: Option<candle::quantized::GgmlDType>,
    ) -> Result<()> {
        let idx = Tensor::from_slice(ids, ids.len(), &self.device)?;
        let weight = self.model.language_model.embeddings().clone();
        let sliced = weight.index_select(&idx, 0)?.contiguous()?;
        self.lm_head = match ggml {
            Some(ggml) => {
                let cpu_f32 = sliced.to_dtype(candle::DType::F32)?.to_device(&Device::Cpu)?;
                let q = candle::quantized::QTensor::quantize_onto(&cpu_f32, ggml, &self.device)?;
                crate::quantized_linear::MixedLinear::from_qtensor(q)?
            }
            None => crate::quantized_linear::MixedLinear::dense(Linear::new(sliced, None)),
        };
        self.head_vocab_ids = Some(idx);
        Ok(())
    }

    /// Map a restricted-head argmax (index into the vocab subset) back to a
    /// global token id via a device-side gather. Identity (clone) when the
    /// head is unrestricted. `sliced` and the result are rank-1 (the greedy
    /// device chain's argmax shape).
    pub fn remap_head_id(&self, sliced: &Tensor) -> Result<Tensor> {
        match &self.head_vocab_ids {
            Some(table) => Ok(table.index_select(sliced, 0)?),
            None => Ok(sliced.clone()),
        }
    }

    /// Host-side single-id remap for the non-device-chain sampling path.
    pub fn remap_head_id_host(&self, sliced: u32) -> Result<u32> {
        match &self.head_vocab_ids {
            Some(table) => {
                let g = table
                    .to_dtype(DType::U32)?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<u32>()?;
                Ok(*g.get(sliced as usize).unwrap_or(&sliced))
            }
            None => Ok(sliced),
        }
    }

    pub fn head_vocab_size(&self) -> Option<usize> {
        self.head_vocab_ids.as_ref().map(|t| t.elem_count())
    }

    pub fn clear_cache(&mut self) {
        self.model.language_model.clear_cache();
    }

    pub fn set_device_capture(&mut self, layers: Option<Vec<usize>>) {
        self.model.language_model.set_device_capture(layers);
    }

    pub fn take_device_captures(&mut self) -> Vec<Tensor> {
        self.model.language_model.take_device_captures()
    }

    pub fn snapshot_decode_state(&self) -> crate::qwen35::DecodeStateSnapshot {
        self.model.language_model.snapshot_decode_state()
    }

    pub fn restore_decode_state(
        &mut self,
        snapshot: &crate::qwen35::DecodeStateSnapshot,
    ) -> Result<()> {
        self.model.language_model.restore_decode_state(snapshot)
    }

    pub fn set_verify_state_capture(&mut self, on: bool) {
        self.model.language_model.set_verify_state_capture(on);
    }

    pub fn rollback_to_prefix(
        &mut self,
        snapshot: &crate::qwen35::DecodeStateSnapshot,
        prefix_len: usize,
    ) -> Result<()> {
        self.model
            .language_model
            .rollback_to_prefix(snapshot, prefix_len)
    }

    pub fn set_text_profiler(&mut self, profiler: Option<Qwen35Profiler>) {
        self.model.language_model.set_profiler(profiler);
    }

    pub fn set_text_trace_recorder(&mut self, trace_recorder: Option<Qwen35TraceRecorder>) {
        self.model.language_model.set_trace_recorder(trace_recorder);
    }

    pub fn apply_quantized_text_artifact(
        &mut self,
        artifact: &QuantizedTextArtifact,
    ) -> Result<usize> {
        self.model.apply_quantized_text_artifact(artifact)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        images: Option<&ProcessedImages>,
        downsample_mode: &str,
        offset: usize,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        // Narrow the hidden state to the last position BEFORE lm_head: the
        // head is an L x 248094 x 1024 matmul, so projecting every prefill
        // position only to discard all but one wasted ~0.5 TFLOP and a
        // ~500 MB logits buffer on a 1k-token prompt. Callers needing dense
        // logits (spec verification) use forward_all_logits.
        let hidden = self
            .forward_hidden(input_ids, images, downsample_mode, offset)?
            .narrow(1, seq_len - 1, 1)?
            .contiguous()?;
        self.lm_head.forward(&hidden)?.squeeze(1)
    }

    pub fn forward_all_logits(
        &mut self,
        input_ids: &Tensor,
        images: Option<&ProcessedImages>,
        downsample_mode: &str,
        offset: usize,
    ) -> Result<Tensor> {
        let hidden = self.forward_hidden(input_ids, images, downsample_mode, offset)?;
        self.lm_head.forward(&hidden)
    }

    /// Whether [`Self::forward_argmax`] is available: a Metal-resident q4_K
    /// quantized head (the deployed configuration).
    pub fn supports_fused_head_argmax(&self) -> bool {
        self.device.is_metal()
            && self
                .lm_head
                .qtensor()
                .is_some_and(|q| q.dtype() == candle::quantized::GgmlDType::Q4K)
    }

    /// Greedy fused head: forward to the last position's hidden state, then
    /// the fused GEMV->argmax pair — the 248k-row logits are never
    /// materialized. Returns the [1] U32 argmax id (feed remap_head_id, same
    /// as `logits.argmax(D::Minus1)` on the stored-logits path; exact by
    /// construction, see fused_head.rs).
    pub fn forward_argmax(
        &mut self,
        input_ids: &Tensor,
        images: Option<&ProcessedImages>,
        downsample_mode: &str,
        offset: usize,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let hidden = self
            .forward_hidden(input_ids, images, downsample_mode, offset)?
            .narrow(1, seq_len - 1, 1)?
            .contiguous()?;
        let head = self
            .lm_head
            .qtensor()
            .ok_or_else(|| candle::Error::Msg("fused head argmax needs a quantized head".into()))?;
        crate::fused_head::q4k_head_argmax(&hidden, head).map_err(candle::Error::wrap)
    }

    /// Dense logits plus the post-final-norm hidden states — the tensor MTP
    /// drafting feeds on (each accepted position's hidden pairs with its
    /// successor token in the head's cache).
    pub fn forward_all_logits_and_hidden(
        &mut self,
        input_ids: &Tensor,
        images: Option<&ProcessedImages>,
        downsample_mode: &str,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let hidden = self.forward_hidden(input_ids, images, downsample_mode, offset)?;
        Ok((self.lm_head.forward(&hidden)?, hidden))
    }

    /// Verify forward returning per-row argmax ids (device U32 [C]) and the
    /// hidden — via the fused mm2d head-argmax when available (the C x V
    /// logits tensor is never materialized; bf16-rounded compares keep it
    /// byte-identical to head-forward + fast_argmax), else the plain pair.
    pub fn forward_verify_ids_and_hidden(
        &mut self,
        input_ids: &Tensor,
        images: Option<&ProcessedImages>,
        downsample_mode: &str,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let hidden = self.forward_hidden(input_ids, images, downsample_mode, offset)?;
        if let Some(ids) = crate::mm2d::mm2d_head_argmax(&hidden, &self.lm_head)
            .map_err(|e| candle::Error::Msg(format!("fused verify argmax: {e:#}")))?
        {
            return Ok((ids, hidden));
        }
        let ids = self
            .lm_head
            .forward(&hidden)?
            .squeeze(0)?
            .argmax(candle::D::Minus1)?;
        Ok((ids, hidden))
    }

    /// Tree verify forward: `input_ids` is the flattened
    /// [anchor, a_1..a_w, b_1..b_w] chunk (shape [1, 1 + 2w]). Returns dense
    /// logits for every row. Commit the winner with [`Self::rollback_tree`].
    pub fn forward_tree_all_logits(
        &mut self,
        input_ids: &Tensor,
        offset: usize,
        branch_width: usize,
    ) -> Result<Tensor> {
        let embeds = self.model.language_model.embed(input_ids)?;
        let hidden = self
            .model
            .language_model
            .forward_tree_embeds(&embeds, offset, branch_width)?;
        self.lm_head.forward(&hidden)
    }

    pub fn rollback_tree(
        &mut self,
        snapshot: &crate::qwen35::DecodeStateSnapshot,
        branch_width: usize,
        on_alt: bool,
        accepted: usize,
    ) -> Result<()> {
        self.model
            .language_model
            .rollback_tree(snapshot, branch_width, on_alt, accepted)
    }

    fn forward_hidden(
        &mut self,
        input_ids: &Tensor,
        images: Option<&ProcessedImages>,
        downsample_mode: &str,
        offset: usize,
    ) -> Result<Tensor> {
        let hidden = if let Some(images) = images {
            if offset != 0 {
                candle::bail!("image features can only be supplied during prefill");
            }
            let mut embeds = self.model.language_model.embed(input_ids)?;
            let image_features =
                self.model
                    .get_image_features(images, downsample_mode, embeds.dtype())?;
            let refs = image_features.iter().collect::<Vec<_>>();
            let image_features = Tensor::cat(&refs, 0)?;
            embeds = self.replace_image_embeddings(input_ids, &embeds, &image_features)?;
            self.model.language_model.forward_embeds(&embeds, offset)?
        } else {
            self.model.language_model.forward_ids(input_ids, offset)?
        };
        Ok(hidden)
    }

    fn replace_image_embeddings(
        &self,
        input_ids: &Tensor,
        inputs_embeds: &Tensor,
        image_features: &Tensor,
    ) -> Result<Tensor> {
        let ids = input_ids
            .to_device(&Device::Cpu)?
            .to_vec2::<u32>()?
            .into_iter()
            .next()
            .unwrap_or_default();
        let positions = ids
            .iter()
            .enumerate()
            .filter_map(|(idx, id)| (*id == self.image_token_id).then_some(idx))
            .collect::<Vec<_>>();
        if positions.len() != image_features.dim(0)? {
            candle::bail!(
                "image placeholder count {} does not match visual feature count {}",
                positions.len(),
                image_features.dim(0)?
            );
        }
        let hidden = inputs_embeds.dim(D::Minus1)?;
        let mut out = inputs_embeds.clone();
        for (feature_idx, token_idx) in positions.into_iter().enumerate() {
            let feature = image_features
                .narrow(0, feature_idx, 1)?
                .reshape((1, 1, hidden))?
                .to_device(&self.device)?
                .to_dtype(inputs_embeds.dtype())?;
            out = out.slice_assign(&[0..1, token_idx..token_idx + 1, 0..hidden], &feature)?;
        }
        Ok(out)
    }
}

fn cumulative_seqlens(target_sizes: &[(usize, usize)]) -> Vec<usize> {
    let mut out = Vec::with_capacity(target_sizes.len() + 1);
    out.push(0);
    for &(h, w) in target_sizes {
        out.push(out.last().copied().unwrap_or(0) + h * w);
    }
    out
}

fn nearest_position_ids(target_sizes: &[(usize, usize)], side: usize) -> Vec<u32> {
    let boundaries = (1..side)
        .map(|idx| idx as f64 / side as f64)
        .collect::<Vec<_>>();
    let mut ids = Vec::new();
    for &(height, width) in target_sizes {
        for h in 0..height {
            let h_coord = h as f64 / height as f64;
            let bucket_h = boundaries.partition_point(|boundary| *boundary <= h_coord);
            for w in 0..width {
                let w_coord = w as f64 / width as f64;
                let bucket_w = boundaries.partition_point(|boundary| *boundary <= w_coord);
                ids.push((bucket_h * side + bucket_w) as u32);
            }
        }
    }
    ids
}

fn vision_window_index(
    target_sizes: &[(usize, usize)],
    spatial_merge_size: usize,
    window_size: usize,
    patch_size: usize,
) -> (Vec<usize>, Vec<usize>) {
    let vit_merger_window_size = window_size / spatial_merge_size / patch_size;
    let spatial_merge_unit = spatial_merge_size * spatial_merge_size;
    let mut window_index = Vec::new();
    let mut cu_window_seqlens = vec![0usize];
    let mut window_index_id = 0usize;

    for &(grid_h, grid_w) in target_sizes {
        let llm_grid_h = grid_h / spatial_merge_size;
        let llm_grid_w = grid_w / spatial_merge_size;
        let pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size;
        let pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size;
        let padded_h = llm_grid_h + pad_h;
        let padded_w = llm_grid_w + pad_w;
        let num_windows_h = padded_h / vit_merger_window_size;
        let num_windows_w = padded_w / vit_merger_window_size;

        for wh in 0..num_windows_h {
            for ww in 0..num_windows_w {
                let before = window_index.len();
                for ih in 0..vit_merger_window_size {
                    for iw in 0..vit_merger_window_size {
                        let h = wh * vit_merger_window_size + ih;
                        let w = ww * vit_merger_window_size + iw;
                        if h < llm_grid_h && w < llm_grid_w {
                            window_index.push(window_index_id + h * llm_grid_w + w);
                        }
                    }
                }
                let seqlen = window_index.len() - before;
                let next =
                    cu_window_seqlens.last().copied().unwrap_or(0) + seqlen * spatial_merge_unit;
                if cu_window_seqlens.last().copied() != Some(next) {
                    cu_window_seqlens.push(next);
                }
            }
        }
        window_index_id += llm_grid_h * llm_grid_w;
    }
    (window_index, cu_window_seqlens)
}
