use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{
    conv2d, layer_norm, linear, Conv2dConfig, LayerNorm, LayerNormConfig, Linear, VarBuilder,
};

use crate::{
    config::{MiniCpmConfig, VisionConfig},
    image_processor::ProcessedImages,
    qwen35::{Qwen35Profiler, Qwen35TextModel},
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
            let q = q.narrow(2, start, len)?;
            let k = k.narrow(2, start, len)?;
            let v = v.narrow(2, start, len)?;
            let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            let attn = candle_nn::ops::softmax_last_dim(&attn.to_dtype(DType::F32)?)?
                .to_dtype(xs.dtype())?;
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
        Ok(Tensor::cat(&refs, 0)?.unsqueeze(0)?)
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
    ) -> Result<Vec<Tensor>> {
        let use_vit_merger = downsample_mode != "4x";
        let (hidden, target_sizes) = self.vision_tower.forward(
            &images.pixel_values,
            &images.target_sizes,
            use_vit_merger,
        )?;
        self.merger.forward(&hidden, &target_sizes)
    }
}

#[derive(Clone, Debug)]
pub struct MiniCpmForConditionalGeneration {
    model: MiniCpmModel,
    lm_head: Linear,
    image_token_id: u32,
    device: Device,
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
            lm_head,
            image_token_id: cfg.image_token_id,
            device: vb.device().clone(),
        })
    }

    pub fn clear_cache(&mut self) {
        self.model.language_model.clear_cache();
    }

    pub fn set_text_profiler(&mut self, profiler: Option<Qwen35Profiler>) {
        self.model.language_model.set_profiler(profiler);
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        images: Option<&ProcessedImages>,
        downsample_mode: &str,
        offset: usize,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let hidden = if let Some(images) = images {
            if offset != 0 {
                candle::bail!("image features can only be supplied during prefill");
            }
            let mut embeds = self.model.language_model.embed(input_ids)?;
            let image_features = self.model.get_image_features(images, downsample_mode)?;
            let refs = image_features.iter().collect::<Vec<_>>();
            let image_features = Tensor::cat(&refs, 0)?;
            embeds = self.replace_image_embeddings(input_ids, &embeds, &image_features)?;
            self.model.language_model.forward_embeds(&embeds, offset)?
        } else {
            self.model.language_model.forward_ids(input_ids, offset)?
        };
        hidden
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)?
            .squeeze(1)
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
