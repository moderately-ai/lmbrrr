use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{embedding, linear_b, linear_no_bias, Activation, Embedding, VarBuilder};
use candle_transformers::utils::repeat_kv;
use serde::Serialize;

use crate::config::{LayerType, TextConfig};
use crate::quantized_linear::{MixedLinear, QuantizedTextArtifact};

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35ProfileEvent {
    pub component: String,
    pub layer_index: Option<usize>,
    pub layer_kind: Option<String>,
    pub seq_len: usize,
    pub offset: usize,
    pub seconds: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Qwen35Profiler {
    events: Arc<Mutex<Vec<Qwen35ProfileEvent>>>,
}

impl Qwen35Profiler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&self) {
        self.events
            .lock()
            .expect("Qwen35 profiler lock poisoned")
            .clear();
    }

    pub fn events(&self) -> Vec<Qwen35ProfileEvent> {
        self.events
            .lock()
            .expect("Qwen35 profiler lock poisoned")
            .clone()
    }

    fn record(
        &self,
        component: &'static str,
        layer_index: Option<usize>,
        layer_kind: Option<&'static str>,
        seq_len: usize,
        offset: usize,
        seconds: f64,
    ) {
        self.events
            .lock()
            .expect("Qwen35 profiler lock poisoned")
            .push(Qwen35ProfileEvent {
                component: component.to_string(),
                layer_index,
                layer_kind: layer_kind.map(str::to_string),
                seq_len,
                offset,
                seconds,
            });
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen35HiddenStateTrace {
    pub layer_index: usize,
    pub layer_kind: String,
    pub seq_len: usize,
    pub offset: usize,
    pub position: usize,
    pub hidden_size: usize,
    pub dtype: String,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Qwen35TraceRecorder {
    selected_layers: Arc<Vec<usize>>,
    records: Arc<Mutex<Vec<Qwen35HiddenStateTrace>>>,
}

impl Qwen35TraceRecorder {
    pub fn new(mut selected_layers: Vec<usize>) -> Self {
        selected_layers.sort_unstable();
        selected_layers.dedup();
        Self {
            selected_layers: Arc::new(selected_layers),
            records: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn selected_layers(&self) -> Vec<usize> {
        self.selected_layers.as_ref().clone()
    }

    pub fn clear(&self) {
        self.records
            .lock()
            .expect("Qwen35 trace lock poisoned")
            .clear();
    }

    pub fn take(&self) -> Vec<Qwen35HiddenStateTrace> {
        std::mem::take(&mut *self.records.lock().expect("Qwen35 trace lock poisoned"))
    }

    fn record_layer(
        &self,
        layer_index: usize,
        layer_kind: &'static str,
        hidden: &Tensor,
        offset: usize,
    ) -> Result<()> {
        if !self.selected_layers.contains(&layer_index) {
            return Ok(());
        }

        let (batch, seq_len, hidden_size) = hidden.dims3()?;
        if batch != 1 {
            candle::bail!("hidden-state tracing expects batch size 1, got {batch}");
        }
        let values = hidden
            .narrow(1, seq_len - 1, 1)?
            .squeeze(1)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        self.records
            .lock()
            .expect("Qwen35 trace lock poisoned")
            .push(Qwen35HiddenStateTrace {
                layer_index,
                layer_kind: layer_kind.to_string(),
                seq_len,
                offset,
                position: offset + seq_len - 1,
                hidden_size,
                dtype: format!("{:?}", hidden.dtype()),
                values,
            });
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Qwen35RmsNorm {
    weight: Tensor,
    eps: f64,
    zero_centered: bool,
}

impl Qwen35RmsNorm {
    fn new(size: usize, eps: f64, zero_centered: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            eps,
            zero_centered,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let inv = (variance + self.eps)?.powf(-0.5)?;
        let ys = xs_f32.broadcast_mul(&inv)?;
        let weight = self.weight.to_dtype(DType::F32)?;
        let weight = if self.zero_centered {
            (weight + 1.0)?
        } else {
            weight
        };
        ys.broadcast_mul(&weight)?.to_dtype(dtype)
    }
}

#[derive(Clone, Debug)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    rotary_dim: usize,
}

impl RotaryEmbedding {
    fn new(cfg: &TextConfig, dtype: DType, device: &Device) -> Result<Self> {
        let rotary_dim = (cfg.head_dim as f64 * cfg.rope_parameters.partial_rotary_factor) as usize;
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| {
                1f32 / (cfg.rope_parameters.rope_theta as f32).powf(i as f32 / rotary_dim as f32)
            })
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, rotary_dim / 2), device)?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
            rotary_dim,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, head_dim) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_rot = q.narrow(D::Minus1, 0, self.rotary_dim)?;
        let k_rot = k.narrow(D::Minus1, 0, self.rotary_dim)?;
        let q_rot = candle_nn::rotary_emb::rope(&q_rot.contiguous()?, &cos, &sin)?;
        let k_rot = candle_nn::rotary_emb::rope(&k_rot.contiguous()?, &cos, &sin)?;
        if self.rotary_dim == head_dim {
            Ok((q_rot, k_rot))
        } else {
            let q_pass = q.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
            let k_pass = k.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
            Ok((
                Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?,
                Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?,
            ))
        }
    }
}

#[derive(Clone, Debug)]
struct Mlp {
    gate_proj: MixedLinear,
    up_proj: MixedLinear,
    down_proj: MixedLinear,
    act: Activation,
}

impl Mlp {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: MixedLinear::dense(linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?),
            up_proj: MixedLinear::dense(linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj"),
            )?),
            down_proj: MixedLinear::dense(linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?),
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(xs)?.apply(&self.act)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }

    fn apply_quantized_text_artifact(
        &mut self,
        layer_index: usize,
        artifact: &QuantizedTextArtifact,
    ) -> Result<usize> {
        let prefix = format!("model.language_model.layers.{layer_index}.mlp");
        let mut replaced = 0usize;
        replaced += replace_quantized_linear(
            &mut self.gate_proj,
            &format!("{prefix}.gate_proj.weight"),
            artifact,
        )?;
        replaced += replace_quantized_linear(
            &mut self.up_proj,
            &format!("{prefix}.up_proj.weight"),
            artifact,
        )?;
        replaced += replace_quantized_linear(
            &mut self.down_proj,
            &format!("{prefix}.down_proj.weight"),
            artifact,
        )?;
        Ok(replaced)
    }
}

#[derive(Clone, Debug)]
struct FullAttention {
    q_proj: MixedLinear,
    k_proj: MixedLinear,
    v_proj: MixedLinear,
    o_proj: MixedLinear,
    q_norm: Qwen35RmsNorm,
    k_norm: Qwen35RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    num_kv_groups: usize,
    rotary: Arc<RotaryEmbedding>,
    kv_cache: Arc<Mutex<candle_nn::kv_cache::KvCache>>,
}

impl FullAttention {
    fn new(cfg: &TextConfig, rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: MixedLinear::dense(linear_b(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim * 2,
                cfg.attention_bias,
                vb.pp("q_proj"),
            )?),
            k_proj: MixedLinear::dense(linear_b(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                cfg.attention_bias,
                vb.pp("k_proj"),
            )?),
            v_proj: MixedLinear::dense(linear_b(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                cfg.attention_bias,
                vb.pp("v_proj"),
            )?),
            o_proj: MixedLinear::dense(linear_b(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                cfg.attention_bias,
                vb.pp("o_proj"),
            )?),
            q_norm: Qwen35RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, true, vb.pp("q_norm"))?,
            k_norm: Qwen35RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, true, vb.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            hidden_size: cfg.num_attention_heads * cfg.head_dim,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            rotary,
            kv_cache: Arc::new(Mutex::new(candle_nn::kv_cache::KvCache::new(
                2,
                cfg.max_position_embeddings,
            ))),
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
        layer_index: usize,
        profiler: Option<&Qwen35Profiler>,
    ) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let device = xs.device().clone();
        let q_gate = profiled(
            profiler,
            &device,
            Some(layer_index),
            Some("full_attention"),
            "full_attention_q_gate_projection",
            l,
            offset,
            || {
                self.q_proj
                    .forward(xs)?
                    .reshape((b, l, self.num_heads, self.head_dim * 2))
            },
        )?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((b, l, self.hidden_size))?;
        let (mut q, mut k, v) = profiled(
            profiler,
            &device,
            Some(layer_index),
            Some("full_attention"),
            "full_attention_kv_projection_norm",
            l,
            offset,
            || {
                let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
                let k = self
                    .k_norm
                    .forward(&self.k_proj.forward(xs)?.reshape((
                        b,
                        l,
                        self.num_kv_heads,
                        self.head_dim,
                    ))?)?
                    .transpose(1, 2)?;
                let v = self
                    .v_proj
                    .forward(xs)?
                    .reshape((b, l, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?;
                Ok((q, k, v))
            },
        )?;

        let (q, _k, v, k_t) = profiled(
            profiler,
            &device,
            Some(layer_index),
            Some("full_attention"),
            "full_attention_rotary_kv_cache",
            l,
            offset,
            || {
                (q, k) = self.rotary.apply(&q, &k, offset)?;
                let (k, v) = self
                    .kv_cache
                    .lock()
                    .expect("full-attention KV cache lock poisoned")
                    .append(&k.contiguous()?, &v.contiguous()?)?;
                let q = q.contiguous()?;
                let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
                let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;
                let k_t = k.transpose(2, 3)?.contiguous()?;
                Ok((q, k, v, k_t))
            },
        )?;

        let out = profiled(
            profiler,
            &device,
            Some(layer_index),
            Some("full_attention"),
            "full_attention_matmul_softmax",
            l,
            offset,
            || {
                let scale = 1.0 / (self.head_dim as f64).sqrt();
                let mut attn = (q.matmul(&k_t)? * scale)?;
                if let Some(mask) = mask {
                    attn = attn.broadcast_add(mask)?;
                }
                let attn = candle_nn::ops::softmax_last_dim(&attn.to_dtype(DType::F32)?)?
                    .to_dtype(xs.dtype())?;
                attn.contiguous()?
                    .matmul(&v)?
                    .transpose(1, 2)?
                    .reshape((b, l, self.hidden_size))
            },
        )?;
        profiled(
            profiler,
            &device,
            Some(layer_index),
            Some("full_attention"),
            "full_attention_output_projection",
            l,
            offset,
            || {
                let out = (out * candle_nn::ops::sigmoid(&gate)?)?;
                self.o_proj.forward(&out)
            },
        )
    }

    fn clear_cache(&self) {
        self.kv_cache
            .lock()
            .expect("full-attention KV cache lock poisoned")
            .reset();
    }

    fn apply_quantized_text_artifact(
        &mut self,
        layer_index: usize,
        artifact: &QuantizedTextArtifact,
    ) -> Result<usize> {
        let prefix = format!("model.language_model.layers.{layer_index}.self_attn");
        let mut replaced = 0usize;
        replaced += replace_quantized_linear(
            &mut self.q_proj,
            &format!("{prefix}.q_proj.weight"),
            artifact,
        )?;
        replaced += replace_quantized_linear(
            &mut self.k_proj,
            &format!("{prefix}.k_proj.weight"),
            artifact,
        )?;
        replaced += replace_quantized_linear(
            &mut self.v_proj,
            &format!("{prefix}.v_proj.weight"),
            artifact,
        )?;
        replaced += replace_quantized_linear(
            &mut self.o_proj,
            &format!("{prefix}.o_proj.weight"),
            artifact,
        )?;
        Ok(replaced)
    }
}

#[derive(Clone, Debug)]
struct GatedDeltaNet {
    in_proj_qkv: MixedLinear,
    in_proj_z: MixedLinear,
    in_proj_b: MixedLinear,
    in_proj_a: MixedLinear,
    out_proj: MixedLinear,
    conv_weight: Tensor,
    dt_bias_f32: Tensor,
    a_log_exp_f32: Tensor,
    norm: Qwen35RmsNorm,
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
    conv_kernel_size: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
}

impl GatedDeltaNet {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
        let value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let dt_bias = vb.get(cfg.linear_num_value_heads, "dt_bias")?;
        let dt_bias_f32 =
            dt_bias
                .to_dtype(DType::F32)?
                .reshape((1, 1, cfg.linear_num_value_heads))?;
        let a_log = vb.get(cfg.linear_num_value_heads, "A_log")?;
        let a_log_exp_f32 =
            a_log
                .to_dtype(DType::F32)?
                .exp()?
                .reshape((1, 1, cfg.linear_num_value_heads))?;
        Ok(Self {
            in_proj_qkv: MixedLinear::dense(linear_no_bias(
                cfg.hidden_size,
                conv_dim,
                vb.pp("in_proj_qkv"),
            )?),
            in_proj_z: MixedLinear::dense(linear_no_bias(
                cfg.hidden_size,
                value_dim,
                vb.pp("in_proj_z"),
            )?),
            in_proj_b: MixedLinear::dense(linear_no_bias(
                cfg.hidden_size,
                cfg.linear_num_value_heads,
                vb.pp("in_proj_b"),
            )?),
            in_proj_a: MixedLinear::dense(linear_no_bias(
                cfg.hidden_size,
                cfg.linear_num_value_heads,
                vb.pp("in_proj_a"),
            )?),
            out_proj: MixedLinear::dense(linear_no_bias(
                value_dim,
                cfg.hidden_size,
                vb.pp("out_proj"),
            )?),
            conv_weight: vb.get((conv_dim, 1, cfg.linear_conv_kernel_dim), "conv1d.weight")?,
            dt_bias_f32,
            a_log_exp_f32,
            norm: Qwen35RmsNorm::new(
                cfg.linear_value_head_dim,
                cfg.rms_norm_eps,
                false,
                vb.pp("norm"),
            )?,
            conv_state: None,
            recurrent_state: None,
            conv_kernel_size: cfg.linear_conv_kernel_dim,
            num_k_heads: cfg.linear_num_key_heads,
            num_v_heads: cfg.linear_num_value_heads,
            head_k_dim: cfg.linear_key_head_dim,
            head_v_dim: cfg.linear_value_head_dim,
            key_dim,
            value_dim,
            conv_dim,
        })
    }

    fn clear_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }

    fn apply_quantized_text_artifact(
        &mut self,
        layer_index: usize,
        artifact: &QuantizedTextArtifact,
    ) -> Result<usize> {
        let prefix = format!("model.language_model.layers.{layer_index}.linear_attn");
        let mut replaced = 0usize;
        replaced += replace_quantized_linear(
            &mut self.in_proj_qkv,
            &format!("{prefix}.in_proj_qkv.weight"),
            artifact,
        )?;
        replaced += replace_quantized_linear(
            &mut self.in_proj_z,
            &format!("{prefix}.in_proj_z.weight"),
            artifact,
        )?;
        replaced += replace_quantized_linear(
            &mut self.in_proj_b,
            &format!("{prefix}.in_proj_b.weight"),
            artifact,
        )?;
        replaced += replace_quantized_linear(
            &mut self.in_proj_a,
            &format!("{prefix}.in_proj_a.weight"),
            artifact,
        )?;
        replaced += replace_quantized_linear(
            &mut self.out_proj,
            &format!("{prefix}.out_proj.weight"),
            artifact,
        )?;
        Ok(replaced)
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        layer_index: usize,
        offset: usize,
        profiler: Option<&Qwen35Profiler>,
    ) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let device = xs.device().clone();
        let mixed = profiled(
            profiler,
            &device,
            Some(layer_index),
            Some("linear_attention"),
            "deltanet_qkv_projection",
            l,
            offset,
            || self.in_proj_qkv.forward(xs)?.transpose(1, 2),
        )?;
        let mixed = profiled(
            profiler,
            &device,
            Some(layer_index),
            Some("linear_attention"),
            "deltanet_depthwise_conv",
            l,
            offset,
            || self.depthwise_causal_conv(&mixed)?.transpose(1, 2),
        )?;

        let query = mixed.narrow(D::Minus1, 0, self.key_dim)?.reshape((
            b,
            l,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let key = mixed
            .narrow(D::Minus1, self.key_dim, self.key_dim)?
            .reshape((b, l, self.num_k_heads, self.head_k_dim))?;
        let value = mixed
            .narrow(D::Minus1, self.key_dim * 2, self.value_dim)?
            .reshape((b, l, self.num_v_heads, self.head_v_dim))?;
        let (query, key, beta, g) = profiled(
            profiler,
            &device,
            Some(layer_index),
            Some("linear_attention"),
            "deltanet_gates_and_repeat",
            l,
            offset,
            || {
                let beta = candle_nn::ops::sigmoid(&self.in_proj_b.forward(xs)?)?;
                let a = self.in_proj_a.forward(xs)?.to_dtype(DType::F32)?;
                let g = (a.broadcast_add(&self.dt_bias_f32)?.exp()? + 1.0)?
                    .log()?
                    .broadcast_mul(&self.a_log_exp_f32)?
                    .neg()?;
                let query = maybe_repeat_heads(query, self.num_v_heads / self.num_k_heads)?;
                let key = maybe_repeat_heads(key, self.num_v_heads / self.num_k_heads)?;
                Ok((query, key, beta, g))
            },
        )?;
        let core = profiled(
            profiler,
            &device,
            Some(layer_index),
            Some("linear_attention"),
            "deltanet_recurrent_rule",
            l,
            offset,
            || self.recurrent_delta_rule(&query, &key, &value, &g, &beta),
        )?;
        profiled(
            profiler,
            &device,
            Some(layer_index),
            Some("linear_attention"),
            "deltanet_output_gate_norm_projection",
            l,
            offset,
            || {
                let z = self.in_proj_z.forward(xs)?.reshape((
                    b,
                    l,
                    self.num_v_heads,
                    self.head_v_dim,
                ))?;
                let core = core.reshape((b * l * self.num_v_heads, self.head_v_dim))?;
                let z = z.reshape((b * l * self.num_v_heads, self.head_v_dim))?;
                let core = self.norm.forward(&core)?;
                let gated = (core * z.to_dtype(DType::F32)?.silu()?.to_dtype(z.dtype())?)?;
                let gated = gated.reshape((b, l, self.value_dim))?;
                self.out_proj.forward(&gated)
            },
        )
    }

    fn depthwise_causal_conv(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, l) = xs.dims3()?;
        if c != self.conv_dim {
            candle::bail!("DeltaNet conv input dim {c} != {}", self.conv_dim);
        }
        let full = match &self.conv_state {
            Some(state) => Tensor::cat(&[state, xs], 2)?,
            None => xs.clone(),
        };
        let full_len = full.dim(2)?;
        let start = full_len - l;
        let weight = self.conv_weight.squeeze(1)?;
        let mut outs = Vec::with_capacity(l);
        for t in start..start + l {
            let mut acc = Tensor::zeros((b, c), xs.dtype(), xs.device())?;
            for k in 0..self.conv_kernel_size {
                let left = self.conv_kernel_size - 1 - k;
                if t >= left {
                    let src_pos = t - left;
                    let value = full.narrow(2, src_pos, 1)?.squeeze(2)?;
                    let w = weight.narrow(1, k, 1)?.reshape((1, c))?;
                    acc = (acc + value.broadcast_mul(&w)?)?;
                }
            }
            outs.push(acc.unsqueeze(2)?);
        }
        let out_refs = outs.iter().collect::<Vec<_>>();
        let out = Tensor::cat(&out_refs, 2)?.silu()?;

        self.conv_state = Some(if full_len >= self.conv_kernel_size {
            full.narrow(2, full_len - self.conv_kernel_size, self.conv_kernel_size)?
                .copy()?
        } else {
            full.pad_with_zeros(2, self.conv_kernel_size - full_len, 0)?
                .copy()?
        });
        Ok(out)
    }

    fn recurrent_delta_rule(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        let (b, l, h, k_dim) = query.dims4()?;
        if l == 1 {
            return self.recurrent_delta_rule_decode(query, key, value, g, beta);
        }

        let v_dim = value.dim(D::Minus1)?;
        let dtype = query.dtype();
        let query = (l2norm(query)?.to_dtype(DType::F32)? * (1.0 / (k_dim as f64).sqrt()))?;
        let key = l2norm(key)?.to_dtype(DType::F32)?;
        let value = value.to_dtype(DType::F32)?;
        let beta = beta.to_dtype(DType::F32)?;
        let g = g.to_dtype(DType::F32)?;

        let mut state = match &self.recurrent_state {
            Some(state) if state.dtype() == DType::F32 => state.clone(),
            Some(state) => state.to_dtype(DType::F32)?,
            None => Tensor::zeros((b, h, k_dim, v_dim), DType::F32, query.device())?,
        };

        let mut outs = Vec::with_capacity(l);
        for idx in 0..l {
            let q_t = query.narrow(1, idx, 1)?.squeeze(1)?;
            let k_t = key.narrow(1, idx, 1)?.squeeze(1)?;
            let v_t = value.narrow(1, idx, 1)?.squeeze(1)?;
            let g_t = g
                .narrow(1, idx, 1)?
                .squeeze(1)?
                .exp()?
                .unsqueeze(2)?
                .unsqueeze(3)?;
            let beta_t = beta.narrow(1, idx, 1)?.squeeze(1)?.unsqueeze(2)?;
            state = state.broadcast_mul(&g_t)?;
            let kv_mem = state.broadcast_mul(&k_t.unsqueeze(3)?)?.sum(2)?;
            let delta = (v_t - kv_mem)?.broadcast_mul(&beta_t)?;
            state = (state + k_t.unsqueeze(3)?.broadcast_mul(&delta.unsqueeze(2)?)?)?;
            let out = state.broadcast_mul(&q_t.unsqueeze(3)?)?.sum(2)?;
            outs.push(out.unsqueeze(1)?);
        }
        self.recurrent_state = Some(state.to_dtype(dtype)?);
        let out_refs = outs.iter().collect::<Vec<_>>();
        Tensor::cat(&out_refs, 1)?.to_dtype(dtype)
    }

    fn recurrent_delta_rule_decode(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        let (b, _, h, k_dim) = query.dims4()?;
        let v_dim = value.dim(D::Minus1)?;
        let dtype = query.dtype();
        let query =
            (l2norm(&query.squeeze(1)?)?.to_dtype(DType::F32)? * (1.0 / (k_dim as f64).sqrt()))?;
        let key = l2norm(&key.squeeze(1)?)?.to_dtype(DType::F32)?;
        let value = value.squeeze(1)?.to_dtype(DType::F32)?;
        let beta = beta.squeeze(1)?.to_dtype(DType::F32)?.unsqueeze(2)?;
        let g = g
            .squeeze(1)?
            .to_dtype(DType::F32)?
            .exp()?
            .unsqueeze(2)?
            .unsqueeze(3)?;

        let mut state = match &self.recurrent_state {
            Some(state) if state.dtype() == DType::F32 => state.clone(),
            Some(state) => state.to_dtype(DType::F32)?,
            None => Tensor::zeros((b, h, k_dim, v_dim), DType::F32, query.device())?,
        };
        state = state.broadcast_mul(&g)?;
        let kv_mem = state.broadcast_mul(&key.unsqueeze(3)?)?.sum(2)?;
        let delta = (value - kv_mem)?.broadcast_mul(&beta)?;
        state = (state + key.unsqueeze(3)?.broadcast_mul(&delta.unsqueeze(2)?)?)?;
        let out = state.broadcast_mul(&query.unsqueeze(3)?)?.sum(2)?;
        self.recurrent_state = Some(state.to_dtype(dtype)?);
        out.unsqueeze(1)?.to_dtype(dtype)
    }
}

fn maybe_repeat_heads(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b, l, h, d) = xs.dims4()?;
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b, l, h * n_rep, d))
    }
}

fn profiled<T>(
    profiler: Option<&Qwen35Profiler>,
    device: &Device,
    layer_index: Option<usize>,
    layer_kind: Option<&'static str>,
    component: &'static str,
    seq_len: usize,
    offset: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if let Some(profiler) = profiler {
        device.synchronize()?;
        let started = Instant::now();
        let output = f()?;
        device.synchronize()?;
        profiler.record(
            component,
            layer_index,
            layer_kind,
            seq_len,
            offset,
            started.elapsed().as_secs_f64(),
        );
        Ok(output)
    } else {
        f()
    }
}

fn l2norm(xs: &Tensor) -> Result<Tensor> {
    let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.powf(-0.5)?;
    xs.broadcast_mul(&denom)
}

#[derive(Clone, Debug)]
enum TokenMixer {
    Full(FullAttention),
    Linear(GatedDeltaNet),
}

impl TokenMixer {
    fn kind(&self) -> &'static str {
        match self {
            Self::Full(_) => "full_attention",
            Self::Linear(_) => "linear_attention",
        }
    }

    fn apply_quantized_text_artifact(
        &mut self,
        layer_index: usize,
        artifact: &QuantizedTextArtifact,
    ) -> Result<usize> {
        match self {
            Self::Full(attn) => attn.apply_quantized_text_artifact(layer_index, artifact),
            Self::Linear(attn) => attn.apply_quantized_text_artifact(layer_index, artifact),
        }
    }
}

#[derive(Clone, Debug)]
struct DecoderLayer {
    mixer: TokenMixer,
    mlp: Mlp,
    input_layernorm: Qwen35RmsNorm,
    post_attention_layernorm: Qwen35RmsNorm,
}

impl DecoderLayer {
    fn new(
        layer_type: LayerType,
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mixer = match layer_type {
            LayerType::FullAttention => {
                TokenMixer::Full(FullAttention::new(cfg, rotary, vb.pp("self_attn"))?)
            }
            LayerType::LinearAttention => {
                TokenMixer::Linear(GatedDeltaNet::new(cfg, vb.pp("linear_attn"))?)
            }
        };
        Ok(Self {
            mixer,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: Qwen35RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                true,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: Qwen35RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                true,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn kind(&self) -> &'static str {
        self.mixer.kind()
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
        layer_index: usize,
        profiler: Option<&Qwen35Profiler>,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = xs.dims3()?;
        let device = xs.device().clone();
        let layer_kind = self.mixer.kind();
        let residual = xs;
        let hidden = profiled(
            profiler,
            &device,
            Some(layer_index),
            Some(layer_kind),
            "input_layernorm",
            seq_len,
            offset,
            || self.input_layernorm.forward(xs),
        )?;
        let hidden = match &mut self.mixer {
            TokenMixer::Full(attn) => attn.forward(&hidden, mask, offset, layer_index, profiler)?,
            TokenMixer::Linear(attn) => attn.forward(&hidden, layer_index, offset, profiler)?,
        };
        let xs = (residual + hidden)?;
        let residual = &xs;
        let hidden = profiled(
            profiler,
            &device,
            Some(layer_index),
            Some(layer_kind),
            "post_attention_layernorm",
            seq_len,
            offset,
            || self.post_attention_layernorm.forward(&xs),
        )?;
        let hidden = profiled(
            profiler,
            &device,
            Some(layer_index),
            Some(layer_kind),
            "mlp",
            seq_len,
            offset,
            || self.mlp.forward(&hidden),
        )?;
        residual + hidden
    }

    fn clear_cache(&mut self) {
        match &mut self.mixer {
            TokenMixer::Full(attn) => attn.clear_cache(),
            TokenMixer::Linear(attn) => attn.clear_cache(),
        }
    }

    fn apply_quantized_text_artifact(
        &mut self,
        layer_index: usize,
        artifact: &QuantizedTextArtifact,
    ) -> Result<usize> {
        let mut replaced = self
            .mixer
            .apply_quantized_text_artifact(layer_index, artifact)?;
        replaced += self
            .mlp
            .apply_quantized_text_artifact(layer_index, artifact)?;
        Ok(replaced)
    }
}

fn replace_quantized_linear(
    linear: &mut MixedLinear,
    name: &str,
    artifact: &QuantizedTextArtifact,
) -> Result<usize> {
    match artifact
        .load_linear(name)
        .map_err(|err| candle::Error::Msg(format!("load quantized linear {name}: {err}")))?
    {
        Some(quantized) => {
            *linear = quantized;
            Ok(1)
        }
        None => Ok(0),
    }
}

#[derive(Clone, Debug)]
pub struct Qwen35TextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: Qwen35RmsNorm,
    device: Device,
    dtype: DType,
    profiler: Option<Qwen35Profiler>,
    trace_recorder: Option<Qwen35TraceRecorder>,
}

impl Qwen35TextModel {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(cfg, vb.dtype(), vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("layers");
        for idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                cfg.layer_types[idx],
                cfg,
                rotary.clone(),
                vb_layers.pp(idx),
            )?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: Qwen35RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, true, vb.pp("norm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            profiler: None,
            trace_recorder: None,
        })
    }

    pub fn embeddings(&self) -> &Tensor {
        self.embed_tokens.embeddings()
    }

    pub fn apply_quantized_text_artifact(
        &mut self,
        artifact: &QuantizedTextArtifact,
    ) -> Result<usize> {
        let mut replaced = 0usize;
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            replaced += layer.apply_quantized_text_artifact(layer_index, artifact)?;
        }
        Ok(replaced)
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward_ids(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let embeds = self.embed(input_ids)?;
        self.forward_embeds(&embeds, offset)
    }

    pub fn forward_embeds(&mut self, inputs_embeds: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l, _) = inputs_embeds.dims3()?;
        let mask = if l > 1 {
            Some(self.causal_mask(b, l, offset)?)
        } else {
            None
        };
        let profiler = self.profiler.clone();
        let trace_recorder = self.trace_recorder.clone();
        let mut hidden = inputs_embeds.clone();
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            let layer_kind = layer.kind();
            hidden = layer.forward(
                &hidden,
                mask.as_ref(),
                offset,
                layer_index,
                profiler.as_ref(),
            )?;
            if let Some(trace_recorder) = trace_recorder.as_ref() {
                trace_recorder.record_layer(layer_index, layer_kind, &hidden, offset)?;
            }
        }
        profiled(
            profiler.as_ref(),
            &self.device,
            None,
            None,
            "final_norm",
            l,
            offset,
            || self.norm.forward(&hidden),
        )
    }

    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    pub fn set_profiler(&mut self, profiler: Option<Qwen35Profiler>) {
        self.profiler = profiler;
    }

    pub fn set_trace_recorder(&mut self, trace_recorder: Option<Qwen35TraceRecorder>) {
        self.trace_recorder = trace_recorder;
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let total = tgt + offset;
        let mask = (0..tgt)
            .flat_map(|i| (0..total).map(move |j| if j <= i + offset { 0.0 } else { minf }))
            .collect::<Vec<_>>();
        Tensor::from_slice(&mask, (b, 1, tgt, total), &self.device)?.to_dtype(self.dtype)
    }
}
