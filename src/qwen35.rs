use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, Embedding};
use candle_transformers::utils::repeat_kv;
use serde::Serialize;

use crate::config::{LayerType, TextConfig};
use crate::linear_source::{LinearPart, LinearSource};
use crate::model_ctx::ModelCtx;
use crate::quantized_linear::{MixedLinear, QuantizedTextArtifact};
use crate::runtime_config::KernelRouteConfig;

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
    // Zero-centred +1.0 pre-applied at load. Two dtype copies because the
    // fused Metal rmsnorm kernel requires alpha to match the input dtype,
    // and callers hit this with both the model dtype and F32 activations.
    weight_f32: Tensor,
    weight_native: Tensor,
    eps: f64,
    routes: Arc<KernelRouteConfig>,
}

impl Qwen35RmsNorm {
    fn new<S: LinearSource>(
        size: usize,
        eps: f64,
        zero_centered: bool,
        source: &S,
        ctx: &ModelCtx,
    ) -> Result<Self> {
        // GGUF folds the +1 zero-centring shift into the stored weight; raw
        // Qwen safetensors store w-1 and need it applied here.
        let zero_centered = zero_centered && !source.norms_pre_shifted();
        let weight = source.tensor(size.into(), "weight")?.to_dtype(DType::F32)?;
        let weight_f32 = if zero_centered { (weight + 1.0)? } else { weight };
        let weight_native = weight_f32.to_dtype(source.dtype())?;
        Ok(Self {
            weight_f32,
            weight_native,
            eps,
            routes: ctx.routes.clone(),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if !self.routes.fused_rmsnorm {
            return self.forward_unfused(xs);
        }
        // Fused kernel: 1 dispatch instead of 9, F32 accumulation inside.
        // The kernel wants contiguous input; the only non-contiguous callers
        // are the q/k-norm narrows, which pay one copy here (still 9 -> 2).
        let weight = if xs.dtype() == DType::F32 {
            &self.weight_f32
        } else {
            &self.weight_native
        };
        let xs = if xs.is_contiguous() {
            xs.clone()
        } else {
            xs.contiguous()?
        };
        candle_nn::ops::rms_norm(&xs, weight, self.eps as f32)
    }

    /// Fused `(a + b, rms_norm(a + b))`: one dispatch for the residual add +
    /// the norm that consumes it, emitting both the new residual (a+b) and
    /// the normed output. Falls back to the two-op form off the Metal/BF16
    /// hot path or under `LMBRRR_UNFUSED_RMSNORM=1`. Not bit-preserving (see
    /// fused_norm) — gated by the model oracle, not bitwise.
    fn forward_add(&self, a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor)> {
        if self.routes.fused_rmsnorm
            && matches!(a.device(), Device::Metal(_))
            && a.dtype() == DType::BF16
            && b.dtype() == DType::BF16
        {
            return crate::fused_norm::fused_rms_norm_add(
                a,
                b,
                &self.weight_native,
                self.eps as f32,
            )
            .map_err(|e| candle::Error::Msg(format!("fused rms_norm_add: {e:#}")));
        }
        let sum = (a + b)?;
        let normed = self.forward(&sum)?;
        Ok((sum, normed))
    }

    /// Reference path (`LMBRRR_UNFUSED_RMSNORM=1`) for drift attribution.
    fn forward_unfused(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let inv = (variance + self.eps)?.powf(-0.5)?;
        let ys = xs_f32.broadcast_mul(&inv)?;
        ys.broadcast_mul(&self.weight_f32)?.to_dtype(dtype)
    }
}

// rotary_dim is not stored: rope_partial derives it from the table width
// (cos/sin rows are rotary_dim/2 wide).
#[derive(Clone, Debug)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
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
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        self.apply_tables(q, k, &cos, &sin)
    }

    /// Rotary application at explicit per-token positions (tree verification:
    /// sibling branch segments share the positions of the main branch).
    fn apply_with_positions(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &[u32],
    ) -> Result<(Tensor, Tensor)> {
        let idx = Tensor::from_slice(positions, positions.len(), q.device())?;
        let cos = self.cos.index_select(&idx, 0)?;
        let sin = self.sin.index_select(&idx, 0)?;
        self.apply_tables(q, k, &cos, &sin)
    }

    fn apply_tables(
        &self,
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // rope_partial rotates the first rotary_dim lanes and passes the
        // rest through in one kernel — bit-identical to the old
        // narrow/rope/cat composition, minus 4 movement dispatches per
        // layer (rd == head_dim degenerates to plain rope).
        Ok((
            candle_nn::rotary_emb::rope_partial(&q.contiguous()?, cos, sin)?,
            candle_nn::rotary_emb::rope_partial(&k.contiguous()?, cos, sin)?,
        ))
    }
}

/// KV cache with truncation for speculative rollback. Grows on demand
/// (candle_nn's KvCache preallocates max_position_embeddings and cannot
/// rewind); `truncate` just moves the length back — the buffer beyond it is
/// overwritten by the re-advance chunk.
#[derive(Clone, Debug)]
pub struct TruncatableKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    len: usize,
}

impl Default for TruncatableKvCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TruncatableKvCache {
    const MIN_CAPACITY: usize = 1024;

    pub fn new() -> Self {
        Self {
            k: None,
            v: None,
            len: 0,
        }
    }

    fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.len = 0;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.len {
            candle::bail!("cannot truncate KV cache from {} to {len}", self.len);
        }
        self.len = len;
        Ok(())
    }

    fn ensure_capacity(slot: &mut Option<Tensor>, template: &Tensor, needed: usize, len: usize) -> Result<()> {
        let (b, h, _, d) = template.dims4()?;
        Self::ensure_capacity_dims(slot, (b, h, d), template.dtype(), template.device(), needed, len)
    }

    fn ensure_capacity_dims(
        slot: &mut Option<Tensor>,
        (b, h, d): (usize, usize, usize),
        dtype: DType,
        device: &Device,
        needed: usize,
        len: usize,
    ) -> Result<()> {
        let current_capacity = match slot {
            Some(buffer) => buffer.dim(2)?,
            None => 0,
        };
        if current_capacity >= needed {
            return Ok(());
        }
        let capacity = needed.next_power_of_two().max(Self::MIN_CAPACITY);
        let buffer = Tensor::zeros((b, h, capacity, d), dtype, device)?;
        if let Some(old) = slot.as_ref() {
            if len > 0 {
                buffer.slice_set(&old.narrow(2, 0, len)?.contiguous()?, 2, 0)?;
            }
        }
        *slot = Some(buffer);
        Ok(())
    }

    /// Fused-append protocol: grow to fit `added` rows and hand out the raw
    /// cache buffers plus the write position; the caller writes the rows
    /// in-kernel (see fused_attn) and then calls `advance`.
    pub fn reserve(
        &mut self,
        kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
        added: usize,
    ) -> Result<(Tensor, Tensor, usize)> {
        let needed = self.len + added;
        let dims = (1, kv_heads, head_dim);
        Self::ensure_capacity_dims(&mut self.k, dims, dtype, device, needed, self.len)?;
        Self::ensure_capacity_dims(&mut self.v, dims, dtype, device, needed, self.len)?;
        let k = self.k.clone().expect("k buffer allocated");
        let v = self.v.clone().expect("v buffer allocated");
        Ok((k, v, self.len))
    }

    pub fn advance(&mut self, added: usize) {
        self.len += added;
    }

    /// Winner-path compaction after tree verification: moves `count` rows
    /// starting at `src` down to `dst` (dst < src) and truncates to
    /// `dst + count`. Used when the alternate branch wins — its rows sit
    /// after the main branch's in the flattened chunk and must become the
    /// sequence suffix.
    pub fn compact_rows(&mut self, dst: usize, src: usize, count: usize) -> Result<()> {
        if dst >= src {
            candle::bail!("compact_rows requires dst < src (got {dst} >= {src})");
        }
        if src + count > self.len {
            candle::bail!(
                "compact_rows source range {src}+{count} exceeds cache length {}",
                self.len
            );
        }
        if count > 0 {
            let (k, v) = match (&self.k, &self.v) {
                (Some(k), Some(v)) => (k, v),
                _ => candle::bail!("compact_rows on an empty KV cache"),
            };
            // slice_set needs a src that is contiguous AND does not alias
            // the cache storage. copy() clones storage but keeps the narrow's
            // strided layout (non-contiguous whenever heads > 1), while
            // contiguous() on an already-contiguous narrow aliases the cache
            // — so pick per case.
            let k_rows = k.narrow(2, src, count)?;
            let v_rows = v.narrow(2, src, count)?;
            let k_rows = if k_rows.is_contiguous() { k_rows.copy()? } else { k_rows.contiguous()? };
            let v_rows = if v_rows.is_contiguous() { v_rows.copy()? } else { v_rows.contiguous()? };
            k.slice_set(&k_rows, 2, dst)?;
            v.slice_set(&v_rows, 2, dst)?;
        }
        self.len = dst + count;
        Ok(())
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let added = k.dim(2)?;
        let needed = self.len + added;
        Self::ensure_capacity(&mut self.k, k, needed, self.len)?;
        Self::ensure_capacity(&mut self.v, v, needed, self.len)?;
        let k_buffer = self.k.as_ref().expect("k buffer allocated");
        let v_buffer = self.v.as_ref().expect("v buffer allocated");
        k_buffer.slice_set(k, 2, self.len)?;
        v_buffer.slice_set(v, 2, self.len)?;
        self.len = needed;
        Ok((
            k_buffer.narrow(2, 0, self.len)?,
            v_buffer.narrow(2, 0, self.len)?,
        ))
    }
}

/// Cheap per-layer decode-state snapshot: DeltaNet tensors are replaced by
/// assignment (never mutated in place) so cloning the handles is enough; the
/// KV cache only needs its length because rollback rewinds and the re-advance
/// overwrites the stale slice.
#[derive(Clone, Debug)]
pub struct DecodeStateSnapshot {
    deltanet: Vec<(Option<Tensor>, Option<Tensor>, bool)>,
    kv_lens: Vec<usize>,
}

/// Per-forward context for two-branch tree verification. The flattened chunk
/// layout is [anchor, a_1..a_w, b_1..b_w]: both branch segments continue from
/// the anchor, so b_i shares a_i's absolute position and each row's attention
/// is limited to history + its own ancestors (the `mask`, built by the model).
#[derive(Clone, Debug)]
pub struct TreeForward {
    /// Tokens per branch (w >= 1); flattened chunk length is 1 + 2w.
    pub branch_width: usize,
    /// Absolute rotary position per flattened-chunk token, length 1 + 2w.
    pub positions: Vec<u32>,
    /// Ancestor attention mask, (1, 1, 1 + 2w, offset + 1 + 2w), model dtype.
    pub mask: Tensor,
}

#[derive(Clone, Debug)]
struct Mlp {
    // gate and up fused into one [2*intermediate, hidden] weight: one wide
    // GEMV streams ~2x the GB/s of two skinny ones (width fusion, K2).
    // Row-concatenation never crosses a quant block, so the fused quantized
    // load is bitwise-identical to separate loads.
    gate_up_proj: MixedLinear,
    down_proj: MixedLinear,
    intermediate_size: usize,
    act: Activation,
}

impl Mlp {
    fn new<S: LinearSource>(cfg: &TextConfig, source: &S) -> Result<Self> {
        Ok(Self {
            gate_up_proj: source.linear(
                &[
                    LinearPart::new("gate_proj.weight", cfg.intermediate_size),
                    LinearPart::new("up_proj.weight", cfg.intermediate_size),
                ],
                cfg.hidden_size,
                false,
            )?,
            down_proj: source.linear(
                &[LinearPart::new("down_proj.weight", cfg.hidden_size)],
                cfg.intermediate_size,
                false,
            )?,
            intermediate_size: cfg.intermediate_size,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(xs)?;
        // Fused SwiGLU: silu(gate) * up in one dispatch over the packed
        // [.., 2*inter] projection (collapses the silu unary + the gate*up
        // bmul + their barrier). Only for Silu; other activations keep the
        // explicit narrow/apply/mul path.
        let gated = if matches!(self.act, Activation::Silu) {
            candle_nn::ops::swiglu(&gate_up)?
        } else {
            let lhs = gate_up
                .narrow(D::Minus1, 0, self.intermediate_size)?
                .apply(&self.act)?;
            let rhs =
                gate_up.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
            (lhs * rhs)?
        };
        self.down_proj.forward(&gated)
    }

    fn apply_quantized_text_artifact(
        &mut self,
        layer_index: usize,
        artifact: &QuantizedTextArtifact,
    ) -> Result<usize> {
        let prefix = format!("model.language_model.layers.{layer_index}.mlp");
        let mut replaced = 0usize;
        if let Some(fused) = artifact
            .load_linear_fused(&[
                &format!("{prefix}.gate_proj.weight"),
                &format!("{prefix}.up_proj.weight"),
            ])
            .map_err(|err| candle::Error::Msg(format!("load fused mlp gate_up {err}")))?
        {
            self.gate_up_proj = fused;
            replaced += 2;
        }
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
    // q (incl. packed output gate) + k + v fused into one wide weight (K2
    // width fusion): rows are [2*heads*hd | kv*hd | kv*hd].
    qkv_proj: MixedLinear,
    o_proj: MixedLinear,
    q_norm: Qwen35RmsNorm,
    k_norm: Qwen35RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    num_kv_groups: usize,
    rotary: Arc<RotaryEmbedding>,
    kv_cache: Arc<Mutex<TruncatableKvCache>>,
    routes: Arc<KernelRouteConfig>,
}

impl FullAttention {
    fn new<S: LinearSource>(
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        source: &S,
        ctx: &ModelCtx,
    ) -> Result<Self> {
        let q_out = cfg.num_attention_heads * cfg.head_dim * 2;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        Ok(Self {
            qkv_proj: source.linear(
                &[
                    LinearPart::new("q_proj.weight", q_out),
                    LinearPart::new("k_proj.weight", kv_out),
                    LinearPart::new("v_proj.weight", kv_out),
                ],
                cfg.hidden_size,
                cfg.attention_bias,
            )?,
            o_proj: source.linear(
                &[LinearPart::new("o_proj.weight", cfg.hidden_size)],
                cfg.num_attention_heads * cfg.head_dim,
                cfg.attention_bias,
            )?,
            q_norm: Qwen35RmsNorm::new(
                cfg.head_dim,
                cfg.rms_norm_eps,
                true,
                &source.sub("q_norm"),
                ctx,
            )?,
            k_norm: Qwen35RmsNorm::new(
                cfg.head_dim,
                cfg.rms_norm_eps,
                true,
                &source.sub("k_norm"),
                ctx,
            )?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            hidden_size: cfg.num_attention_heads * cfg.head_dim,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            rotary,
            kv_cache: Arc::new(Mutex::new(TruncatableKvCache::new())),
            routes: ctx.routes.clone(),
        })
    }

    fn kv_len(&self) -> usize {
        self.kv_cache
            .lock()
            .expect("full-attention KV cache lock poisoned")
            .len()
    }

    fn truncate_kv(&self, len: usize) -> Result<()> {
        self.kv_cache
            .lock()
            .expect("full-attention KV cache lock poisoned")
            .truncate(len)
    }

    /// Alternate-branch winner after a tree verify: its `accepted` rows sit
    /// at [history + 1 + w ..) in the flattened append order and become the
    /// sequence suffix right after the anchor.
    fn compact_tree_kv(&self, history: usize, branch_width: usize, accepted: usize) -> Result<()> {
        self.kv_cache
            .lock()
            .expect("full-attention KV cache lock poisoned")
            .compact_rows(history + 1, history + 1 + branch_width, accepted)
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
        layer_index: usize,
        profiler: Option<&Qwen35Profiler>,
        rope_positions: Option<&[u32]>,
    ) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let device = xs.device().clone();
        let q_out = self.num_heads * self.head_dim * 2;
        let kv_out = self.num_kv_heads * self.head_dim;
        let qkv = profiled(
            profiler,
            &device,
            Some((layer_index, "full_attention")),
            "full_attention_q_gate_projection",
            l,
            offset,
            || self.qkv_proj.forward(xs),
        )?;
        let q_gate = qkv
            .narrow(D::Minus1, 0, q_out)?
            .reshape((b, l, self.num_heads, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((b, l, self.hidden_size))?;
        // Decode (l == 1, no mask) routes to the fused SDPA vector kernel:
        // native GQA (no repeat_kv materialization of the whole cache) and
        // strided k/v (the cache narrows feed in directly, no k_t
        // transpose+contiguous copy). Verify chunks (2 <= l <= 16 with the
        // model's causal mask) route to SDPA-full with an explicitly
        // materialized (b, qheads, l, kv) mask — the kernel's do_causal
        // alignment does NOT match the offset-causal semantics (measured;
        // gates rejected it), and stride-0 broadcast masks measure wrong,
        // so the mask copy is the price of the fused path. It is tiny for
        // chunks but heavy for long prefill, hence the l <= 16 bound; long
        // prefill stays on the tensor path below. LMBRRR_UNFUSED_SDPA=1
        // restores both reference paths.
        let use_sdpa =
            (l == 1 && mask.is_none() || (2..=16).contains(&l) && mask.is_some())
                && self.routes.fused_sdpa;
        // Contiguous-position forwards take the fused prep path: q/k
        // head-norm + partial rope + direct cache write in two dispatches,
        // bit-identical to the chain below (see fused_attn). Tree
        // verification passes explicit rope_positions and stays unfused.
        let fused_prep = self.routes.fused_attn_prep
            && b == 1
            && rope_positions.is_none()
            && qkv.dtype() == DType::BF16
            && matches!(&device, Device::Metal(_))
            && crate::fused_attn::supports_head_dim(self.head_dim);
        let (q, k, v, k_t) = if fused_prep {
            profiled(
                profiler,
                &device,
                Some((layer_index, "full_attention")),
                "full_attention_fused_qkv_prep",
                l,
                offset,
                || {
                    let mut cache = self
                        .kv_cache
                        .lock()
                        .expect("full-attention KV cache lock poisoned");
                    let (cache_k, cache_v, write_pos) =
                        cache.reserve(self.num_kv_heads, self.head_dim, qkv.dtype(), &device, l)?;
                    let q = crate::fused_attn::fused_qkv_prep(
                        &qkv,
                        &self.q_norm.weight_native,
                        &self.k_norm.weight_native,
                        self.q_norm.eps as f32,
                        &self.rotary.cos,
                        &self.rotary.sin,
                        offset,
                        self.num_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        &cache_k,
                        &cache_v,
                        write_pos,
                    )
                    .map_err(|e| candle::Error::Msg(format!("fused attn prep: {e:#}")))?;
                    cache.advance(l);
                    let len = cache.len();
                    let k = cache_k.narrow(2, 0, len)?;
                    let v = cache_v.narrow(2, 0, len)?;
                    if use_sdpa {
                        return Ok((q, k, v, None));
                    }
                    let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
                    let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;
                    let k_t = k.transpose(2, 3)?.contiguous()?;
                    Ok((q, k, v, Some(k_t)))
                },
            )?
        } else {
            let (mut q, mut k, v) = profiled(
                profiler,
                &device,
                Some((layer_index, "full_attention")),
                "full_attention_kv_projection_norm",
                l,
                offset,
                || {
                    let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
                    let k = self
                        .k_norm
                        .forward(&qkv.narrow(D::Minus1, q_out, kv_out)?.reshape((
                            b,
                            l,
                            self.num_kv_heads,
                            self.head_dim,
                        ))?)?
                        .transpose(1, 2)?;
                    let v = qkv
                        .narrow(D::Minus1, q_out + kv_out, kv_out)?
                        .reshape((b, l, self.num_kv_heads, self.head_dim))?
                        .transpose(1, 2)?;
                    Ok((q, k, v))
                },
            )?;
            profiled(
                profiler,
                &device,
                Some((layer_index, "full_attention")),
                "full_attention_rotary_kv_cache",
                l,
                offset,
                || {
                    (q, k) = match rope_positions {
                        Some(positions) => self.rotary.apply_with_positions(&q, &k, positions)?,
                        None => self.rotary.apply(&q, &k, offset)?,
                    };
                    let (k, v) = self
                        .kv_cache
                        .lock()
                        .expect("full-attention KV cache lock poisoned")
                        .append(&k.contiguous()?, &v.contiguous()?)?;
                    let q = q.contiguous()?;
                    if use_sdpa {
                        return Ok((q, k, v, None));
                    }
                    let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
                    let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;
                    let k_t = k.transpose(2, 3)?.contiguous()?;
                    Ok((q, k, v, Some(k_t)))
                },
            )?
        };

        let out = profiled(
            profiler,
            &device,
            Some((layer_index, "full_attention")),
            "full_attention_matmul_softmax",
            l,
            offset,
            || {
                let scale = 1.0 / (self.head_dim as f64).sqrt();
                let Some(k_t) = k_t else {
                    // Chunks pass the model's offset-causal mask explicitly,
                    // broadcast to the kernel's (b, qheads, l, kv) contract
                    // (the kernel's do_causal alignment does NOT match the
                    // offset-causal semantics — measured, gates rejected it).
                    let sdpa_mask = match mask {
                        Some(m) if l > 1 => Some(
                            m.broadcast_as((b, self.num_heads, l, k.dim(2)?))?
                                .contiguous()?,
                        ),
                        _ => None,
                    };
                    let out = candle_nn::ops::sdpa(
                        &q,
                        &k,
                        &v,
                        sdpa_mask.as_ref(),
                        false,
                        scale as f32,
                        1.0,
                    )?;
                    return out.transpose(1, 2)?.reshape((b, l, self.hidden_size));
                };
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
            Some((layer_index, "full_attention")),
            "full_attention_output_projection",
            l,
            offset,
            || {
                // Fused x * sigmoid(gate): one dispatch instead of sigmoid +
                // bmul + their barrier; bit-identical (see fused_gate).
                let out = crate::fused_gate::mul_sigmoid(&out, &gate)
                    .map_err(|e| candle::Error::Msg(format!("mul_sigmoid: {e:#}")))?;
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
        if let Some(fused) = artifact
            .load_linear_fused(&[
                &format!("{prefix}.q_proj.weight"),
                &format!("{prefix}.k_proj.weight"),
                &format!("{prefix}.v_proj.weight"),
            ])
            .map_err(|err| candle::Error::Msg(format!("load fused attn qkv {err}")))?
        {
            self.qkv_proj = fused;
            replaced += 3;
        }
        replaced += replace_quantized_linear(
            &mut self.o_proj,
            &format!("{prefix}.o_proj.weight"),
            artifact,
        )?;
        Ok(replaced)
    }
}

/// Chunk intermediates retained during a verify forward so a partial accept
/// can reconstruct the recurrent/conv state at any prefix position in closed
/// form instead of re-advancing the prefix through the whole model:
/// S_j = exp(G_j)·S0 + Σ_{i≤j} exp(G_j−G_i)·k_i ⊗ δ_i — the chunk-end update
/// is exactly this formula at j = C−1, so the reconstruction shares its math.
#[derive(Clone, Debug)]
/// Conv-window source for rollback reconstruction. Only the last `ksz`
/// columns at the rollback prefix are ever read, so the fused paths defer
/// the transpose+cat to the rollback instead of materializing the whole
/// window every captured round (18 strided copies/round saved).
enum ConvCapture {
    /// Assembled window cat(prev_conv_state, chunk_inputs) — the tensor path
    /// (where the conv stages it anyway) and the tree path use this.
    Full {
        conv_full: Tensor,
        /// Columns of conv_full belonging to the previous conv state.
        prev_conv_len: usize,
    },
    /// Deferred: previous conv state (b, conv_dim, prev) + the RAW
    /// un-transposed chunk inputs (b, l, conv_dim) as a zero-copy view of
    /// the qkvz projection; the needed columns transpose on rollback only.
    Lazy { conv_prev: Tensor, mixed_raw: Tensor },
}

impl ConvCapture {
    /// The conv state as of `prefix_len` chunk positions: the last `ksz`
    /// columns of cat(prev_state, inputs[..prefix]), front-padded with
    /// zeros pre-warmup. Byte-identical between variants.
    fn window(&self, prefix_len: usize, ksz: usize) -> Result<Tensor> {
        match self {
            Self::Full {
                conv_full,
                prev_conv_len,
            } => {
                let window_len = prev_conv_len + prefix_len;
                if window_len >= ksz {
                    conv_full.narrow(2, window_len - ksz, ksz)?.copy()
                } else {
                    conv_full
                        .narrow(2, 0, window_len)?
                        .pad_with_zeros(2, ksz - window_len, 0)?
                        .copy()
                }
            }
            Self::Lazy {
                conv_prev,
                mixed_raw,
            } => {
                let prev_len = conv_prev.dim(2)?;
                let window_len = prev_len + prefix_len;
                let start = window_len.saturating_sub(ksz);
                let take = window_len - start;
                let mixed_cols = |c0: usize, n: usize| -> Result<Tensor> {
                    mixed_raw.narrow(1, c0, n)?.transpose(1, 2)?.contiguous()
                };
                let window = if start >= prev_len {
                    mixed_cols(start - prev_len, take)?
                } else if start + take <= prev_len {
                    conv_prev.narrow(2, start, take)?.copy()?
                } else {
                    let head = conv_prev.narrow(2, start, prev_len - start)?;
                    let tail = mixed_cols(0, start + take - prev_len)?;
                    Tensor::cat(&[&head, &tail], 2)?
                };
                if take >= ksz {
                    Ok(window)
                } else {
                    window.pad_with_zeros(2, ksz - take, 0)
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct DeltaVerifyCapture {
    /// Pre-chunk recurrent state, F32 (b, h, k_dim, v_dim).
    s0: Tensor,
    /// L2-normed keys, F32 (b, h, C, k_dim).
    kc: Tensor,
    /// WY pseudo-values, F32 (b, h, C, v_dim).
    delta: Tensor,
    /// Inclusive per-position log-decay cumsum, F32 (b, h, C).
    gcs: Tensor,
    /// Pre-conv input window (assembled or deferred).
    conv: ConvCapture,
    /// Storage dtype the chunked path would have used for the state.
    dtype: DType,
    /// Whether `s0` (and thus the reconstructed state) is in the v2
    /// transposed layout (b, h, v_dim, k_dim) instead of (b, h, k_dim, v_dim).
    transposed: bool,
}

#[derive(Clone, Debug)]
struct GatedDeltaNet {
    // qkv (conv_dim rows) + z (value_dim rows) fused into one wide weight
    // (K2 width fusion); b + a fused into one 2*heads-row projection whose
    // output IS the v2 kernels' [b | a] ba-buffer layout. The decay gates
    // stay dense-protected — never row-concat them into the QUANTIZED qkvz
    // weight; fusing the two dense gates with each other is layout-only.
    in_proj_qkvz: MixedLinear,
    in_proj_ba: MixedLinear,
    out_proj: MixedLinear,
    // Per-tap broadcast tensors (1, conv_dim, 1), precomputed at load: the
    // narrow+reshape of the raw (conv_dim, 1, ksz) weight is non-contiguous
    // and would run a copy kernel per tap per decode step.
    conv_taps: Vec<Tensor>,
    // Squeezed (conv_dim, ksz) weight for the fused decode kernel.
    conv_weight_full: Tensor,
    dt_bias_f32: Tensor,
    a_log_exp_f32: Tensor,
    norm: Qwen35RmsNorm,
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
    // Layout of recurrent_state: false = v1 (b, h, dk, dv); true = v2
    // transposed (b, h, dv, dk). Same shape at dk == dv, so this flag is the
    // only record of which one the tensor holds. v2 kernel paths transpose
    // lazily on entry; v1/tensor paths transpose back.
    state_transposed: bool,
    // Verify-state capture: enabled by the speculative runner around verify
    // chunks; single-chunk forwards retain reconstruction intermediates.
    verify_capture: bool,
    verify_captured: Option<DeltaVerifyCapture>,
    // Per-segment captures of the last tree verify forward: (main segment
    // [anchor, a_1..a_w], alternate segment [b_1..b_w]).
    tree_captured: Option<(DeltaVerifyCapture, DeltaVerifyCapture)>,
    // Conv input window staged by depthwise_causal_conv (which runs before
    // the recurrent rule in the layer forward) for capture assembly.
    pending_conv_window: Option<(Tensor, usize)>,
    conv_kernel_size: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    routes: Arc<KernelRouteConfig>,
}

impl GatedDeltaNet {
    fn new<S: LinearSource>(cfg: &TextConfig, source: &S, ctx: &ModelCtx) -> Result<Self> {
        let key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
        let value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let dt_bias = source.tensor(cfg.linear_num_value_heads.into(), "dt_bias")?;
        let dt_bias_f32 =
            dt_bias
                .to_dtype(DType::F32)?
                .reshape((1, 1, cfg.linear_num_value_heads))?;
        let conv_weight = source.tensor(
            (conv_dim, 1, cfg.linear_conv_kernel_dim).into(),
            "conv1d.weight",
        )?;
        let conv_weight = conv_weight.squeeze(1)?;
        let conv_taps = (0..cfg.linear_conv_kernel_dim)
            .map(|k| conv_weight.narrow(1, k, 1)?.reshape((1, conv_dim, 1)))
            .collect::<candle::Result<Vec<_>>>()?;
        let conv_weight_full = conv_weight.contiguous()?;
        let a_log = source.tensor(cfg.linear_num_value_heads.into(), "A_log")?;
        let a_log_exp_f32 =
            a_log
                .to_dtype(DType::F32)?
                .exp()?
                .reshape((1, 1, cfg.linear_num_value_heads))?;
        Ok(Self {
            in_proj_qkvz: source.linear(
                &[
                    LinearPart::new("in_proj_qkv.weight", conv_dim),
                    LinearPart::new("in_proj_z.weight", value_dim),
                ],
                cfg.hidden_size,
                false,
            )?,
            in_proj_ba: source.linear(
                &[
                    LinearPart::new("in_proj_b.weight", cfg.linear_num_value_heads),
                    LinearPart::new("in_proj_a.weight", cfg.linear_num_value_heads),
                ],
                cfg.hidden_size,
                false,
            )?,
            out_proj: source.linear(
                &[LinearPart::new("out_proj.weight", cfg.hidden_size)],
                value_dim,
                false,
            )?,
            conv_taps,
            conv_weight_full,
            dt_bias_f32,
            a_log_exp_f32,
            norm: Qwen35RmsNorm::new(
                cfg.linear_value_head_dim,
                cfg.rms_norm_eps,
                false,
                &source.sub("norm"),
                ctx,
            )?,
            conv_state: None,
            recurrent_state: None,
            state_transposed: false,
            verify_capture: false,
            verify_captured: None,
            tree_captured: None,
            pending_conv_window: None,
            conv_kernel_size: cfg.linear_conv_kernel_dim,
            num_k_heads: cfg.linear_num_key_heads,
            num_v_heads: cfg.linear_num_value_heads,
            head_k_dim: cfg.linear_key_head_dim,
            head_v_dim: cfg.linear_value_head_dim,
            key_dim,
            value_dim,
            conv_dim,
            routes: ctx.routes.clone(),
        })
    }

    fn clear_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
        self.state_transposed = false;
        self.verify_captured = None;
        self.tree_captured = None;
        self.pending_conv_window = None;
    }

    /// Closed-form recurrent + conv state as of the first `prefix_len`
    /// positions of a captured chunk:
    /// S_j = exp(G_j)·S0 + Σ_{i≤j} exp(G_j−G_i)·k_i ⊗ δ_i.
    /// Returns (recurrent in capture dtype, conv window).
    /// `fused_reconstruct` is the route gate (from the owning layer's
    /// KernelRouteConfig); this is a static helper so callers pass it
    /// explicitly (the equivalence test drives both paths).
    fn reconstruct_capture_state(
        cap: &DeltaVerifyCapture,
        prefix_len: usize,
        ksz: usize,
        fused_reconstruct: bool,
    ) -> Result<(Tensor, Tensor)> {
        let chunk_len = cap.gcs.dim(2)?;
        if prefix_len == 0 || prefix_len > chunk_len {
            candle::bail!("verify-state prefix {prefix_len} outside 1..={chunk_len}");
        }
        // Fused single-dispatch reconstruction (fork kernel): the tensor-op
        // chain below (8-10 dispatches of f32 broadcast/exp/GEMM per layer,
        // measured ~20% of a spec round) is kept as the reference path —
        // fused_reconstruct == false or any precondition miss routes there.
        // Margin-class: f32 accumulation order differs from the matmul.
        let recurrent = if fused_reconstruct {
            crate::fused_deltanet::gated_delta_v2_reconstruct(
                &cap.s0,
                &cap.kc,
                &cap.delta,
                &cap.gcs,
                prefix_len,
                cap.transposed,
                cap.dtype,
            )
            .ok()
        } else {
            None
        };
        let recurrent = match recurrent {
            Some(t) => t,
            None => Self::reconstruct_state_reference(cap, prefix_len)?,
        };

        let conv = cap.conv.window(prefix_len, ksz)?;
        Ok((recurrent, conv))
    }

    /// Reference reconstruction (tensor-op chain) — the fused kernel's
    /// fallback and its drift-attribution baseline.
    fn reconstruct_state_reference(cap: &DeltaVerifyCapture, prefix_len: usize) -> Result<Tensor> {
        let j = prefix_len - 1;
        let gcs_j = cap.gcs.narrow(2, j, 1)?; // (b, h, 1)
        let rel = gcs_j
            .broadcast_sub(&cap.gcs.narrow(2, 0, prefix_len)?)?
            .exp()?; // (b, h, j+1), entries ≤ 1
        let wk = cap
            .kc
            .narrow(2, 0, prefix_len)?
            .broadcast_mul(&rel.unsqueeze(3)?)?; // (b, h, j+1, k_dim)
        // v2 stores S transposed; (Σ wk_i ⊗ δ_i)^T = δ^T · wk, so the same
        // capture tensors reconstruct either layout.
        let state = if cap.transposed {
            (cap.delta
                .narrow(2, 0, prefix_len)?
                .transpose(2, 3)?
                .contiguous()?
                .matmul(&wk.contiguous()?)?
                + cap.s0.broadcast_mul(&gcs_j.exp()?.unsqueeze(3)?)?)?
        } else {
            (wk.transpose(2, 3)?
                .contiguous()?
                .matmul(&cap.delta.narrow(2, 0, prefix_len)?.contiguous()?)?
                + cap.s0.broadcast_mul(&gcs_j.exp()?.unsqueeze(3)?)?)?
        };
        // Match the dtype the chunked store (and thus a re-advance) would
        // have produced at this boundary.
        state.to_dtype(cap.dtype)
    }

    /// Reconstructs and installs the recurrent + conv state as of the first
    /// `prefix_len` positions of the last captured verify chunk, replacing a
    /// restore + re-advance forward on partial accept.
    fn select_verify_state(&mut self, prefix_len: usize) -> Result<()> {
        let Some(cap) = self.verify_captured.take() else {
            candle::bail!("select_verify_state without a captured verify chunk");
        };
        let transposed = cap.transposed;
        let (recurrent, conv) = Self::reconstruct_capture_state(
            &cap,
            prefix_len,
            self.conv_kernel_size,
            self.routes.fused_reconstruct,
        )?;
        self.recurrent_state = Some(recurrent);
        self.state_transposed = transposed;
        self.conv_state = Some(conv);
        Ok(())
    }

    /// Installs the winner-path state after a tree verify: the main segment's
    /// capture covers [anchor, a_1..a_w] (select `1 + accepted`), the
    /// alternate's covers [b_1..b_w] continuing from the anchor state (select
    /// `accepted`).
    fn select_tree_state(&mut self, on_alt: bool, prefix_in_segment: usize) -> Result<()> {
        let Some((cap_main, cap_alt)) = self.tree_captured.take() else {
            candle::bail!("select_tree_state without a captured tree forward");
        };
        let cap = if on_alt { cap_alt } else { cap_main };
        let transposed = cap.transposed;
        let (recurrent, conv) = Self::reconstruct_capture_state(
            &cap,
            prefix_in_segment,
            self.conv_kernel_size,
            self.routes.fused_reconstruct,
        )?;
        self.recurrent_state = Some(recurrent);
        self.state_transposed = transposed;
        self.conv_state = Some(conv);
        Ok(())
    }

    fn apply_quantized_text_artifact(
        &mut self,
        layer_index: usize,
        artifact: &QuantizedTextArtifact,
    ) -> Result<usize> {
        let prefix = format!("model.language_model.layers.{layer_index}.linear_attn");
        let mut replaced = 0usize;
        if let Some(fused) = artifact
            .load_linear_fused(&[
                &format!("{prefix}.in_proj_qkv.weight"),
                &format!("{prefix}.in_proj_z.weight"),
            ])
            .map_err(|err| candle::Error::Msg(format!("load fused deltanet qkvz {err}")))?
        {
            self.in_proj_qkvz = fused;
            replaced += 2;
        }
        // Deployed quant policy dense-protects the decay gates, so this is
        // normally a no-op; a manifest that does carry them must carry BOTH
        // (the fused ba weight cannot be half-replaced).
        match artifact
            .load_linear_fused(&[
                &format!("{prefix}.in_proj_b.weight"),
                &format!("{prefix}.in_proj_a.weight"),
            ])
            .map_err(|err| candle::Error::Msg(format!("load fused deltanet ba {err}")))?
        {
            Some(fused) => {
                self.in_proj_ba = fused;
                replaced += 2;
            }
            None => {
                for name in ["in_proj_b", "in_proj_a"] {
                    if artifact
                        .load_linear(&format!("{prefix}.{name}.weight"))
                        .map_err(|err| candle::Error::Msg(format!("probe {name} {err}")))?
                        .is_some()
                    {
                        candle::bail!(
                            "quantized manifest carries {prefix}.{name}.weight without its \
                             fused partner; the decay gates quantize together or not at all"
                        );
                    }
                }
            }
        }
        replaced += replace_quantized_linear(
            &mut self.out_proj,
            &format!("{prefix}.out_proj.weight"),
            artifact,
        )?;
        Ok(replaced)
    }

    /// Whole-layer fused decode step (one Metal dispatch after the packed
    /// projection cat); requires the shapes the kernel supports and a
    /// populated conv state (i.e. any step after prefill).
    fn forward_fused_decode(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, _, _) = xs.dims3()?;
        // v2 decode keeps the transposed layout end-to-end (no per-round
        // v1<->v2 state transposes between decode and chunk rounds).
        let use_v2 =
            self.routes.deltanet_v2 && self.head_k_dim == 128 && self.head_v_dim == 128;
        if !use_v2 {
            self.ensure_v1_state_layout()?;
        }
        let qkvz = self.in_proj_qkvz.forward(xs)?;
        let ba = self.in_proj_ba.forward(xs)?;
        let conv_state = self
            .conv_state
            .as_ref()
            .expect("fused decode requires a populated conv state")
            .clone();
        let dims = crate::fused_deltanet::GatedDeltaDims {
            heads: self.num_v_heads,
            dk: self.head_k_dim,
            dv: self.head_v_dim,
            conv_dim: self.conv_dim,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            ksz: self.conv_kernel_size,
        };
        if use_v2 {
            let state_t = self.take_state_for_v2(b_sz, xs.device())?;
            let (out, conv_new, state_new) = crate::fused_deltanet::gated_delta_v2_decode(
                &qkvz.flatten_all()?.contiguous()?,
                &ba.flatten_all()?.contiguous()?,
                b_sz,
                &conv_state,
                &state_t.contiguous()?,
                &crate::fused_deltanet::GatedDeltaDecodeWeights {
                    conv_weight: &self.conv_weight_full,
                    dt_bias_f32: &self.dt_bias_f32.flatten_all()?,
                    a_log_exp_f32: &self.a_log_exp_f32.flatten_all()?,
                    norm_weight_f32: &self.norm.weight_f32,
                    dims: &dims,
                    l2_eps: 1e-6,
                    norm_eps: self.norm.eps as f32,
                },
            )
            .map_err(candle::Error::wrap)?;
            self.conv_state = Some(conv_new);
            self.recurrent_state = Some(state_new);
            return self.out_proj.forward(&out);
        }
        let recurrent_state = match &self.recurrent_state {
            Some(state) if state.dtype() == DType::F32 => state.clone(),
            Some(state) => state.to_dtype(DType::F32)?,
            None => Tensor::zeros(
                (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                DType::F32,
                xs.device(),
            )?,
        };
        // v1 kernel keeps the packed [qkv | z | b | a] layout; ba's row
        // layout is exactly the packed tail, so a 2-tensor cat suffices.
        let proj = Tensor::cat(&[&qkvz, &ba], D::Minus1)?
            .flatten_all()?
            .contiguous()?;
        let (out, conv_new, state_new) = crate::fused_deltanet::gated_delta_decode(
            &proj,
            b_sz,
            &conv_state,
            &recurrent_state.contiguous()?,
            &self.conv_weight_full,
            &self.dt_bias_f32.flatten_all()?,
            &self.a_log_exp_f32.flatten_all()?,
            &self.norm.weight_f32,
            &dims,
            1e-6,
            self.norm.eps as f32,
        )
        .map_err(candle::Error::wrap)?;
        self.conv_state = Some(conv_new);
        self.recurrent_state = Some(state_new);
        self.out_proj.forward(&out)
    }

    fn fused_decode_eligible(&self, xs: &Tensor, b: usize, l: usize) -> bool {
        (1..=32).contains(&b)
            && l == 1
            && self.routes.fused_deltanet
            && matches!(xs.device(), Device::Metal(_))
            && xs.dtype() == DType::BF16
            && self.conv_state.is_some()
            && self.num_k_heads == self.num_v_heads
            && self.head_k_dim == self.head_v_dim
            && self.head_v_dim.is_multiple_of(32)
            && self.head_v_dim <= 256
    }

    fn fused_chunk_eligible(&self, xs: &Tensor, b: usize, l: usize) -> bool {
        b == 1
            && (2..=12).contains(&l)
            && self.routes.fused_deltanet
            && matches!(xs.device(), Device::Metal(_))
            && xs.dtype() == DType::BF16
            && self.conv_state.is_some()
            && self.num_k_heads == self.num_v_heads
            && self.head_k_dim == 128
            && self.head_v_dim == 128
    }

    /// Whole-layer fused chunk step (one dispatch, 2 <= l <= 12) with
    /// rollback-capture assembly matching the tensor path's semantics.
    fn forward_fused_chunk(&mut self, xs: &Tensor, l: usize) -> Result<Tensor> {
        self.ensure_v1_state_layout()?;
        let qkvz = self.in_proj_qkvz.forward(xs)?;
        let ba = self.in_proj_ba.forward(xs)?;
        let proj = Tensor::cat(&[&qkvz, &ba], D::Minus1)?; // [1, l, 8224] packed for v1
        let conv_state = self
            .conv_state
            .as_ref()
            .expect("fused chunk requires a populated conv state")
            .clone();
        let recurrent_state = match &self.recurrent_state {
            Some(state) if state.dtype() == DType::F32 => state.clone(),
            Some(state) => state.to_dtype(DType::F32)?,
            None => Tensor::zeros(
                (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                DType::F32,
                xs.device(),
            )?,
        };
        let dims = crate::fused_deltanet::GatedDeltaDims {
            heads: self.num_v_heads,
            dk: self.head_k_dim,
            dv: self.head_v_dim,
            conv_dim: self.conv_dim,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            ksz: self.conv_kernel_size,
        };
        let (out, conv_new, state_new, cap) = crate::fused_deltanet::gated_delta_chunk(
            &proj.flatten_to(1)?,
            l,
            &conv_state,
            &recurrent_state.contiguous()?,
            &self.conv_weight_full,
            &self.dt_bias_f32.flatten_all()?,
            &self.a_log_exp_f32.flatten_all()?,
            &self.norm.weight_f32,
            &dims,
            1e-6,
            self.norm.eps as f32,
        )
        .map_err(|e| candle::Error::msg(format!("{e:#}")))?;

        if self.verify_capture {
            // Same reconstruction contract as the tensor path; the conv
            // window assembles lazily on rollback from the raw projection.
            self.verify_captured = Some(DeltaVerifyCapture {
                s0: recurrent_state,
                kc: cap.kc,
                delta: cap.delta,
                gcs: cap.gcs,
                conv: ConvCapture::Lazy {
                    conv_prev: conv_state,
                    mixed_raw: proj.narrow(D::Minus1, 0, self.conv_dim)?,
                },
                dtype: xs.dtype(),
                transposed: false,
            });
        } else {
            self.verify_captured = None;
        }
        self.conv_state = Some(conv_new);
        self.recurrent_state = Some(state_new);
        self.out_proj.forward(&out)
    }

    /// Converts `recurrent_state` to the layout a v1 (or tensor) path
    /// expects, transposing back from the v2 layout if needed. dk == dv makes
    /// the shapes identical, so the flag is the only source of truth.
    fn ensure_v1_state_layout(&mut self) -> Result<()> {
        if self.state_transposed {
            if let Some(state) = &self.recurrent_state {
                self.recurrent_state = Some(state.transpose(2, 3)?.contiguous()?);
            }
            self.state_transposed = false;
        }
        Ok(())
    }

    /// F32, offset-0, v2-transposed (b, h, dv, dk) copy of the live state
    /// for the v2 kernels; flips the layout flag.
    fn take_state_for_v2(&mut self, b: usize, device: &Device) -> Result<Tensor> {
        let state = match &self.recurrent_state {
            Some(state) if state.dtype() == DType::F32 => state.clone(),
            Some(state) => state.to_dtype(DType::F32)?,
            None => Tensor::zeros(
                (b, self.num_v_heads, self.head_v_dim, self.head_k_dim),
                DType::F32,
                device,
            )?,
        };
        let state = if self.state_transposed || self.recurrent_state.is_none() {
            state
        } else {
            state.transpose(2, 3)?.contiguous()?
        };
        self.state_transposed = true;
        Ok(state)
    }

    fn fused_v2_eligible(&self, xs: &Tensor, b: usize, l: usize) -> bool {
        // Chunks only: the re-gridded core wins there and compounds with l
        // (measured -7%/-8% at l=8/12), but the three-dispatch structure
        // costs ~0.8 ms on a single-token step where the v1 whole-layer
        // kernel is ~0.9 ms total — decode stays on v1.
        self.routes.deltanet_v2
            && self.routes.fused_deltanet
            && b == 1
            && (2..=12).contains(&l)
            && matches!(xs.device(), Device::Metal(_))
            && xs.dtype() == DType::BF16
            && self.conv_state.is_some()
            && self.num_k_heads == self.num_v_heads
            && self.head_k_dim == 128
            && self.head_v_dim == 128
    }

    /// v2 fused path: unified decode/chunk with the re-gridded kernels and
    /// transposed state. Mirrors forward_fused_decode (l == 1: captures
    /// untouched, exactly like the v1 decode route) and forward_fused_chunk
    /// (l >= 2: capture assembly) semantics.
    fn forward_fused_v2(&mut self, xs: &Tensor, b: usize, l: usize) -> Result<Tensor> {
        let qkvz = self.in_proj_qkvz.forward(xs)?;
        let ba = self.in_proj_ba.forward(xs)?;
        let conv_state = self
            .conv_state
            .as_ref()
            .expect("fused v2 requires a populated conv state")
            .clone();
        let recurrent_state_t = self.take_state_for_v2(b, xs.device())?;
        let dims = crate::fused_deltanet::GatedDeltaDims {
            heads: self.num_v_heads,
            dk: self.head_k_dim,
            dv: self.head_v_dim,
            conv_dim: self.conv_dim,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            ksz: self.conv_kernel_size,
        };
        let (out, conv_new, state_new, cap) = crate::fused_deltanet::gated_delta_v2(
            &qkvz.flatten_all()?.contiguous()?,
            &ba.flatten_all()?.contiguous()?,
            l,
            b,
            &conv_state,
            &recurrent_state_t.contiguous()?,
            &self.conv_weight_full,
            &self.dt_bias_f32.flatten_all()?,
            &self.a_log_exp_f32.flatten_all()?,
            &self.norm.weight_f32,
            &dims,
            1e-6,
            self.norm.eps as f32,
        )
        .map_err(|e| candle::Error::msg(format!("{e:#}")))?;

        if l >= 2 {
            if self.verify_capture {
                self.verify_captured = Some(DeltaVerifyCapture {
                    s0: recurrent_state_t,
                    kc: cap.kc,
                    delta: cap.delta,
                    gcs: cap.gcs,
                    conv: ConvCapture::Lazy {
                        conv_prev: conv_state,
                        mixed_raw: qkvz.narrow(D::Minus1, 0, self.conv_dim)?,
                    },
                    dtype: xs.dtype(),
                    transposed: true,
                });
            } else {
                self.verify_captured = None;
            }
        }
        self.conv_state = Some(conv_new);
        self.recurrent_state = Some(state_new);
        self.out_proj.forward(&out)
    }

    /// Two-branch tree verify: segment decomposition over the flattened
    /// [anchor, a_1..a_w, b_1..b_w] layout. Projections/MLP shapes upstream
    /// ran once over the whole flattened chunk. On the v2 path both segments
    /// run in ONE fused dispatch — the alternate restarts inside the core
    /// kernel from the post-anchor state (returned as the alternate
    /// capture's S0). The v1 fallback runs the chunk kernel per segment with
    /// the branch seed composed on the host from the main capture's closed
    /// form.
    fn forward_tree(&mut self, xs: &Tensor, branch_width: usize) -> Result<Tensor> {
        let w = branch_width;
        let seg1 = w + 1;
        let l = 1 + 2 * w;
        if xs.dim(1)? != l {
            candle::bail!("tree forward expects {} tokens, got {}", l, xs.dim(1)?);
        }
        if !self.fused_chunk_eligible(xs, xs.dim(0)?, seg1) {
            candle::bail!("tree forward requires the fused chunk path (metal, BF16, w+1 <= 12)");
        }
        // The tree runs on the same kernel generation as chain verifies so
        // the tree-check equivalence gate compares like with like: v2 chunk
        // kernels (transposed state) when active, else v1.
        let use_v2 = self.routes.deltanet_v2;
        let qkvz = self.in_proj_qkvz.forward(xs)?;
        let ba = self.in_proj_ba.forward(xs)?;
        let conv_state = self
            .conv_state
            .as_ref()
            .expect("tree forward requires a populated conv state")
            .clone();
        let s0 = if use_v2 {
            self.take_state_for_v2(1, xs.device())?
        } else {
            self.ensure_v1_state_layout()?;
            match &self.recurrent_state {
                Some(state) if state.dtype() == DType::F32 => state.clone(),
                Some(state) => state.to_dtype(DType::F32)?,
                None => Tensor::zeros(
                    (1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                    DType::F32,
                    xs.device(),
                )?,
            }
        };
        let dims = crate::fused_deltanet::GatedDeltaDims {
            heads: self.num_v_heads,
            dk: self.head_k_dim,
            dv: self.head_v_dim,
            conv_dim: self.conv_dim,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            ksz: self.conv_kernel_size,
        };
        if use_v2 {
            // Single fused dispatch: the alternate segment restarts inside
            // the core kernel from the post-anchor state, which comes back
            // as state_mid — the alternate capture's S0. No host-composed
            // branch seed, no second per-layer kernel call.
            let (out, conv_main, state_main, state_mid, cap) =
                crate::fused_deltanet::gated_delta_v2_tree(
                    &qkvz.flatten_all()?.contiguous()?,
                    &ba.flatten_all()?.contiguous()?,
                    seg1,
                    w,
                    1,
                    &conv_state,
                    &s0.contiguous()?,
                    &self.conv_weight_full,
                    &self.dt_bias_f32.flatten_all()?,
                    &self.a_log_exp_f32.flatten_all()?,
                    &self.norm.weight_f32,
                    &dims,
                    1e-6,
                    self.norm.eps as f32,
                )
                .map_err(|e| candle::Error::msg(format!("{e:#}")))?;
            let mixed_t = qkvz
                .narrow(D::Minus1, 0, self.conv_dim)?
                .transpose(1, 2)?
                .contiguous()?; // [1, conv_dim, l]
            let ksz = self.conv_kernel_size;
            // Post-anchor conv window = last ksz of [conv_state | anchor
            // input] — the same view reconstruct_capture_state(cap_main, 1)
            // produces on the host-composed path.
            let branch_conv = Tensor::cat(&[&conv_state, &mixed_t.narrow(2, 0, 1)?], 2)?
                .narrow(2, 1, ksz)?;
            let conv_full_main = Tensor::cat(&[&conv_state, &mixed_t.narrow(2, 0, seg1)?], 2)?;
            let conv_full_alt = Tensor::cat(&[&branch_conv, &mixed_t.narrow(2, seg1, w)?], 2)?;
            let cap_main = DeltaVerifyCapture {
                s0: s0.clone(),
                kc: cap.kc.narrow(2, 0, seg1)?,
                delta: cap.delta.narrow(2, 0, seg1)?,
                gcs: cap.gcs.narrow(2, 0, seg1)?,
                conv: ConvCapture::Full {
                    conv_full: conv_full_main,
                    prev_conv_len: conv_state.dim(2)?,
                },
                dtype: xs.dtype(),
                transposed: true,
            };
            let cap_alt = DeltaVerifyCapture {
                s0: state_mid,
                kc: cap.kc.narrow(2, seg1, w)?,
                delta: cap.delta.narrow(2, seg1, w)?,
                gcs: cap.gcs.narrow(2, seg1, w)?,
                conv: ConvCapture::Full {
                    conv_full: conv_full_alt,
                    prev_conv_len: ksz,
                },
                dtype: xs.dtype(),
                transposed: true,
            };
            self.tree_captured = Some((cap_main, cap_alt));
            self.verify_captured = None;
            // Live state stays at the main segment's end, as on the
            // two-dispatch path.
            self.conv_state = Some(conv_main);
            self.recurrent_state = Some(state_main);
            return self.out_proj.forward(&out);
        }
        // v1 fallback: two dispatches with the host-composed branch seed on
        // the packed [qkv | z | b | a] layout.
        let proj = Tensor::cat(&[&qkvz, &ba], D::Minus1)?; // [1, l, 8224]
        let run_segment = |proj_seg: &Tensor,
                           seg_len: usize,
                           conv: &Tensor,
                           state: &Tensor|
         -> Result<(Tensor, Tensor, Tensor, DeltaVerifyCapture)> {
            let flat = proj_seg.flatten_to(1)?.contiguous()?;
            let (out, conv_new, state_new, cap) = crate::fused_deltanet::gated_delta_chunk(
                &flat,
                seg_len,
                conv,
                &state.contiguous()?,
                &self.conv_weight_full,
                &self.dt_bias_f32.flatten_all()?,
                &self.a_log_exp_f32.flatten_all()?,
                &self.norm.weight_f32,
                &dims,
                1e-6,
                self.norm.eps as f32,
            )
            .map_err(|e| candle::Error::msg(format!("{e:#}")))?;
            let capture = DeltaVerifyCapture {
                s0: state.clone(),
                kc: cap.kc,
                delta: cap.delta,
                gcs: cap.gcs,
                conv: ConvCapture::Lazy {
                    conv_prev: conv.clone(),
                    mixed_raw: proj_seg.narrow(D::Minus1, 0, self.conv_dim)?,
                },
                dtype: xs.dtype(),
                transposed: false,
            };
            Ok((out, conv_new, state_new, capture))
        };
        let proj_main = proj.narrow(1, 0, seg1)?;
        let proj_alt = proj.narrow(1, seg1, w)?;
        let (out_main, conv_main, state_main, cap_main) =
            run_segment(&proj_main, seg1, &conv_state, &s0)?;
        // Branch-point (post-anchor) state from the main segment's own
        // capture — the same closed form the partial-accept rollback uses.
        let (branch_rec, branch_conv) = Self::reconstruct_capture_state(
            &cap_main,
            1,
            self.conv_kernel_size,
            self.routes.fused_reconstruct,
        )?;
        let branch_rec_f32 = if branch_rec.dtype() == DType::F32 {
            branch_rec
        } else {
            branch_rec.to_dtype(DType::F32)?
        };
        let (out_alt, _conv_alt, _state_alt, cap_alt) =
            run_segment(&proj_alt, w, &branch_conv, &branch_rec_f32)?;
        self.tree_captured = Some((cap_main, cap_alt));
        self.verify_captured = None;
        // Leave the live state at the main segment's end; the runner always
        // installs the winner via select_tree_state before the next forward.
        self.conv_state = Some(conv_main);
        self.recurrent_state = Some(state_main);
        let out = Tensor::cat(&[&out_main, &out_alt], 1)?;
        self.out_proj.forward(&out)
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
        if self.fused_v2_eligible(xs, b, l) {
            return profiled(
                profiler,
                &device,
                Some((layer_index, "linear_attention")),
                if l == 1 { "deltanet_fused_v2_decode" } else { "deltanet_fused_v2_chunk" },
                l,
                offset,
                || self.forward_fused_v2(xs, b, l),
            );
        }
        if self.fused_decode_eligible(xs, b, l) {
            return profiled(
                profiler,
                &device,
                Some((layer_index, "linear_attention")),
                "deltanet_fused_decode",
                l,
                offset,
                || self.forward_fused_decode(xs),
            );
        }
        if self.fused_chunk_eligible(xs, b, l) {
            return profiled(
                profiler,
                &device,
                Some((layer_index, "linear_attention")),
                "deltanet_fused_chunk",
                l,
                offset,
                || self.forward_fused_chunk(xs, l),
            );
        }
        // One fused qkv+z projection; the z slice is consumed later at the
        // output-gate stage (same math, computed a stage earlier).
        let qkvz = profiled(
            profiler,
            &device,
            Some((layer_index, "linear_attention")),
            "deltanet_qkv_projection",
            l,
            offset,
            || self.in_proj_qkvz.forward(xs),
        )?;
        let z_flat = qkvz.narrow(D::Minus1, self.conv_dim, self.value_dim)?;
        let mixed = qkvz
            .narrow(D::Minus1, 0, self.conv_dim)?
            .transpose(1, 2)?;
        let mixed = profiled(
            profiler,
            &device,
            Some((layer_index, "linear_attention")),
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
            Some((layer_index, "linear_attention")),
            "deltanet_gates_and_repeat",
            l,
            offset,
            || {
                let ba = self.in_proj_ba.forward(xs)?;
                let beta = candle_nn::ops::sigmoid(
                    &ba.narrow(D::Minus1, 0, self.num_v_heads)?.contiguous()?,
                )?;
                let a = ba
                    .narrow(D::Minus1, self.num_v_heads, self.num_v_heads)?
                    .contiguous()?
                    .to_dtype(DType::F32)?;
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
            Some((layer_index, "linear_attention")),
            "deltanet_recurrent_rule",
            l,
            offset,
            || self.recurrent_delta_rule(&query, &key, &value, &g, &beta),
        )?;
        profiled(
            profiler,
            &device,
            Some((layer_index, "linear_attention")),
            "deltanet_output_gate_norm_projection",
            l,
            offset,
            || {
                let z = z_flat.reshape((
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
        if self.verify_capture && l > 1 {
            self.pending_conv_window = Some((full.clone(), start));
        }
        // Shifted-window depthwise conv: position t accumulates kernel taps in
        // ascending k order, matching the original per-position loop's
        // floating-point order exactly (left zero padding covers t < left).
        let ksz = self.conv_kernel_size;
        let window_len = l + ksz - 1;
        let padded = if start >= ksz - 1 {
            full.narrow(2, start - (ksz - 1), window_len)?
        } else {
            let deficit = (ksz - 1) - start;
            Tensor::cat(
                &[
                    &Tensor::zeros((b, c, deficit), xs.dtype(), xs.device())?,
                    &full.narrow(2, 0, start + l)?,
                ],
                2,
            )?
        };
        let mut acc = padded.narrow(2, 0, l)?.broadcast_mul(&self.conv_taps[0])?;
        for k in 1..ksz {
            acc = (acc + padded.narrow(2, k, l)?.broadcast_mul(&self.conv_taps[k])?)?;
        }
        let out = acc.silu()?;

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
        let (_, l, _, _) = query.dims4()?;
        if l == 1 {
            return self.recurrent_delta_rule_decode(query, key, value, g, beta);
        }
        if self.routes.deltanet_sequential_fallback {
            return self.recurrent_delta_rule_sequential(query, key, value, g, beta);
        }
        self.recurrent_delta_rule_chunked(query, key, value, g, beta)
    }

    /// Chunked (WY/UT-transform) gated delta rule for seq_len > 1.
    ///
    /// Algebraically identical to the sequential recurrence
    /// `S_t = (I - b_t k_t k_t^T) a_t S_{t-1} + b_t k_t v_t^T`, `o_t = q_t^T S_t`
    /// (a_t = exp(g_t) is a per-head scalar), evaluated per chunk via the
    /// pseudo-value solve `(I + B) D = diag(b)(V - diag(gamma) K S_0)` where
    /// `B[t,j] = b_t exp(G_t - G_j) k_t^T k_j` (strictly lower). All decay
    /// factors are relative (`exp(G_t - G_j) <= 1` for t >= j), so nothing
    /// overflows regardless of decay strength. `(I + B)^{-1}` is exact after
    /// ceil(log2 C) Neumann-doubling steps because B is nilpotent.
    fn recurrent_delta_rule_chunked(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        const CHUNK: usize = 32;
        self.ensure_v1_state_layout()?;
        let (b, l, h, k_dim) = query.dims4()?;
        let v_dim = value.dim(D::Minus1)?;
        let dtype = query.dtype();
        let device = query.device().clone();
        // A fresh forward invalidates any previous chunk's capture.
        self.verify_captured = None;
        let capture_this = self.verify_capture && l <= CHUNK;
        if !capture_this {
            self.pending_conv_window = None;
        }

        let q = (l2norm(query)?.to_dtype(DType::F32)? * (1.0 / (k_dim as f64).sqrt()))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = l2norm(key)?
            .to_dtype(DType::F32)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = value
            .to_dtype(DType::F32)?
            .transpose(1, 2)?
            .contiguous()?;
        let g = g.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
        let beta = beta.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;

        let mut state = match &self.recurrent_state {
            Some(state) if state.dtype() == DType::F32 => state.clone(),
            Some(state) => state.to_dtype(DType::F32)?,
            None => Tensor::zeros((b, h, k_dim, v_dim), DType::F32, &device)?,
        };

        let mut outs = Vec::with_capacity(l.div_ceil(CHUNK));
        let mut start = 0usize;
        while start < l {
            let c = CHUNK.min(l - start);
            let qc = q.narrow(2, start, c)?.contiguous()?;
            let kc = k.narrow(2, start, c)?.contiguous()?;
            let vc = v.narrow(2, start, c)?.contiguous()?;
            let gc = g.narrow(2, start, c)?.contiguous()?;
            let bc = beta.narrow(2, start, c)?.contiguous()?;

            let gcs = gc.cumsum(D::Minus1)?;
            let gamma = gcs.exp()?;
            // rel[t, j] = exp(G_t - G_j); clamp keeps masked upper entries
            // finite before the triangular masks zero them.
            let rel = gcs
                .unsqueeze(3)?
                .broadcast_sub(&gcs.unsqueeze(2)?)?
                .clamp(-1e30, 0.0)?
                .exp()?;
            let tril_incl = Tensor::tril2(c, DType::F32, &device)?;
            let eye = Tensor::eye(c, DType::F32, &device)?;
            let tril_strict = (&tril_incl - &eye)?;

            let kk = kc.matmul(&kc.transpose(2, 3)?.contiguous()?)?;
            let bmat = rel
                .mul(&kk)?
                .broadcast_mul(&tril_strict)?
                .broadcast_mul(&bc.unsqueeze(3)?)?;
            let ks0 = kc.matmul(&state)?;
            let r = (vc - ks0.broadcast_mul(&gamma.unsqueeze(3)?)?)?
                .broadcast_mul(&bc.unsqueeze(3)?)?;

            let mut e = bmat.neg()?;
            let mut x = e.broadcast_add(&eye)?;
            let steps = if c <= 1 {
                0
            } else {
                usize::BITS as usize - ((c - 1).leading_zeros() as usize)
            };
            for _ in 0..steps {
                e = e.matmul(&e)?;
                x = (&x + e.matmul(&x)?)?;
            }
            let delta = x.matmul(&r)?;

            if capture_this {
                let Some((conv_full, prev_conv_len)) = self.pending_conv_window.take() else {
                    candle::bail!("verify capture enabled but no conv window was staged");
                };
                // `state` still holds S0 here; the chunk-end update below
                // replaces the binding rather than mutating the tensor.
                self.verify_captured = Some(DeltaVerifyCapture {
                    s0: state.clone(),
                    kc: kc.clone(),
                    delta: delta.clone(),
                    gcs: gcs.clone(),
                    conv: ConvCapture::Full {
                        conv_full,
                        prev_conv_len,
                    },
                    dtype,
                    transposed: false,
                });
            }

            let qs0 = qc
                .matmul(&state)?
                .broadcast_mul(&gamma.unsqueeze(3)?)?;
            let qk = qc.matmul(&kc.transpose(2, 3)?.contiguous()?)?;
            let n = rel.mul(&qk)?.broadcast_mul(&tril_incl)?;
            outs.push((qs0 + n.matmul(&delta)?)?);

            let g_last = gcs.narrow(2, c - 1, 1)?;
            let decay_to_end = g_last.broadcast_sub(&gcs)?.exp()?;
            let kbar = kc.broadcast_mul(&decay_to_end.unsqueeze(3)?)?;
            let gamma_c = g_last.exp()?.unsqueeze(3)?;
            state = (state.broadcast_mul(&gamma_c)?
                + kbar.transpose(2, 3)?.contiguous()?.matmul(&delta)?)?;
            start += c;
        }

        // One boundary cast per chunk is cheap; the decode path (per-token)
        // is where the cast pair costs and keeps F32 instead.
        self.recurrent_state = Some(state.to_dtype(dtype)?);
        let out_refs = outs.iter().collect::<Vec<_>>();
        Tensor::cat(&out_refs, 2)?.transpose(1, 2)?.to_dtype(dtype)
    }

    fn recurrent_delta_rule_sequential(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        self.ensure_v1_state_layout()?;
        let (b, l, h, k_dim) = query.dims4()?;
        let v_dim = value.dim(D::Minus1)?;
        let dtype = query.dtype();
        // The sequential fallback never captures verify intermediates.
        self.verify_captured = None;
        self.pending_conv_window = None;
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
        self.ensure_v1_state_layout()?;
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
        // The state stays F32 across decode steps: the recurrence compounds
        // BF16 rounding every token, and the cast pair costs 2 dispatches x
        // 18 layers per token. Cross-path numerics (a BF16-stored chunk state
        // feeding an F32-resident decode run, or a 1-token rollback
        // re-advance landing here) shift logits by at most a few BF16 ulps,
        // which the rollback oracle's logit-noise bound covers explicitly.
        self.recurrent_state = Some(state);
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
    layer: Option<(usize, &'static str)>,
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
            layer.map(|(index, _)| index),
            layer.map(|(_, kind)| kind),
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

/// Row-major ancestor-visibility data for [`Qwen35TextModel::tree_mask`]
/// ((1 + 2w) x (offset + 1 + 2w); 0 = visible, -inf = masked), pure so the
/// visibility rules are unit-testable without a model.
fn tree_mask_data(w: usize, offset: usize) -> Vec<f32> {
    let l = 1 + 2 * w;
    let total = offset + l;
    let mut data = vec![f32::NEG_INFINITY; l * total];
    for row in 0..l {
        for col in 0..=offset {
            data[row * total + col] = 0.0; // history + anchor (col offset)
        }
        let (seg_start, seg_pos) = if row == 0 {
            (0, 0)
        } else if row <= w {
            (1, row) // a-segment, 1-based index within branch
        } else {
            (w + 1, row - w) // b-segment
        };
        if seg_pos > 0 {
            for i in 0..seg_pos {
                data[row * total + offset + seg_start + i] = 0.0;
            }
        }
    }
    data
}

#[cfg(test)]
#[path = "qwen35_tests.rs"]
mod tests;

#[derive(Clone, Debug)]
enum TokenMixer {
    Full(FullAttention),
    Linear(Box<GatedDeltaNet>),
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
    fn new<S: LinearSource>(
        layer_type: LayerType,
        cfg: &TextConfig,
        rotary: Arc<RotaryEmbedding>,
        source: &S,
        ctx: &ModelCtx,
    ) -> Result<Self> {
        let mixer = match layer_type {
            LayerType::FullAttention => {
                TokenMixer::Full(FullAttention::new(cfg, rotary, &source.sub("self_attn"), ctx)?)
            }
            LayerType::LinearAttention => TokenMixer::Linear(Box::new(GatedDeltaNet::new(
                cfg,
                &source.sub("linear_attn"),
                ctx,
            )?)),
        };
        Ok(Self {
            mixer,
            mlp: Mlp::new(cfg, &source.sub("mlp"))?,
            input_layernorm: Qwen35RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                true,
                &source.sub("input_layernorm"),
                ctx,
            )?,
            post_attention_layernorm: Qwen35RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                true,
                &source.sub("post_attention_layernorm"),
                ctx,
            )?,
        })
    }

    fn kind(&self) -> &'static str {
        self.mixer.kind()
    }

    /// One decoder block over pre-normed input: `normed` is
    /// input_layernorm(hidden), computed by the caller's loop — which fuses
    /// it into the PREVIOUS layer's MLP residual add (cross-layer add+norm:
    /// one dispatch instead of badd + rmsnorm + their barrier, ~24 serial
    /// dispatches per token across the stack). Returns (hidden_next,
    /// normed_next) where normed_next is `next_norm` applied to hidden_next
    /// in the same fused tail — the following layer's input norm, or the
    /// model's final norm on the last layer. The sums stay bit-identical to
    /// the unfused badd (per-layer captures/traces are unchanged); only the
    /// normed outputs carry the sub-noise F32-sum-input difference (margin
    /// class, same as the post-attention site).
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        hidden: &Tensor,
        normed: &Tensor,
        next_norm: &Qwen35RmsNorm,
        mask: Option<&Tensor>,
        offset: usize,
        layer_index: usize,
        profiler: Option<&Qwen35Profiler>,
        tree: Option<&TreeForward>,
    ) -> Result<(Tensor, Tensor)> {
        let (_, seq_len, _) = hidden.dims3()?;
        let device = hidden.device().clone();
        let layer_kind = self.mixer.kind();
        let mixed = match (&mut self.mixer, tree) {
            (TokenMixer::Full(attn), _) => attn.forward(
                normed,
                mask,
                offset,
                layer_index,
                profiler,
                tree.map(|t| t.positions.as_slice()),
            )?,
            (TokenMixer::Linear(attn), None) => {
                attn.forward(normed, layer_index, offset, profiler)?
            }
            (TokenMixer::Linear(attn), Some(tree)) => {
                attn.forward_tree(normed, tree.branch_width)?
            }
        };
        // Fused residual-add + post-attention norm: one dispatch emits both
        // the new residual (hidden + mixer_out) and its norm (fed to MLP).
        let (xs, mlp_in) = profiled(
            profiler,
            &device,
            Some((layer_index, layer_kind)),
            "post_attention_layernorm",
            seq_len,
            offset,
            || self.post_attention_layernorm.forward_add(hidden, &mixed),
        )?;
        let mlp_out = profiled(
            profiler,
            &device,
            Some((layer_index, layer_kind)),
            "mlp",
            seq_len,
            offset,
            || self.mlp.forward(&mlp_in),
        )?;
        // Fused MLP residual + the NEXT norm (cross-layer).
        profiled(
            profiler,
            &device,
            Some((layer_index, layer_kind)),
            "mlp_residual_next_norm",
            seq_len,
            offset,
            || next_norm.forward_add(&xs, &mlp_out),
        )
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

/// The surface the decode loop needs from a causal text model. Static-dispatch
/// (`generate` is generic over it) so the hot loop stays monomorphized.
pub trait CausalTextModel {
    fn clear_cache(&mut self);
    /// Forward `input_ids` [1, seq] at `offset`; returns last-position logits
    /// [1, vocab].
    fn forward(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor>;
}

/// Text-only causal LM over [`Qwen35TextModel`] (the ternary GGUF path). Wraps
/// the shared trunk with the `output`-weight head that `MiniCpm` owns for the
/// vision model — no vision prefill.
#[derive(Clone, Debug)]
pub struct Qwen35CausalLM {
    model: Qwen35TextModel,
    lm_head: MixedLinear,
    device: Device,
    dtype: DType,
}

impl Qwen35CausalLM {
    pub fn new<S: LinearSource>(cfg: &TextConfig, source: &S, ctx: &ModelCtx) -> Result<Self> {
        let model = Qwen35TextModel::new(cfg, source, ctx)?;
        let lm_head = source.linear(
            &[LinearPart::new("lm_head.weight", cfg.vocab_size)],
            cfg.hidden_size,
            false,
        )?;
        Ok(Self {
            model,
            lm_head,
            device: source.device().clone(),
            dtype: source.dtype(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

impl CausalTextModel for Qwen35CausalLM {
    fn clear_cache(&mut self) {
        self.model.clear_cache();
    }

    fn forward(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let hidden = self
            .model
            .forward_ids(input_ids, offset)?
            .narrow(1, seq_len - 1, 1)?
            .contiguous()?;
        self.lm_head.forward(&hidden)?.squeeze(1)
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
    // On-device capture of selected layer outputs for the DSpark drafter's
    // fused context (full sequence, no CPU copies; ascending layer order).
    device_capture_layers: Option<Vec<usize>>,
    device_captures: Vec<Tensor>,
}

impl Qwen35TextModel {
    pub fn new<S: LinearSource>(cfg: &TextConfig, source: &S, ctx: &ModelCtx) -> Result<Self> {
        let embed_w = source.tensor(
            (cfg.vocab_size, cfg.hidden_size).into(),
            "embed_tokens.weight",
        )?;
        let embed_tokens = Embedding::new(embed_w, cfg.hidden_size);
        let rotary = Arc::new(RotaryEmbedding::new(cfg, source.dtype(), source.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let layers_src = source.sub("layers");
        for idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                cfg.layer_types[idx],
                cfg,
                rotary.clone(),
                &layers_src.sub(&idx.to_string()),
                ctx,
            )?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: Qwen35RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                true,
                &source.sub("norm"),
                ctx,
            )?,
            device: source.device().clone(),
            dtype: source.dtype(),
            profiler: None,
            trace_recorder: None,
            device_capture_layers: None,
            device_captures: Vec::new(),
        })
    }

    pub fn set_device_capture(&mut self, layers: Option<Vec<usize>>) {
        self.device_capture_layers = layers.map(|mut layers| {
            layers.sort_unstable();
            layers.dedup();
            layers
        });
        self.device_captures.clear();
    }

    /// Captured layer outputs of the most recent forward, ascending layer
    /// order, each [b, seq, hidden] on device.
    pub fn take_device_captures(&mut self) -> Vec<Tensor> {
        std::mem::take(&mut self.device_captures)
    }

    pub fn embeddings(&self) -> &Tensor {
        self.embed_tokens.embeddings()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
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
        let capture_layers = self.device_capture_layers.clone();
        let mut captures = Vec::new();
        // Cross-layer fused add+norm: the loop feeds each layer its
        // pre-normed input and hands it the NEXT norm to fuse into its MLP
        // residual tail (norm handles are Arc-cheap clones; the layer loop
        // holds &mut self.layers). Layer 0's input norm runs standalone; the
        // last layer's tail applies the model's final norm, so the loop ends
        // holding the final-normed hidden.
        let input_norms: Vec<Qwen35RmsNorm> = self
            .layers
            .iter()
            .map(|layer| layer.input_layernorm.clone())
            .collect();
        let mut hidden = inputs_embeds.clone();
        let mut normed = profiled(
            profiler.as_ref(),
            &self.device,
            Some((0, "input_layernorm")),
            "input_layernorm",
            l,
            offset,
            || input_norms[0].forward(&hidden),
        )?;
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            let layer_kind = layer.kind();
            let next_norm = input_norms.get(layer_index + 1).unwrap_or(&self.norm);
            let (h_next, n_next) = layer.forward(
                &hidden,
                &normed,
                next_norm,
                mask.as_ref(),
                offset,
                layer_index,
                profiler.as_ref(),
                None,
            )?;
            hidden = h_next;
            normed = n_next;
            if let Some(trace_recorder) = trace_recorder.as_ref() {
                trace_recorder.record_layer(layer_index, layer_kind, &hidden, offset)?;
            }
            if let Some(capture_layers) = capture_layers.as_ref() {
                if capture_layers.contains(&layer_index) {
                    captures.push(hidden.clone());
                }
            }
        }
        if capture_layers.is_some() {
            self.device_captures = captures;
        }
        // `normed` is final_norm(hidden) via the last layer's fused tail.
        Ok(normed)
    }

    /// Two-branch tree verify forward over the flattened
    /// [anchor, a_1..a_w, b_1..b_w] layout (both branches continue from the
    /// anchor; siblings share rotary positions). Returns final-norm hidden
    /// states for all 1 + 2w rows. Commit the winner with
    /// [`Self::rollback_tree`] before the next forward.
    pub fn forward_tree_embeds(
        &mut self,
        inputs_embeds: &Tensor,
        offset: usize,
        branch_width: usize,
    ) -> Result<Tensor> {
        let w = branch_width;
        let (b, l, _) = inputs_embeds.dims3()?;
        if w == 0 || l != 1 + 2 * w || b != 1 {
            candle::bail!(
                "tree forward expects [1, 1 + 2w, hidden] with w >= 1; got b={b} l={l} w={w}"
            );
        }
        let mut positions = Vec::with_capacity(l);
        positions.push(offset as u32);
        for i in 1..=w {
            positions.push((offset + i) as u32);
        }
        for i in 1..=w {
            positions.push((offset + i) as u32);
        }
        let tree = TreeForward {
            branch_width: w,
            positions,
            mask: self.tree_mask(w, offset)?,
        };
        let profiler = self.profiler.clone();
        let capture_layers = self.device_capture_layers.clone();
        let mut captures = Vec::new();
        // Same cross-layer fused add+norm structure as forward_embeds.
        let input_norms: Vec<Qwen35RmsNorm> = self
            .layers
            .iter()
            .map(|layer| layer.input_layernorm.clone())
            .collect();
        let mut hidden = inputs_embeds.clone();
        let mut normed = input_norms[0].forward(&hidden)?;
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            let next_norm = input_norms.get(layer_index + 1).unwrap_or(&self.norm);
            let (h_next, n_next) = layer.forward(
                &hidden,
                &normed,
                next_norm,
                Some(&tree.mask),
                offset,
                layer_index,
                profiler.as_ref(),
                Some(&tree),
            )?;
            hidden = h_next;
            normed = n_next;
            if let Some(capture_layers) = capture_layers.as_ref() {
                if capture_layers.contains(&layer_index) {
                    captures.push(hidden.clone());
                }
            }
        }
        if capture_layers.is_some() {
            self.device_captures = captures;
        }
        // `normed` is final_norm(hidden) via the last layer's fused tail.
        Ok(normed)
    }

    /// Ancestor mask for the flattened tree layout: every row sees history
    /// and the anchor; a-rows additionally see earlier a-rows, b-rows earlier
    /// b-rows. (1, 1, 1 + 2w, offset + 1 + 2w), 0 = visible, -inf = masked.
    fn tree_mask(&self, w: usize, offset: usize) -> Result<Tensor> {
        let l = 1 + 2 * w;
        let total = offset + l;
        let data = tree_mask_data(w, offset);
        Tensor::from_vec(data, (1, 1, l, total), &self.device)?.to_dtype(self.dtype)
    }

    /// Installs the winner path after a tree verify: `accepted` is the number
    /// of accepted draft tokens along the winning branch (0..=w; 0 only valid
    /// on the main branch — the branches are identical at zero). DeltaNet
    /// states select from the winning segment's capture; full-attention KV
    /// keeps [anchor + winner rows], compacting the alternate's rows down
    /// over the main branch's when the alternate wins.
    pub fn rollback_tree(
        &mut self,
        snapshot: &DecodeStateSnapshot,
        branch_width: usize,
        on_alt: bool,
        accepted: usize,
    ) -> Result<()> {
        if on_alt && accepted == 0 {
            candle::bail!("zero-accept tree rollback must use the main branch");
        }
        if accepted > branch_width {
            candle::bail!("accepted {accepted} exceeds branch width {branch_width}");
        }
        let prefix_in_segment = if on_alt { accepted } else { accepted + 1 };
        // Full main accept: forward_tree already left the DeltaNet states at
        // the main segment's kernel chunk end — exact, no reconstruction
        // needed (or wanted: the closed form carries rollback-class FP noise).
        let full_main = !on_alt && accepted == branch_width;
        let mut full_idx = 0usize;
        for layer in &mut self.layers {
            match &mut layer.mixer {
                TokenMixer::Linear(attn) => {
                    if full_main {
                        attn.tree_captured = None;
                    } else {
                        attn.select_tree_state(on_alt, prefix_in_segment)?;
                    }
                }
                TokenMixer::Full(attn) => {
                    let Some(history) = snapshot.kv_lens.get(full_idx) else {
                        candle::bail!("decode-state snapshot has too few KV entries");
                    };
                    if on_alt {
                        attn.compact_tree_kv(*history, branch_width, accepted)?;
                    } else {
                        attn.truncate_kv(history + 1 + accepted)?;
                    }
                    full_idx += 1;
                }
            }
        }
        Ok(())
    }

    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    pub fn snapshot_decode_state(&self) -> DecodeStateSnapshot {
        let mut deltanet = Vec::new();
        let mut kv_lens = Vec::new();
        for layer in &self.layers {
            match &layer.mixer {
                TokenMixer::Linear(attn) => {
                    deltanet.push((
                        attn.conv_state.clone(),
                        attn.recurrent_state.clone(),
                        attn.state_transposed,
                    ));
                }
                TokenMixer::Full(attn) => kv_lens.push(attn.kv_len()),
            }
        }
        DecodeStateSnapshot { deltanet, kv_lens }
    }

    /// Enables/disables retention of verify-chunk reconstruction
    /// intermediates in the DeltaNet layers (see [`DeltaVerifyCapture`]).
    pub fn set_verify_state_capture(&mut self, on: bool) {
        for layer in &mut self.layers {
            if let TokenMixer::Linear(attn) = &mut layer.mixer {
                attn.verify_capture = on;
                if !on {
                    attn.verify_captured = None;
                    attn.pending_conv_window = None;
                }
            }
        }
    }

    /// Partial-accept rollback without a re-advance forward: DeltaNet states
    /// are reconstructed at `prefix_len` from the captured verify chunk, and
    /// full-attention KV keeps the chunk's first `prefix_len` rows (they are
    /// causally valid — each row depends only on its own position's hidden,
    /// computed under the true pre-chunk state).
    pub fn rollback_to_prefix(
        &mut self,
        snapshot: &DecodeStateSnapshot,
        prefix_len: usize,
    ) -> Result<()> {
        let mut full_idx = 0usize;
        for layer in &mut self.layers {
            match &mut layer.mixer {
                TokenMixer::Linear(attn) => attn.select_verify_state(prefix_len)?,
                TokenMixer::Full(attn) => {
                    let Some(len) = snapshot.kv_lens.get(full_idx) else {
                        candle::bail!("decode-state snapshot has too few KV entries");
                    };
                    attn.truncate_kv(len + prefix_len)?;
                    full_idx += 1;
                }
            }
        }
        Ok(())
    }

    pub fn restore_decode_state(&mut self, snapshot: &DecodeStateSnapshot) -> Result<()> {
        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        for layer in &mut self.layers {
            match &mut layer.mixer {
                TokenMixer::Linear(attn) => {
                    let Some((conv, recurrent, transposed)) = snapshot.deltanet.get(linear_idx)
                    else {
                        candle::bail!("decode-state snapshot has too few DeltaNet entries");
                    };
                    attn.conv_state = conv.clone();
                    attn.recurrent_state = recurrent.clone();
                    attn.state_transposed = *transposed;
                    linear_idx += 1;
                }
                TokenMixer::Full(attn) => {
                    let Some(len) = snapshot.kv_lens.get(full_idx) else {
                        candle::bail!("decode-state snapshot has too few KV entries");
                    };
                    attn.truncate_kv(*len)?;
                    full_idx += 1;
                }
            }
        }
        if linear_idx != snapshot.deltanet.len() || full_idx != snapshot.kv_lens.len() {
            candle::bail!(
                "decode-state snapshot layer counts do not match the model: \
                 {linear_idx}/{} DeltaNet, {full_idx}/{} full-attention",
                snapshot.deltanet.len(),
                snapshot.kv_lens.len()
            );
        }
        Ok(())
    }

    pub fn set_profiler(&mut self, profiler: Option<Qwen35Profiler>) {
        self.profiler = profiler;
    }

    pub fn set_trace_recorder(&mut self, trace_recorder: Option<Qwen35TraceRecorder>) {
        self.trace_recorder = trace_recorder;
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        causal_mask(b, tgt, offset, self.dtype, &self.device)
    }
}

// On-device construction: log(tril) is exactly 0 on visible entries and
// -inf on masked ones, replacing the O(tgt * total) CPU build + upload.
fn causal_mask(b: usize, tgt: usize, offset: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let causal = Tensor::tril2(tgt, DType::F32, device)?.log()?;
    let mask = if offset > 0 {
        let visible = Tensor::zeros((tgt, offset), DType::F32, device)?;
        Tensor::cat(&[&visible, &causal], 1)?
    } else {
        causal
    };
    mask.reshape((1, 1, tgt, tgt + offset))?
        .broadcast_as((b, 1, tgt, tgt + offset))?
        .to_dtype(dtype)
}

/// Qwen3.5 multi-token-prediction head, transplanted from the base
/// checkpoint's 15 `mtp.*` tensors (the MiniCPM finetune ships none):
/// `fc([norm(embed(t_{p+1})); norm(hidden_p)])` -> one standard
/// full-attention decoder layer with its OWN plain KV cache -> norm. Logits
/// come from the caller's lm_head over the returned post-norm hidden.
///
/// Reference semantics (vLLM qwen3_5_mtp.py + mlx-lm PR #1456, which agree
/// exactly): the layer attends only within its own cache — never the
/// trunk's; its rope position is the cache offset (uniformly absolute-1,
/// which cancels in q.k inside a cache-internal attention); rollback is a
/// plain truncate by the rejected count.
#[derive(Clone, Debug)]
pub struct MtpHead {
    fc: MixedLinear,
    pre_fc_norm_embedding: Qwen35RmsNorm,
    pre_fc_norm_hidden: Qwen35RmsNorm,
    norm: Qwen35RmsNorm,
    layer: DecoderLayer,
    device: Device,
    dtype: DType,
    routes: Arc<KernelRouteConfig>,
}

/// Profiler label slot for the MTP layer (out of the trunk's 0..24 range).
const MTP_LAYER_INDEX: usize = 1000;

impl MtpHead {
    /// Loads from a VarBuilder rooted at the checkpoint (the head's tensors
    /// live under `mtp.*`). `cfg` is the TARGET tower's text config — the
    /// base checkpoint is architecturally identical (verified: hidden 1024,
    /// 8 q / 2 kv heads x 256, mlp 3584) — and all norms load zero-centred
    /// (raw Qwen checkpoints store `w - 1`; Qwen35RmsNorm shifts at load).
    pub fn new<S: LinearSource>(
        cfg: &crate::config::TextConfig,
        source: &S,
        ctx: &ModelCtx,
    ) -> Result<Self> {
        let source = source.sub("mtp");
        // Fresh rope table from the same cfg/dtype/device as the trunk's —
        // bit-identical construction, no need to thread the trunk's Arc.
        let rotary = Arc::new(RotaryEmbedding::new(cfg, source.dtype(), source.device())?);
        let layers_src = source.sub("layers");
        Ok(Self {
            fc: source.linear(
                &[LinearPart::new("fc.weight", cfg.hidden_size)],
                2 * cfg.hidden_size,
                false,
            )?,
            pre_fc_norm_embedding: Qwen35RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                true,
                &source.sub("pre_fc_norm_embedding"),
                ctx,
            )?,
            pre_fc_norm_hidden: Qwen35RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                true,
                &source.sub("pre_fc_norm_hidden"),
                ctx,
            )?,
            norm: Qwen35RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                true,
                &source.sub("norm"),
                ctx,
            )?,
            layer: DecoderLayer::new(
                crate::config::LayerType::FullAttention,
                cfg,
                rotary,
                &layers_src.sub("0"),
                ctx,
            )?,
            device: source.device().clone(),
            dtype: source.dtype(),
            routes: ctx.routes.clone(),
        })
    }

    /// One MTP forward over S (hidden, successor-token) pairs. `hidden` is
    /// [1, S, H]: post-final-norm TRUNK hiddens at positions p..p+S-1 for
    /// catch-up, or this head's own post-norm output when chaining draft
    /// depth. `embeds` is [1, S, H]: embeddings of the tokens at positions
    /// p+1..p+S (each hidden's committed or drafted successor). Appends S
    /// entries to the head's cache and returns the post-norm hidden
    /// [1, S, H] — row k predicts the token at position p+k+2 through the
    /// caller's lm_head.
    pub fn step(&mut self, hidden: &Tensor, embeds: &Tensor) -> Result<Tensor> {
        let (b, s, _) = hidden.dims3()?;
        let x = self.fc_prep(hidden, embeds, b * s)?;
        let offset = self.kv_len();
        let mask = if s > 1 {
            Some(causal_mask(b, s, offset, self.dtype, &self.device)?)
        } else {
            None
        };
        // The layer's fused tail applies the head's final norm directly
        // (next_norm = mtp.norm), so the returned normed IS the post hidden.
        let normed_in = self.layer.input_layernorm.forward(&x)?;
        let (_, post) = self.layer.forward(
            &x,
            &normed_in,
            &self.norm,
            mask.as_ref(),
            offset,
            MTP_LAYER_INDEX,
            None,
            None,
        )?;
        Ok(post)
    }

    /// The pre-fc prologue: `fc(cat(norm_e(embeds), norm_h(hidden)))`. At
    /// m == 1 (every chain-step draft) the fused one-dispatch kernel replaces
    /// the 4-dispatch chain bit-identically; m > 1 (catch-up flush, priming)
    /// keeps the unfused chain — its GEMM accumulation order is not
    /// reproducible in the fused GEMV shape, and it runs once per round.
    fn fc_prep(&self, hidden: &Tensor, embeds: &Tensor, m: usize) -> Result<Tensor> {
        if m == 1
            && self.routes.fused_mtp_fc
            && matches!(hidden.device(), Device::Metal(_))
            && hidden.dtype() == DType::BF16
            && embeds.dtype() == DType::BF16
            && crate::fused_mtp::eligible_h_dim(hidden.dim(D::Minus1)?)
        {
            if let MixedLinear::Dense(lin) = &self.fc {
                let w = lin.weight();
                let (out_dim, k_dim) = w.dims2()?;
                if lin.bias().is_none()
                    && w.dtype() == DType::BF16
                    && k_dim == 2 * hidden.dim(D::Minus1)?
                    && out_dim.is_multiple_of(16)
                {
                    return crate::fused_mtp::fused_mtp_fc_prep(
                        embeds,
                        hidden,
                        &self.pre_fc_norm_embedding.weight_native,
                        &self.pre_fc_norm_hidden.weight_native,
                        w,
                        self.pre_fc_norm_embedding.eps as f32,
                    )
                    .map_err(|e| candle::Error::Msg(format!("fused mtp_fc_prep: {e:#}")));
                }
            }
        }
        let e = self.pre_fc_norm_embedding.forward(embeds)?;
        let h = self.pre_fc_norm_hidden.forward(hidden)?;
        self.fc.forward(&Tensor::cat(&[&e, &h], D::Minus1)?)
    }

    /// Quantizes the head's dense linears in place (fc, qkv, o_proj,
    /// gate_up, down — ~37MB -> ~10MB of reads per chain step at q4k).
    /// Draft-side only: the target's verify is untouched, so committed
    /// output stays lossless; drafted tokens are margin-class (acceptance
    /// is the arbiter). Biased linears are skipped (MixedLinear's quantized
    /// form carries no bias; this model's attention_bias is false). Pack
    /// hits upload the stored blocks bit-identically (same CPU quantizer);
    /// misses record for the pack write — `key_prefix` must encode the
    /// drafter checkpoint identity + tier so checkpoints coexist.
    pub fn quantize_with_pack(
        &mut self,
        ggml: candle::quantized::GgmlDType,
        pack: Option<&crate::pack::PackStore>,
        key_prefix: &str,
        ctx: &ModelCtx,
        only: Option<&str>,
    ) -> Result<()> {
        fn q(
            lin: &mut MixedLinear,
            ggml: candle::quantized::GgmlDType,
            pack: Option<&crate::pack::PackStore>,
            key: String,
            ctx: &ModelCtx,
        ) -> Result<()> {
            if let Some(pack) = pack {
                if let Some(qt) = pack.take(&key) {
                    *lin = ctx.quantized_linear(qt)?;
                    return Ok(());
                }
            }
            let qt = {
                if let MixedLinear::Dense(inner) = &*lin {
                    if inner.bias().is_some() {
                        return Ok(());
                    }
                }
                let Some(w) = lin.dense_weight() else {
                    return Ok(());
                };
                let device = w.device().clone();
                let cpu = w.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
                match pack {
                    Some(pack) => pack
                        .quantize_and_record(&key, &cpu, ggml)
                        .map_err(|e| candle::Error::Msg(format!("pack {key}: {e:#}")))?,
                    None => candle::quantized::QTensor::quantize_onto(&cpu, ggml, &device)?,
                }
            };
            *lin = ctx.quantized_linear(qt)?;
            Ok(())
        }
        // Bisection hook (diagnostics): `only` = fc|qkv|o|gate_up|down
        // quantizes a single linear, isolating per-path damage (filed q8_0
        // m-in-[2,8] anomaly, 2026-07-15). Entrypoint-resolved
        // (LMBRRR_MTP_Q_ONLY), threaded as a param.
        let want = |name: &str| only.is_none_or(|o| o == name);
        if want("fc") {
            q(&mut self.fc, ggml, pack, format!("{key_prefix}fc"), ctx)?;
        }
        match &mut self.layer.mixer {
            TokenMixer::Full(attn) => {
                if want("qkv") {
                    q(&mut attn.qkv_proj, ggml, pack, format!("{key_prefix}qkv"), ctx)?;
                }
                if want("o") {
                    q(&mut attn.o_proj, ggml, pack, format!("{key_prefix}o"), ctx)?;
                }
            }
            TokenMixer::Linear(_) => unreachable!("MTP layer is full attention"),
        }
        if want("gate_up") {
            q(
                &mut self.layer.mlp.gate_up_proj,
                ggml,
                pack,
                format!("{key_prefix}gate_up"),
                ctx,
            )?;
        }
        if want("down") {
            q(
                &mut self.layer.mlp.down_proj,
                ggml,
                pack,
                format!("{key_prefix}down"),
                ctx,
            )?;
        }
        Ok(())
    }

    /// Pairs fed so far (the head's rope offset).
    pub fn kv_len(&self) -> usize {
        match &self.layer.mixer {
            TokenMixer::Full(attn) => attn.kv_len(),
            TokenMixer::Linear(_) => unreachable!("MTP layer is full attention"),
        }
    }

    /// Rollback: keep the first `len` pairs (plain cache truncate).
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        match &self.layer.mixer {
            TokenMixer::Full(attn) => attn.truncate_kv(len),
            TokenMixer::Linear(_) => unreachable!("MTP layer is full attention"),
        }
    }

    pub fn clear(&mut self) {
        self.layer.clear_cache();
    }
}
