//! Host-side wrapper for the fork's fused attention-prep kernels.
//!
//! Per full-attention layer the unfused chain is ten tiny dispatches (two
//! strided-narrow copies, two rmsnorms, two partial ropes, a v copy, two
//! cache slice_sets and a contiguous fix-up), each a serialization point in
//! the decode timeline (gputrace receipt on the fused-attention-pre-post
//! ticket). `fused_qkv_prep` replaces them with two dispatches: q-side
//! head-norm + rope written in attention layout, and k/v-side head-norm +
//! rope + raw copy written DIRECTLY into the KV-cache slots.
//!
//! Bit-preserving: the kernels mirror rmsnorm/rope_partial semantics exactly
//! (fork test attn_prep_matches_unfused_chain enforces byte identity), and
//! the copies are byte-exact, so the composed layer output is bitwise equal
//! to the unfused path.

use anyhow::{Context, Result};
use candle::op::BackpropOp;
use candle::{DType, Device, Storage, Tensor};
use candle_metal_kernels::kernels::{
    call_attn_kv_prep, call_attn_q_prep, ATTN_PREP_BLOCK, ATTN_PREP_MAX_D,
};

/// Kernel block contract (see attn_prep.metal): one 128-thread threadgroup
/// per row, static threadgroup storage for up to 256 lanes.
pub fn supports_head_dim(head_dim: usize) -> bool {
    head_dim <= ATTN_PREP_MAX_D && (head_dim / 2).next_power_of_two() == ATTN_PREP_BLOCK
}

fn metal_buffer(t: &Tensor, name: &str) -> Result<candle_metal_kernels::metal::Buffer> {
    let (storage, layout) = t.storage_and_layout();
    if !layout.is_contiguous() || layout.start_offset() != 0 {
        anyhow::bail!("fused attn prep: {name} must be contiguous at offset 0");
    }
    match &*storage {
        Storage::Metal(ms) => Ok(ms.buffer().clone()),
        _ => anyhow::bail!("fused attn prep: {name} not on Metal"),
    }
}

/// Fused q/k head-norm + partial rope + KV-cache append for one attention
/// layer. `qkv` is the packed `(1, l, 2*heads*d + 2*kv_heads*d)` projection
/// (per-head `[q | gate]` in the q block); `cache_k`/`cache_v` are the
/// `(1, kv_heads, capacity, d)` cache buffers, written at rows
/// `[write_pos, write_pos + l)`. `pos_base` is the rope position of the
/// first token. Returns q as `(1, heads, l, d)`.
#[allow(clippy::too_many_arguments)]
pub fn fused_qkv_prep(
    qkv: &Tensor,
    q_norm_w: &Tensor,
    k_norm_w: &Tensor,
    eps: f32,
    cos: &Tensor,
    sin: &Tensor,
    pos_base: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    cache_k: &Tensor,
    cache_v: &Tensor,
    write_pos: usize,
) -> Result<Tensor> {
    let Device::Metal(device) = qkv.device() else {
        anyhow::bail!("fused attn prep requires a Metal device");
    };
    let (b, l, width) = qkv.dims3().context("fused attn prep: qkv must be rank-3")?;
    let q_out = heads * head_dim * 2;
    let kv_out = kv_heads * head_dim;
    if b != 1 || width != q_out + 2 * kv_out {
        anyhow::bail!(
            "fused attn prep: qkv (b={b}, w={width}) does not match heads={heads} kv={kv_heads} d={head_dim}"
        );
    }
    let (max_pos, half_rd) = cos.dims2().context("fused attn prep: cos must be rank-2")?;
    let rd = 2 * half_rd;
    if sin.dims2()? != (max_pos, half_rd) {
        anyhow::bail!("fused attn prep: sin shape {:?} != cos {:?}", sin.shape(), cos.shape());
    }
    if pos_base + l > max_pos {
        anyhow::bail!("fused attn prep: positions {pos_base}+{l} exceed rope table {max_pos}");
    }
    let (cb, ch, cap, cd) = cache_k.dims4().context("fused attn prep: cache_k must be rank-4")?;
    if (cb, ch, cd) != (1, kv_heads, head_dim) || cache_v.dims4()? != (cb, ch, cap, cd) {
        anyhow::bail!(
            "fused attn prep: cache shapes k {:?} / v {:?} do not match (1, {kv_heads}, cap, {head_dim})",
            cache_k.shape(),
            cache_v.shape()
        );
    }
    if write_pos + l > cap {
        anyhow::bail!("fused attn prep: write {write_pos}+{l} exceeds cache capacity {cap}");
    }
    if (head_dim / 2).next_power_of_two() != ATTN_PREP_BLOCK {
        anyhow::bail!("fused attn prep: head_dim {head_dim} outside the kernel's block contract");
    }
    for (t, name) in [
        (qkv, "qkv"),
        (q_norm_w, "q_norm weight"),
        (k_norm_w, "k_norm weight"),
        (cos, "cos"),
        (sin, "sin"),
        (cache_k, "cache_k"),
        (cache_v, "cache_v"),
    ] {
        if t.dtype() != DType::BF16 {
            anyhow::bail!("fused attn prep: {name} must be BF16, got {:?}", t.dtype());
        }
    }

    let qkv_buf = metal_buffer(qkv, "qkv")?;
    let q_w_buf = metal_buffer(q_norm_w, "q_norm weight")?;
    let k_w_buf = metal_buffer(k_norm_w, "k_norm weight")?;
    let cos_buf = metal_buffer(cos, "cos")?;
    let sin_buf = metal_buffer(sin, "sin")?;
    let cache_k_buf = metal_buffer(cache_k, "cache_k")?;
    let cache_v_buf = metal_buffer(cache_v, "cache_v")?;

    let q_elems = heads * l * head_dim;
    let q_buf = device
        .new_buffer_builder()
        .with_size_for(q_elems, DType::BF16)
        .with_label("attn_q_prep")
        .build()?;

    let encoder = device.command_encoder()?;
    call_attn_q_prep(
        device.metal_device(),
        &encoder,
        device.kernels(),
        "attn_q_prep_bf16",
        heads,
        l,
        head_dim,
        rd,
        width,
        2 * head_dim,
        0,
        pos_base,
        eps,
        &qkv_buf,
        0,
        &q_w_buf,
        &cos_buf,
        &sin_buf,
        &q_buf,
    )
    .map_err(candle::Error::wrap)
    .context("attn_q_prep dispatch")?;
    call_attn_kv_prep(
        device.metal_device(),
        &encoder,
        device.kernels(),
        "attn_kv_prep_bf16",
        kv_heads,
        l,
        head_dim,
        rd,
        width,
        q_out,
        q_out + kv_out,
        cap,
        write_pos,
        pos_base,
        eps,
        &qkv_buf,
        0,
        &k_w_buf,
        &cos_buf,
        &sin_buf,
        &cache_k_buf,
        &cache_v_buf,
    )
    .map_err(candle::Error::wrap)
    .context("attn_kv_prep dispatch")?;
    drop(encoder);

    let storage = candle::MetalStorage::new(q_buf, device.clone(), q_elems, DType::BF16);
    Ok(Tensor::from_storage(
        Storage::Metal(storage),
        (1, heads, l, head_dim),
        BackpropOp::none(),
        false,
    ))
}
