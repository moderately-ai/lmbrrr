//! Host-side wrapper for the fork's fused MTP pre-fc kernel.
//!
//! Collapses the MTP chain-step prologue — `rmsnorm(embeds)`, `rmsnorm(hidden)`,
//! `cat`, dense fc GEMV — from 4 serially-dependent dispatches to 1. The MTP
//! chain step is the drafter's latency-critical path (depth dispatches per
//! round, each a serialization point on the GPU timeline).
//!
//! Bit-preserving at m == 1: the kernel replicates rms_norm's 512-wide
//! reduction pairing and the mlx gemv tile's accumulation order exactly
//! (enforced by the fork's `mtp_fc_prep_matches_unfused_chain` test). At
//! m > 1 the unfused fc takes the simdgroup-matrix GEMM whose accumulation
//! order a GEMV shape cannot reproduce, so callers route m == 1 only.

use anyhow::{Context, Result};
use candle::op::BackpropOp;
use candle::{DType, Device, Storage, Tensor};
use candle_metal_kernels::{call_mtp_fc_prep, MTP_FC_LEAVES, MTP_FC_MAX_H};

/// Shapes the kernel serves: rms_norm must pick BLOCKSIZE 512 for the
/// emulated reduction to pair identically, and K = 2*h must fill whole
/// 128-element gemv iterations.
pub fn eligible_h_dim(h: usize) -> bool {
    h <= MTP_FC_MAX_H && (h / 2).next_power_of_two() == MTP_FC_LEAVES && h.is_multiple_of(64)
}

/// Fused `fc_w x cat(rmsnorm(embeds) * alpha_e, rmsnorm(hidden) * alpha_h)`
/// over the last dim. All tensors BF16 on Metal; `embeds`/`hidden` share a
/// shape whose last dim is `h_dim`; `fc_w` is `[out_dim, 2*h_dim]` row-major.
/// Returns the fc output with the last dim replaced by `out_dim`.
pub fn fused_mtp_fc_prep(
    embeds: &Tensor,
    hidden: &Tensor,
    alpha_e: &Tensor,
    alpha_h: &Tensor,
    fc_w: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let Device::Metal(device) = embeds.device() else {
        anyhow::bail!("fused mtp_fc_prep requires a Metal device");
    };
    if embeds.shape() != hidden.shape() {
        anyhow::bail!(
            "fused mtp_fc_prep: embeds {:?} != hidden {:?}",
            embeds.shape(),
            hidden.shape()
        );
    }
    let h_dim = embeds.dim(candle::D::Minus1)?;
    if !eligible_h_dim(h_dim) {
        anyhow::bail!("fused mtp_fc_prep: ineligible h_dim {h_dim}");
    }
    let (out_dim, k_dim) = fc_w.dims2().context("fused mtp_fc_prep: fc_w must be rank-2")?;
    if k_dim != 2 * h_dim || out_dim == 0 || !out_dim.is_multiple_of(16) {
        anyhow::bail!("fused mtp_fc_prep: fc_w [{out_dim}, {k_dim}] incompatible with h_dim {h_dim}");
    }
    for (t, dims, name) in [
        (alpha_e, h_dim, "alpha_e"),
        (alpha_h, h_dim, "alpha_h"),
    ] {
        if t.dims1()? != dims {
            anyhow::bail!("fused mtp_fc_prep: {name} must be [{dims}]");
        }
    }
    let m = embeds.elem_count() / h_dim;

    let inputs = [embeds, hidden, alpha_e, alpha_h, fc_w];
    let names = ["embeds", "hidden", "alpha_e", "alpha_h", "fc_w"];
    let mut guards = Vec::with_capacity(inputs.len());
    let mut offsets = [0usize; 5];
    for (i, (t, name)) in inputs.iter().zip(names).enumerate() {
        if t.dtype() != DType::BF16 {
            anyhow::bail!("fused mtp_fc_prep: {name} must be BF16, got {:?}", t.dtype());
        }
        let (storage, layout) = t.storage_and_layout();
        if !layout.is_contiguous() {
            anyhow::bail!("fused mtp_fc_prep: {name} must be contiguous");
        }
        offsets[i] = layout.start_offset() * DType::BF16.size_in_bytes();
        guards.push(storage);
    }
    let buffer = |i: usize| -> Result<_> {
        match &*guards[i] {
            Storage::Metal(ms) => Ok(ms.buffer().clone()),
            _ => anyhow::bail!("fused mtp_fc_prep: input {i} not on Metal"),
        }
    };
    let embeds_buf = buffer(0)?;
    let hidden_buf = buffer(1)?;
    let alpha_e_buf = buffer(2)?;
    let alpha_h_buf = buffer(3)?;
    let fc_w_buf = buffer(4)?;

    let out_buf = device
        .new_buffer_builder()
        .with_size_for(m * out_dim, DType::BF16)
        .with_label("mtp_fc_prep_out")
        .build()?;

    let encoder = device.command_encoder()?;
    call_mtp_fc_prep(
        device.metal_device(),
        &encoder,
        device.kernels(),
        "mtp_fc_prep_bf16",
        m,
        h_dim,
        out_dim,
        eps,
        &embeds_buf,
        offsets[0],
        &hidden_buf,
        offsets[1],
        &alpha_e_buf,
        offsets[2],
        &alpha_h_buf,
        offsets[3],
        &fc_w_buf,
        offsets[4],
        &out_buf,
    )
    .map_err(candle::Error::wrap)
    .context("fused mtp_fc_prep dispatch")?;
    drop(encoder);
    drop(guards);

    let mut dims = embeds.dims().to_vec();
    *dims.last_mut().expect("rank >= 1") = out_dim;
    let storage = candle::MetalStorage::new(out_buf, device.clone(), m * out_dim, DType::BF16);
    Ok(Tensor::from_storage(
        Storage::Metal(storage),
        dims,
        BackpropOp::none(),
        false,
    ))
}
