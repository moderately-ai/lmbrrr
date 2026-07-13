//! Host-side wrapper for the fork's fused residual-add + RMSNorm kernel.
//!
//! Collapses the `xs = residual + hidden; normed = rms_norm(xs)` pair (a
//! `badd` dispatch + a `rmsnorm` dispatch, plus the auto-barrier between
//! them) into one dispatch that emits BOTH `xs` (the next block's residual)
//! and `normed`. Two independent offset-0 output buffers — see the module
//! rationale in the ticket: the residual outlives the normed output, so they
//! must not share a buffer.
//!
//! Not bit-preserving: `xs` is bit-identical to a BF16 `badd`
//! (`bf16(f32(a)+f32(b))`), but `normed` consumes the F32 sum directly, so it
//! differs from add-then-norm by the BF16 rounding of the intermediate
//! (sub-noise, margin-gated per the campaign protocol).

use anyhow::{Context, Result};
use candle::op::BackpropOp;
use candle::{DType, Device, Storage, Tensor};
use candle_metal_kernels::kernels::call_rms_norm_add;

/// The kernels want offset-0 contiguous buffers; anything narrowed/offset
/// materializes once here (mirrors `fused_deltanet::offset0`).
fn offset0(t: &Tensor) -> Result<Tensor> {
    let needs_fix = {
        let (_guard, layout) = t.storage_and_layout();
        !layout.is_contiguous() || layout.start_offset() != 0
    };
    Ok(if !needs_fix {
        t.clone()
    } else if !t.is_contiguous() {
        t.contiguous()?
    } else {
        t.affine(1.0, 0.0)?
    })
}

/// Fused `(a + b, rms_norm(a + b) * weight)` over the last dim. `a`, `b`,
/// `weight` must be BF16 on Metal; `weight` is `[dim]`. Returns
/// `(sum, normed)`, both the shape of `a`.
pub fn fused_rms_norm_add(
    a: &Tensor,
    b: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let Device::Metal(device) = a.device() else {
        anyhow::bail!("fused rms_norm_add requires a Metal device");
    };
    if a.shape() != b.shape() {
        anyhow::bail!("fused rms_norm_add: a {:?} != b {:?}", a.shape(), b.shape());
    }
    let dim = weight.dims1().context("fused rms_norm_add: weight must be rank-1")?;
    if a.dim(candle::D::Minus1)? != dim {
        anyhow::bail!(
            "fused rms_norm_add: last dim {} != weight dim {dim}",
            a.dim(candle::D::Minus1)?
        );
    }
    let elem_count = a.elem_count();
    if !elem_count.is_multiple_of(dim) {
        anyhow::bail!("fused rms_norm_add: elem_count {elem_count} not a multiple of dim {dim}");
    }
    let rows = elem_count / dim;

    let sanitized = [offset0(a)?, offset0(b)?, offset0(weight)?];
    for (t, name) in sanitized.iter().zip(["a", "b", "weight"]) {
        if t.dtype() != DType::BF16 {
            anyhow::bail!("fused rms_norm_add: {name} must be BF16, got {:?}", t.dtype());
        }
    }
    let mut guards = Vec::with_capacity(3);
    for t in &sanitized {
        let (storage, _) = t.storage_and_layout();
        guards.push(storage);
    }
    let buffer = |i: usize| -> Result<_> {
        match &*guards[i] {
            Storage::Metal(ms) => Ok(ms.buffer().clone()),
            _ => anyhow::bail!("fused rms_norm_add: input {i} not on Metal"),
        }
    };
    let a_buf = buffer(0)?;
    let b_buf = buffer(1)?;
    let w_buf = buffer(2)?;

    let sum_buf = device
        .new_buffer_builder()
        .with_size_for(elem_count, DType::BF16)
        .with_label("rms_norm_add_sum")
        .build()?;
    let norm_buf = device
        .new_buffer_builder()
        .with_size_for(elem_count, DType::BF16)
        .with_label("rms_norm_add_norm")
        .build()?;

    let encoder = device.command_encoder()?;
    call_rms_norm_add(
        device.metal_device(),
        &encoder,
        device.kernels(),
        "rmsnorm_add_bf16",
        rows,
        dim,
        eps,
        &a_buf,
        0,
        &b_buf,
        0,
        &w_buf,
        0,
        &sum_buf,
        &norm_buf,
    )
    .map_err(candle::Error::wrap)
    .context("fused rms_norm_add dispatch")?;
    drop(encoder);
    drop(guards);

    let shape = a.shape().clone();
    let mk = |buf| -> Tensor {
        let storage = candle::MetalStorage::new(buf, device.clone(), elem_count, DType::BF16);
        Tensor::from_storage(Storage::Metal(storage), shape.clone(), BackpropOp::none(), false)
    };
    Ok((mk(sum_buf), mk(norm_buf)))
}
