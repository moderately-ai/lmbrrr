//! Host-side wrapper for the fork's fused head GEMV -> argmax kernels.
//!
//! Greedy decode needs only argmax(logits); the stored-logits path runs the
//! 248094-row head GEMV (a ~0.5 MB bf16 write), a barrier, then a separate
//! argmax dispatch reading it all back. The fused pair keeps per-row sums in
//! registers and reduces straight to the winning row index — exact by
//! construction: per-row arithmetic is byte-for-byte the production nr2sg2
//! kernel, the comparison happens on the bf16-rounded values candle's
//! fast_argmax would see, and ties resolve to the lowest index (candle's
//! rule). Gated by the fork bench task `argmax-head` (24 varied activations
//! + manufactured exact ties).

use anyhow::{Context, Result};
use candle::op::BackpropOp;
use candle::quantized::{GgmlDType, QStorage, QTensor};
use candle::{DType, Device, Storage, Tensor};
use candle_metal_kernels::{call_quantized_matmul_mv_q4k_argmax, MV_ARGMAX_ROWS_PER_TG};

/// One greedy head evaluation: `hidden` is the last position's BF16 hidden
/// state ([1, 1, k] or any contiguous k-element shape), `head` the q4_K
/// lm_head. Returns the argmax row id as a rank-1 [1] U32 tensor on device
/// (same shape the stored-logits path's `logits.argmax(D::Minus1)` yields).
pub fn q4k_head_argmax(hidden: &Tensor, head: &QTensor) -> Result<Tensor> {
    let Device::Metal(device) = hidden.device() else {
        anyhow::bail!("fused head argmax requires a Metal device");
    };
    anyhow::ensure!(
        head.dtype() == GgmlDType::Q4K,
        "fused head argmax requires a q4_K head (got {:?})",
        head.dtype()
    );
    anyhow::ensure!(
        hidden.dtype() == DType::BF16,
        "fused head argmax requires BF16 hidden (got {:?})",
        hidden.dtype()
    );
    let dims = head.shape().dims();
    anyhow::ensure!(dims.len() == 2, "head must be [n, k]; got {dims:?}");
    let (n, k) = (dims[0], dims[1]);
    anyhow::ensure!(
        hidden.elem_count() == k,
        "hidden holds {} elements, head wants k={k}",
        hidden.elem_count()
    );

    let hidden = if hidden.is_contiguous() {
        hidden.clone()
    } else {
        hidden.contiguous()?
    };
    let (hidden_storage, hidden_layout) = hidden.storage_and_layout();
    let Storage::Metal(hidden_ms) = &*hidden_storage else {
        anyhow::bail!("hidden is not Metal-resident");
    };
    let hidden_offset = hidden_layout.start_offset() * 2;
    let QStorage::Metal(head_ms) = head.storage() else {
        anyhow::bail!("head qtensor is not Metal-resident");
    };

    let ntg = n.div_ceil(MV_ARGMAX_ROWS_PER_TG);
    let partials = device
        .new_buffer_builder()
        .with_size(ntg * 8)
        .with_label("head_argmax_partials")
        .build()?;
    let out = device
        .new_buffer_builder()
        .with_size(4)
        .with_label("head_argmax_out")
        .build()?;

    {
        let encoder = device.command_encoder()?;
        call_quantized_matmul_mv_q4k_argmax(
            device.metal_device(),
            &encoder,
            device.kernels(),
            (n, k),
            (hidden_ms.buffer(), hidden_offset),
            (head_ms.buffer(), 0),
            &partials,
            &out,
        )
        .map_err(candle::Error::wrap)
        .context("fused head argmax dispatch")?;
    }
    drop(hidden_storage);

    let storage = candle::MetalStorage::new(out, device.clone(), 1, DType::U32);
    Ok(Tensor::from_storage(
        Storage::Metal(storage),
        (1,),
        BackpropOp::none(),
        false,
    ))
}
