//! Host-side wrapper for the fork's fused `x * sigmoid(gate)` binary kernel
//! (the full-attention output gate: previously a sigmoid dispatch + a bmul
//! dispatch + their barrier, x6 attention layers per token).
//!
//! BIT-IDENTICAL to the two-op sequence: the kernel instantiates the same
//! sigmoid template the unary kernel uses (rounded to the storage dtype
//! exactly like its store) and multiplies with bmul's expression. Non-Metal
//! devices fall back to the two-op form (CPU parity tests).

use anyhow::Result;
use candle::{DType, Device, Storage, Tensor};
use candle_metal_kernels::kernels::binary::call_binary_contiguous;
use candle_metal_kernels::BufferOffset;

/// `x * sigmoid(gate)`, elementwise; shapes must match.
pub fn mul_sigmoid(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
    anyhow::ensure!(
        x.shape() == gate.shape(),
        "mul_sigmoid: x {:?} != gate {:?}",
        x.shape(),
        gate.shape()
    );
    let Device::Metal(device) = x.device() else {
        // Reference path (CPU tests): the exact two-op sequence.
        return Ok((x * candle_nn::ops::sigmoid(gate)?)?);
    };
    let kernel = match x.dtype() {
        DType::BF16 => "bmulsig_bf16",
        DType::F32 => "bmulsig_f32",
        DType::F16 => "bmulsig_f16",
        other => anyhow::bail!("mul_sigmoid: unsupported dtype {other:?}"),
    };
    let x = if x.is_contiguous() { x.clone() } else { x.contiguous()? };
    let gate = if gate.is_contiguous() {
        gate.clone()
    } else {
        gate.contiguous()?
    };
    let elem_count = x.elem_count();
    let out = device
        .new_buffer_builder()
        .with_size_for(elem_count, x.dtype())
        .with_label("mul_sigmoid")
        .build()?;
    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (g_storage, g_layout) = gate.storage_and_layout();
        let Storage::Metal(x_ms) = &*x_storage else {
            anyhow::bail!("mul_sigmoid: x not Metal-resident");
        };
        let Storage::Metal(g_ms) = &*g_storage else {
            anyhow::bail!("mul_sigmoid: gate not Metal-resident");
        };
        let encoder = device.command_encoder()?;
        call_binary_contiguous(
            device.metal_device(),
            &encoder,
            device.kernels(),
            kernel,
            x.dtype().size_in_bytes(),
            elem_count,
            BufferOffset {
                buffer: x_ms.buffer(),
                offset_in_bytes: x_layout.start_offset() * x.dtype().size_in_bytes(),
            },
            BufferOffset {
                buffer: g_ms.buffer(),
                offset_in_bytes: g_layout.start_offset() * x.dtype().size_in_bytes(),
            },
            &out,
        )
        .map_err(candle::Error::wrap)?;
    }
    let storage = candle::MetalStorage::new(out, device.clone(), elem_count, x.dtype());
    Ok(Tensor::from_storage(
        Storage::Metal(storage),
        x.shape().clone(),
        candle::op::BackpropOp::none(),
        false,
    ))
}
