//! Host-side wrapper for the fork's fused GatedDeltaNet decode kernel.
//!
//! One dispatch covers conv + gates + l2norm + delta rule + group norm +
//! z-gating for a single token (see gated_delta.metal in the candle fork).
//! Outputs are fresh tensors — states are never mutated in place, so the
//! runner's replace-by-assignment snapshot semantics hold verbatim.

use anyhow::{Context, Result};
use candle::op::BackpropOp;
use candle::{DType, Device, Storage, Tensor};
use candle_metal_kernels::kernels::{call_gated_delta_decode, GatedDeltaParams};

pub struct GatedDeltaDims {
    pub heads: usize,
    pub dk: usize,
    pub dv: usize,
    pub conv_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub ksz: usize,
}

/// Runs the fused decode step. `proj` is the packed projection
/// [qkv | z | b | a] of one token; states/weights per the kernel contract.
/// Returns (gated_output [1, 1, value_dim], conv_state, recurrent_state).
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_decode(
    proj: &Tensor,
    conv_state: &Tensor,
    recurrent_state: &Tensor,
    conv_weight: &Tensor,
    dt_bias_f32: &Tensor,
    a_log_exp_f32: &Tensor,
    norm_weight_f32: &Tensor,
    dims: &GatedDeltaDims,
    l2_eps: f32,
    norm_eps: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let Device::Metal(device) = proj.device() else {
        anyhow::bail!("fused gated-delta decode requires a Metal device");
    };

    // The kernel wants offset-0 contiguous buffers. Hot-path tensors already
    // are; anything narrowed/offset (e.g. the post-prefill conv window)
    // materializes once here.
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
            // Contiguous but offset: force a fresh offset-0 buffer.
            t.affine(1.0, 0.0)?
        })
    }
    let proj = offset0(proj)?;
    let conv_state = offset0(conv_state)?;
    let recurrent_state = offset0(recurrent_state)?;
    let conv_weight = offset0(conv_weight)?;
    let dt_bias_f32 = offset0(dt_bias_f32)?;
    let a_log_exp_f32 = offset0(a_log_exp_f32)?;
    let norm_weight_f32 = offset0(norm_weight_f32)?;
    let (proj, conv_state, recurrent_state, conv_weight, dt_bias_f32, a_log_exp_f32, norm_weight_f32) = (
        &proj,
        &conv_state,
        &recurrent_state,
        &conv_weight,
        &dt_bias_f32,
        &a_log_exp_f32,
        &norm_weight_f32,
    );

    // Contiguity/offset/dtype checks; guards stay alive across the encode.
    let inputs = [
        (proj, DType::BF16, "proj"),
        (conv_state, DType::BF16, "conv_state"),
        (recurrent_state, DType::F32, "recurrent_state"),
        (conv_weight, DType::BF16, "conv_weight"),
        (dt_bias_f32, DType::F32, "dt_bias"),
        (a_log_exp_f32, DType::F32, "a_log_exp"),
        (norm_weight_f32, DType::F32, "norm_weight"),
    ];
    let mut guards = Vec::with_capacity(inputs.len());
    for (t, dtype, name) in &inputs {
        if t.dtype() != *dtype {
            anyhow::bail!("fused gated-delta: {name} must be {dtype:?}, got {:?}", t.dtype());
        }
        let (storage, layout) = t.storage_and_layout();
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            anyhow::bail!("fused gated-delta: {name} must be contiguous at offset 0");
        }
        guards.push(storage);
    }
    let buffer = |i: usize| -> Result<_> {
        match &*guards[i] {
            Storage::Metal(ms) => Ok(ms.buffer().clone()),
            _ => anyhow::bail!("fused gated-delta: {} not on Metal", inputs[i].2),
        }
    };
    let proj_buf = buffer(0)?;
    let conv_in_buf = buffer(1)?;
    let state_in_buf = buffer(2)?;
    let conv_w_buf = buffer(3)?;
    let dt_bias_buf = buffer(4)?;
    let a_log_exp_buf = buffer(5)?;
    let norm_w_buf = buffer(6)?;

    let out_buf = device
        .new_buffer_builder()
        .with_size_for(dims.value_dim, DType::BF16)
        .with_label("gated_delta_out")
        .build()?;
    let conv_out_buf = device
        .new_buffer_builder()
        .with_size_for(dims.conv_dim * dims.ksz, DType::BF16)
        .with_label("gated_delta_conv_state")
        .build()?;
    let state_out_buf = device
        .new_buffer_builder()
        .with_size_for(dims.heads * dims.dk * dims.dv, DType::F32)
        .with_label("gated_delta_state")
        .build()?;

    let encoder = device.command_encoder()?;
    call_gated_delta_decode(
        device.metal_device(),
        &encoder,
        device.kernels(),
        GatedDeltaParams {
            heads: dims.heads as u32,
            dk: dims.dk as u32,
            dv: dims.dv as u32,
            conv_dim: dims.conv_dim as u32,
            key_dim: dims.key_dim as u32,
            value_dim: dims.value_dim as u32,
            ksz: dims.ksz as u32,
            l2_eps,
            norm_eps,
        },
        &proj_buf,
        &conv_in_buf,
        &state_in_buf,
        &conv_w_buf,
        &dt_bias_buf,
        &a_log_exp_buf,
        &norm_w_buf,
        &out_buf,
        &conv_out_buf,
        &state_out_buf,
    )
    .map_err(candle::Error::wrap)
    .context("fused gated-delta dispatch")?;
    drop(encoder);
    drop(guards);

    let mk = |buf, count: usize, dtype, shape: Vec<usize>| -> Tensor {
        let storage = candle::MetalStorage::new(buf, device.clone(), count, dtype);
        Tensor::from_storage(Storage::Metal(storage), shape, BackpropOp::none(), false)
    };
    let out = mk(out_buf, dims.value_dim, DType::BF16, vec![1, 1, dims.value_dim]);
    let conv_out = mk(
        conv_out_buf,
        dims.conv_dim * dims.ksz,
        DType::BF16,
        vec![1, dims.conv_dim, dims.ksz],
    );
    let state_out = mk(
        state_out_buf,
        dims.heads * dims.dk * dims.dv,
        DType::F32,
        vec![1, dims.heads, dims.dk, dims.dv],
    );
    Ok((out, conv_out, state_out))
}
