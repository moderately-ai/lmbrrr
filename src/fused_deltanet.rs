//! Host-side wrapper for the fork's fused GatedDeltaNet decode kernel.
//!
//! One dispatch covers conv + gates + l2norm + delta rule + group norm +
//! z-gating for a single token (see gated_delta.metal in the candle fork).
//! Outputs are fresh tensors — states are never mutated in place, so the
//! runner's replace-by-assignment snapshot semantics hold verbatim.

use anyhow::{Context, Result};
use candle::op::BackpropOp;
use candle::{DType, Device, Storage, Tensor};
use candle_metal_kernels::kernels::{
    call_gated_delta_chunk, call_gated_delta_decode, GatedDeltaParams,
};

/// Rollback-capture tensors emitted by the fused chunk kernel, matching the
/// tensor-path intermediates consumed by closed-form state selection.
pub struct FusedChunkCapture {
    /// F32 (1, heads, l, dk) l2-normed keys.
    pub kc: Tensor,
    /// F32 (1, heads, l, dv) WY pseudo-values.
    pub delta: Tensor,
    /// F32 (1, heads, l) inclusive log-decay cumsum.
    pub gcs: Tensor,
}

/// The kernels want offset-0 contiguous buffers. Hot-path tensors already
/// are; anything narrowed/offset (e.g. the post-prefill conv window)
/// materializes once here.
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

pub struct GatedDeltaDims {
    pub heads: usize,
    pub dk: usize,
    pub dv: usize,
    pub conv_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub ksz: usize,
}

/// Runs the fused chunk step (2 <= l <= 12). `proj` is the packed
/// per-position projection [l, qkv | z | b | a]. Returns the gated output
/// [1, l, value_dim], updated states, and the rollback capture.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_chunk(
    proj: &Tensor,
    seq_len: usize,
    conv_state: &Tensor,
    recurrent_state: &Tensor,
    conv_weight: &Tensor,
    dt_bias_f32: &Tensor,
    a_log_exp_f32: &Tensor,
    norm_weight_f32: &Tensor,
    dims: &GatedDeltaDims,
    l2_eps: f32,
    norm_eps: f32,
) -> Result<(Tensor, Tensor, Tensor, FusedChunkCapture)> {
    let Device::Metal(device) = proj.device() else {
        anyhow::bail!("fused gated-delta chunk requires a Metal device");
    };
    let sanitized = [
        offset0(proj)?,
        offset0(conv_state)?,
        offset0(recurrent_state)?,
        offset0(conv_weight)?,
        offset0(dt_bias_f32)?,
        offset0(a_log_exp_f32)?,
        offset0(norm_weight_f32)?,
    ];
    let dtypes = [
        DType::BF16,
        DType::BF16,
        DType::F32,
        DType::BF16,
        DType::F32,
        DType::F32,
        DType::F32,
    ];
    let mut guards = Vec::with_capacity(sanitized.len());
    for (t, dtype) in sanitized.iter().zip(dtypes.iter()) {
        if t.dtype() != *dtype {
            anyhow::bail!("fused gated-delta chunk: dtype mismatch {:?} vs {dtype:?}", t.dtype());
        }
        let (storage, _) = t.storage_and_layout();
        guards.push(storage);
    }
    let buffer = |i: usize| -> Result<_> {
        match &*guards[i] {
            Storage::Metal(ms) => Ok(ms.buffer().clone()),
            _ => anyhow::bail!("fused gated-delta chunk: input {i} not on Metal"),
        }
    };
    let bufs: Vec<_> = (0..sanitized.len()).map(&buffer).collect::<Result<_>>()?;

    let alloc = |count: usize, dtype: DType, label: &str| {
        device
            .new_buffer_builder()
            .with_size_for(count, dtype)
            .with_label(label)
            .build()
    };
    let l = seq_len;
    let out_buf = alloc(l * dims.value_dim, DType::BF16, "gdc_out")?;
    let conv_out_buf = alloc(dims.conv_dim * dims.ksz, DType::BF16, "gdc_conv")?;
    let state_out_buf = alloc(dims.heads * dims.dk * dims.dv, DType::F32, "gdc_state")?;
    let cap_k_buf = alloc(dims.heads * l * dims.dk, DType::F32, "gdc_cap_k")?;
    let cap_delta_buf = alloc(dims.heads * l * dims.dv, DType::F32, "gdc_cap_delta")?;
    let cap_gcs_buf = alloc(dims.heads * l, DType::F32, "gdc_cap_gcs")?;

    let encoder = device.command_encoder()?;
    call_gated_delta_chunk(
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
        l,
        &bufs[0],
        &bufs[1],
        &bufs[2],
        &bufs[3],
        &bufs[4],
        &bufs[5],
        &bufs[6],
        &out_buf,
        &conv_out_buf,
        &state_out_buf,
        &cap_k_buf,
        &cap_delta_buf,
        &cap_gcs_buf,
    )
    .map_err(candle::Error::wrap)
    .context("fused gated-delta chunk dispatch")?;
    drop(encoder);
    drop(guards);

    let mk = |buf, count: usize, dtype, shape: Vec<usize>| -> Tensor {
        let storage = candle::MetalStorage::new(buf, device.clone(), count, dtype);
        Tensor::from_storage(Storage::Metal(storage), shape, BackpropOp::none(), false)
    };
    let out = mk(out_buf, l * dims.value_dim, DType::BF16, vec![1, l, dims.value_dim]);
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
    let capture = FusedChunkCapture {
        kc: mk(cap_k_buf, dims.heads * l * dims.dk, DType::F32, vec![1, dims.heads, l, dims.dk]),
        delta: mk(
            cap_delta_buf,
            dims.heads * l * dims.dv,
            DType::F32,
            vec![1, dims.heads, l, dims.dv],
        ),
        gcs: mk(cap_gcs_buf, dims.heads * l, DType::F32, vec![1, dims.heads, l]),
    };
    Ok((out, conv_out, state_out, capture))
}

/// Runs the fused decode step. `proj` is the packed projection
/// [qkv | z | b | a] of one token; states/weights per the kernel contract.
/// Returns (gated_output [1, 1, value_dim], conv_state, recurrent_state).
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_decode(
    proj: &Tensor,
    batch: usize,
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
        .with_size_for(batch * dims.value_dim, DType::BF16)
        .with_label("gated_delta_out")
        .build()?;
    let conv_out_buf = device
        .new_buffer_builder()
        .with_size_for(batch * dims.conv_dim * dims.ksz, DType::BF16)
        .with_label("gated_delta_conv_state")
        .build()?;
    let state_out_buf = device
        .new_buffer_builder()
        .with_size_for(batch * dims.heads * dims.dk * dims.dv, DType::F32)
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
        batch,
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
    let out = mk(
        out_buf,
        batch * dims.value_dim,
        DType::BF16,
        vec![batch, 1, dims.value_dim],
    );
    let conv_out = mk(
        conv_out_buf,
        batch * dims.conv_dim * dims.ksz,
        DType::BF16,
        vec![batch, dims.conv_dim, dims.ksz],
    );
    let state_out = mk(
        state_out_buf,
        batch * dims.heads * dims.dk * dims.dv,
        DType::F32,
        vec![batch, dims.heads, dims.dk, dims.dv],
    );
    Ok((out, conv_out, state_out))
}

/// Runs the v2 fused path (unified decode/chunk, 1 <= l <= 12; three
/// dispatches: prep, delta core, epilogue). The recurrent state is TRANSPOSED
/// relative to v1: f32 (b, heads, dv, dk) — same shape at dk == dv == 128,
/// different semantic layout; callers track which layout a tensor holds.
/// Returns (out [b, l, value_dim], conv_out, state_out (transposed), capture).
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_v2(
    proj: &Tensor,
    seq_len: usize,
    batch: usize,
    conv_state: &Tensor,
    recurrent_state_t: &Tensor,
    conv_weight: &Tensor,
    dt_bias_f32: &Tensor,
    a_log_exp_f32: &Tensor,
    norm_weight_f32: &Tensor,
    dims: &GatedDeltaDims,
    l2_eps: f32,
    norm_eps: f32,
) -> Result<(Tensor, Tensor, Tensor, FusedChunkCapture)> {
    use candle_metal_kernels::kernels::{call_gated_delta_v2, GatedDeltaV2Stages};
    let Device::Metal(device) = proj.device() else {
        anyhow::bail!("fused gated-delta v2 requires a Metal device");
    };
    let sanitized = [
        offset0(proj)?,
        offset0(conv_state)?,
        offset0(recurrent_state_t)?,
        offset0(conv_weight)?,
        offset0(dt_bias_f32)?,
        offset0(a_log_exp_f32)?,
        offset0(norm_weight_f32)?,
    ];
    let dtypes = [
        DType::BF16,
        DType::BF16,
        DType::F32,
        DType::BF16,
        DType::F32,
        DType::F32,
        DType::F32,
    ];
    let mut guards = Vec::with_capacity(sanitized.len());
    for (t, dtype) in sanitized.iter().zip(dtypes.iter()) {
        if t.dtype() != *dtype {
            anyhow::bail!("fused gated-delta v2: dtype mismatch {:?} vs {dtype:?}", t.dtype());
        }
        let (storage, _) = t.storage_and_layout();
        guards.push(storage);
    }
    let buffer = |i: usize| -> Result<_> {
        match &*guards[i] {
            Storage::Metal(ms) => Ok(ms.buffer().clone()),
            _ => anyhow::bail!("fused gated-delta v2: input {i} not on Metal"),
        }
    };
    let bufs: Vec<_> = (0..sanitized.len()).map(&buffer).collect::<Result<_>>()?;

    let alloc = |count: usize, dtype: DType, label: &str| {
        device
            .new_buffer_builder()
            .with_size_for(count, dtype)
            .with_label(label)
            .build()
    };
    let l = seq_len;
    let b = batch.max(1);
    let bh = b * dims.heads;
    let out_buf = alloc(b * l * dims.value_dim, DType::BF16, "gd2_out")?;
    let conv_out_buf = alloc(b * dims.conv_dim * dims.ksz, DType::BF16, "gd2_conv")?;
    let state_out_buf = alloc(bh * dims.dk * dims.dv, DType::F32, "gd2_state")?;
    let kn_buf = alloc(bh * l * dims.dk, DType::F32, "gd2_kn")?;
    let qn_buf = alloc(bh * l * dims.dk, DType::F32, "gd2_qn")?;
    let vc_buf = alloc(bh * l * dims.dv, DType::F32, "gd2_vc")?;
    let g_step_buf = alloc(bh * l, DType::F32, "gd2_gstep")?;
    let beta_buf = alloc(bh * l, DType::F32, "gd2_beta")?;
    let cap_gcs_buf = alloc(bh * l, DType::F32, "gd2_cap_gcs")?;
    let cap_delta_buf = alloc(bh * l * dims.dv, DType::F32, "gd2_cap_delta")?;
    let o_pre_buf = alloc(b * l * dims.value_dim, DType::F32, "gd2_o_pre")?;

    let encoder = device.command_encoder()?;
    call_gated_delta_v2(
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
        l,
        b,
        &bufs[0],
        &bufs[1],
        &bufs[2],
        &bufs[3],
        &bufs[4],
        &bufs[5],
        &bufs[6],
        &out_buf,
        &conv_out_buf,
        &state_out_buf,
        GatedDeltaV2Stages {
            kn: &kn_buf,
            qn: &qn_buf,
            vc: &vc_buf,
            g_step: &g_step_buf,
            beta: &beta_buf,
            cap_gcs: &cap_gcs_buf,
            cap_delta: &cap_delta_buf,
            o_pre: &o_pre_buf,
        },
    )
    .map_err(candle::Error::wrap)
    .context("fused gated-delta v2 dispatch")?;
    drop(encoder);
    drop(guards);

    let mk = |buf, count: usize, dtype, shape: Vec<usize>| -> Tensor {
        let storage = candle::MetalStorage::new(buf, device.clone(), count, dtype);
        Tensor::from_storage(Storage::Metal(storage), shape, BackpropOp::none(), false)
    };
    let out = mk(out_buf, b * l * dims.value_dim, DType::BF16, vec![b, l, dims.value_dim]);
    let conv_out = mk(
        conv_out_buf,
        b * dims.conv_dim * dims.ksz,
        DType::BF16,
        vec![b, dims.conv_dim, dims.ksz],
    );
    // Transposed layout: (b, heads, dv, dk).
    let state_out = mk(
        state_out_buf,
        bh * dims.dk * dims.dv,
        DType::F32,
        vec![b, dims.heads, dims.dv, dims.dk],
    );
    let capture = FusedChunkCapture {
        kc: mk(kn_buf, bh * l * dims.dk, DType::F32, vec![b, dims.heads, l, dims.dk]),
        delta: mk(
            cap_delta_buf,
            bh * l * dims.dv,
            DType::F32,
            vec![b, dims.heads, l, dims.dv],
        ),
        gcs: mk(cap_gcs_buf, bh * l, DType::F32, vec![b, dims.heads, l]),
    };
    Ok((out, conv_out, state_out, capture))
}
