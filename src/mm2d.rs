//! Host-side wrapper for the fork's tensor-op (matmul2d) q4_K matmul.
//!
//! Verify chunks (m in [2,8]) ride the matrix units at flat cost: the whole
//! 8-row tile costs ~1.25x one m=1 GEMV read on the lm_head shape (receipts
//! on eval-matmul2d-uint4b-tensor-op). Exact q4_K semantics — per-32
//! sub-block scales folded on the C tile — but NOT bit-compatible with the
//! mv/wide kernels (different accumulation order + fp16 d*sc planes):
//! margin-class, oracle-gated, LMBRRR_MM2D opt-in.
//!
//! The kernel consumes a repacked plane layout built once per tensor (CPU,
//! lazy on first eligible forward) from the resident ggml blocks. macOS
//! releases without Metal-4 tensor ops fail at metallib load; the first
//! failure disables the route process-wide (warn once) and callers fall
//! back to the wide kernels.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use candle::op::BackpropOp;
use candle::quantized::k_quants::BlockQ4K;
use candle::quantized::{GgmlDType, QTensor};
use candle::{DType, MetalDevice, Storage, Tensor};
use candle_metal_kernels::metal::Buffer;
use candle_metal_kernels::{
    call_quantized_matmul_mm2d_q4k, call_quantized_matmul_mm2d_q4k_splitk,
};

/// Route master switch (default off until the verify refit ships it).
pub fn mm2d_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LMBRRR_MM2D").is_ok_and(|v| v != "0"))
}

/// Minimum weight rows (n) for the tensor-op route. Small-n dispatches have
/// too few threadgroups to hide the serial K-loop latency in a dependent
/// layer chain; LMBRRR_MM2D_MIN_N isolates that effect (e.g. 100000 =
/// lm_head only).
pub fn mm2d_min_n() -> usize {
    static MIN_N: OnceLock<usize> = OnceLock::new();
    *MIN_N.get_or_init(|| {
        std::env::var("LMBRRR_MM2D_MIN_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Minimum chunk rows for routing BODY linears (n below the head class)
/// through the tensor op. With split-K (the default) the body wins at every
/// m in [2,8]: suite mean 145.2 vs 137.5 tok/s at BODY_MIN_M=2 vs the
/// head-only route (2026-07-14). Without split-K the crossover was m=5.
pub fn mm2d_body_min_m() -> usize {
    static MIN_M: OnceLock<usize> = OnceLock::new();
    *MIN_M.get_or_init(|| {
        std::env::var("LMBRRR_MM2D_BODY_MIN_M")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2)
    })
}

const HEAD_CLASS_MIN_N: usize = 100_000;

/// Split-K for body shapes. Default ON: the in-loop A/B (2026-07-14, d5
/// warm) cut verify 13.36 -> 10.98 ms/round and the suite confirmed the
/// mean win; LMBRRR_MM2D_SPLITK=0 restores the single-dispatch kernel.
pub fn mm2d_splitk_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LMBRRR_MM2D_SPLITK").map_or(true, |v| v != "0"))
}

/// Split-K grid target (threadgroups). Suite-arbitrated at 128
/// (2026-07-15): 384 cut the round 2ms (gate_up finally split) but lost
/// the suite mean 182.6 -> 176.7 — the deeper split's accumulation-order
/// perturbation flips enough drafter near-ties to eat the speed. Retest
/// after r2 lifts acceptance (fewer near-ties); LMBRRR_MM2D_SPLIT_TGS
/// overrides.
pub fn mm2d_split_target_tgs() -> usize {
    static TGS: OnceLock<usize> = OnceLock::new();
    *TGS.get_or_init(|| {
        std::env::var("LMBRRR_MM2D_SPLIT_TGS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128)
    })
}

/// Process-wide kill switch, set on the first dispatch failure (pre-26.4 OS).
static MM2D_BROKEN: AtomicBool = AtomicBool::new(false);

/// Device-resident repacked planes for one q4_K weight (see
/// candle::quantized::metal::q4k_mm2d_planes for the layout).
pub struct Mm2dPlanes {
    nibbles: Arc<Buffer>,
    dsc: Arc<Buffer>,
    dmm: Arc<Buffer>,
    n: usize,
    n_pad: usize,
    k: usize,
}

impl Mm2dPlanes {
    /// CPU repack of the tensor's ggml blocks + upload. One-time per tensor
    /// (the wrapper caches it); the blit readback + nibble transpose costs
    /// seconds on the lm_head — acceptable as a first-verify warmup cost for
    /// the eval; pack-sidecar caching is the production follow-up.
    pub fn from_qtensor(weight: &QTensor, device: &MetalDevice) -> Result<Self> {
        anyhow::ensure!(
            weight.dtype() == GgmlDType::Q4K,
            "mm2d planes need q4_K (got {:?})",
            weight.dtype()
        );
        let dims = weight.shape().dims();
        anyhow::ensure!(dims.len() == 2, "mm2d planes need [n, k]; got {dims:?}");
        let (n, k) = (dims[0], dims[1]);
        let data = weight.data().context("read ggml blocks")?;
        anyhow::ensure!(
            data.len() % std::mem::size_of::<BlockQ4K>() == 0,
            "q4_K data length {} is not block-aligned",
            data.len()
        );
        let blocks = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const BlockQ4K,
                data.len() / std::mem::size_of::<BlockQ4K>(),
            )
        };
        let planes = candle::quantized::metal::q4k_mm2d_planes(blocks, n, k)?;
        let upload = |bytes: &[u8], label: &str| -> Result<Arc<Buffer>> {
            device
                .new_buffer_builder()
                .with_data(bytes)
                .with_label(label)
                .build()
                .with_context(|| format!("upload {label}"))
        };
        let as_bytes = |p: *const u8, len: usize| unsafe { std::slice::from_raw_parts(p, len) };
        Ok(Self {
            nibbles: upload(&planes.nibbles, "mm2d_nibbles")?,
            dsc: upload(
                as_bytes(planes.dsc.as_ptr() as *const u8, planes.dsc.len() * 2),
                "mm2d_dsc",
            )?,
            dmm: upload(
                as_bytes(planes.dmm.as_ptr() as *const u8, planes.dmm.len() * 2),
                "mm2d_dmm",
            )?,
            n: planes.n,
            n_pad: planes.n_pad,
            k: planes.k,
        })
    }
}

/// Whether this activation shape is eligible for the tensor-op route.
pub fn mm2d_eligible(xs: &Tensor) -> bool {
    if !mm2d_enabled() || MM2D_BROKEN.load(Ordering::Relaxed) {
        return false;
    }
    if xs.dtype() != DType::BF16 || !xs.device().is_metal() {
        return false;
    }
    let dims = xs.dims();
    let m: usize = dims[..dims.len() - 1].iter().product();
    (2..=8).contains(&m)
}

/// One matmul through the tensor-op kernel: xs [.., m, k] BF16 -> [.., m, n]
/// BF16. Errors mark the route broken process-wide (warn once) so the caller
/// can fall back; pass a fresh forward to the wide route on Err.
pub fn mm2d_q4k_forward(xs: &Tensor, planes: &Mm2dPlanes) -> Result<Tensor> {
    anyhow::ensure!(planes.n >= mm2d_min_n(), "below LMBRRR_MM2D_MIN_N");
    if planes.n < HEAD_CLASS_MIN_N {
        let dims = xs.dims();
        let m: usize = dims[..dims.len() - 1].iter().product();
        anyhow::ensure!(m >= mm2d_body_min_m(), "body shape below BODY_MIN_M");
    }
    let candle::Device::Metal(device) = xs.device() else {
        anyhow::bail!("mm2d forward requires a Metal device");
    };
    let dims = xs.dims().to_vec();
    let k = *dims.last().context("empty activation shape")?;
    let m: usize = dims[..dims.len() - 1].iter().product();
    anyhow::ensure!(k == planes.k, "activation k={k} vs planes k={}", planes.k);
    anyhow::ensure!((1..=8).contains(&m), "mm2d m={m} out of tile range");

    let xs = if xs.is_contiguous() {
        xs.clone()
    } else {
        xs.contiguous()?
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Metal(ms) = &*storage else {
        anyhow::bail!("activations are not Metal-resident");
    };
    let lhs_offset = layout.start_offset() * 2;

    let dst = device
        .new_buffer_builder()
        .with_size(m * planes.n * 2)
        .with_label("mm2d_dst")
        .build()?;

    // Split-K for body shapes (in-loop arbitrated): small-N grids
    // under-occupy the GPU on a serial K loop; partition the K/32 slices so
    // the grid reaches the target threadgroup count. The original 128
    // target left gate_up (N=7168, 112 TGs) unsplit at 47 GB/s — 24 plain
    // dispatches ~2ms/round in the labeled trace (2026-07-15); 256 splits
    // it 2x. LMBRRR_MM2D_SPLIT_TGS overrides the target for A/B.
    let n_splits = if planes.n < HEAD_CLASS_MIN_N && mm2d_splitk_enabled() {
        (mm2d_split_target_tgs() / (planes.n_pad / 64)).clamp(1, planes.k / 32)
    } else {
        1
    };
    let partials = if n_splits > 1 {
        Some(
            device
                .new_buffer_builder()
                .with_size(n_splits * 8 * planes.n_pad * 4)
                .with_label("mm2d_splitk_partials")
                .build()?,
        )
    } else {
        None
    };

    let dispatch = || -> Result<()> {
        let encoder = device.command_encoder().context("mm2d encoder")?;
        match &partials {
            Some(partials) => call_quantized_matmul_mm2d_q4k_splitk(
                device.metal_device(),
                &encoder,
                device.kernels(),
                (m, planes.n, planes.n_pad, planes.k, n_splits),
                ms.buffer(),
                lhs_offset,
                &planes.nibbles,
                &planes.dsc,
                &planes.dmm,
                partials,
                0,
                &dst,
            )
            .context("mm2d splitk dispatch")?,
            None => call_quantized_matmul_mm2d_q4k(
                device.metal_device(),
                &encoder,
                device.kernels(),
                (m, planes.n, planes.n_pad, planes.k),
                ms.buffer(),
                lhs_offset,
                &planes.nibbles,
                &planes.dsc,
                &planes.dmm,
                0,
                &dst,
            )
            .context("mm2d matmul dispatch")?,
        }
        Ok(())
    };
    if let Err(err) = dispatch() {
        if !MM2D_BROKEN.swap(true, Ordering::Relaxed) {
            eprintln!("warning: mm2d q4_K route unavailable, using wide kernels ({err})");
        }
        anyhow::bail!("mm2d dispatch failed: {err}");
    }
    drop(storage);

    let mut out_dims = dims;
    *out_dims.last_mut().expect("non-empty dims") = planes.n;
    let storage =
        candle::MetalStorage::new(dst, device.clone(), m * planes.n, DType::BF16);
    Ok(Tensor::from_storage(
        Storage::Metal(storage),
        out_dims,
        BackpropOp::none(),
        false,
    ))
}

/// Fused verify-head argmax: per-row argmax ids (device U32 [m]) straight
/// from the mm2d planes — the m x V logits tensor is never materialized.
/// bf16-rounded compares + lowest-index ties keep it byte-identical to
/// head-forward + fast_argmax. Returns Ok(None) when the route does not
/// apply (caller falls back); LMBRRR_FUSED_VERIFY_ARGMAX=0 disables.
pub fn mm2d_head_argmax(
    xs: &Tensor,
    head: &crate::quantized_linear::MixedLinear,
) -> Result<Option<Tensor>> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        std::env::var("LMBRRR_FUSED_VERIFY_ARGMAX").map_or(true, |v| v != "0")
    });
    if !enabled || !mm2d_enabled() || MM2D_BROKEN.load(Ordering::Relaxed) {
        return Ok(None);
    }
    let Some(planes) = head.mm2d_planes() else {
        return Ok(None);
    };
    if xs.dtype() != DType::BF16 || !xs.device().is_metal() {
        return Ok(None);
    }
    let dims = xs.dims().to_vec();
    let k = *dims.last().context("empty hidden shape")?;
    let m: usize = dims[..dims.len() - 1].iter().product();
    if !(2..=8).contains(&m) || k != planes.k {
        return Ok(None);
    }
    let candle::Device::Metal(device) = xs.device() else {
        return Ok(None);
    };

    let xs = if xs.is_contiguous() {
        xs.clone()
    } else {
        xs.contiguous()?
    };
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Metal(ms) = &*storage else {
        anyhow::bail!("hidden not Metal-resident");
    };
    let lhs_offset = layout.start_offset() * 2;
    let n_tiles = planes.n_pad / 64;
    let pval = device
        .new_buffer_builder()
        .with_size(n_tiles * 8 * 4)
        .with_label("mm2d_amax_pval")
        .build()?;
    let pidx = device
        .new_buffer_builder()
        .with_size(n_tiles * 8 * 4)
        .with_label("mm2d_amax_pidx")
        .build()?;
    let out = device
        .new_buffer_builder()
        .with_size(m * 4)
        .with_label("mm2d_amax_out")
        .build()?;

    let dispatch = || -> Result<()> {
        let encoder = device.command_encoder().context("mm2d argmax encoder")?;
        candle_metal_kernels::call_quantized_matmul_mm2d_q4k_argmax(
            device.metal_device(),
            &encoder,
            device.kernels(),
            (m, planes.n, planes.n_pad, planes.k),
            ms.buffer(),
            lhs_offset,
            &planes.nibbles,
            &planes.dsc,
            &planes.dmm,
            &pval,
            &pidx,
            &out,
        )
        .context("mm2d argmax dispatch")?;
        Ok(())
    };
    if let Err(err) = dispatch() {
        if !MM2D_BROKEN.swap(true, Ordering::Relaxed) {
            eprintln!("warning: mm2d argmax unavailable ({err})");
        }
        return Ok(None);
    }
    drop(storage);

    let storage = candle::MetalStorage::new(out, device.clone(), m, DType::U32);
    Ok(Some(Tensor::from_storage(
        Storage::Metal(storage),
        vec![m],
        BackpropOp::none(),
        false,
    )))
}
