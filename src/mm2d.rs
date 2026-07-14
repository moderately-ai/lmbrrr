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
use candle_metal_kernels::call_quantized_matmul_mm2d_q4k;

/// Route master switch (default off until the verify refit ships it).
pub fn mm2d_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LMBRRR_MM2D").is_ok_and(|v| v != "0"))
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

    let dispatch = || -> Result<()> {
        let encoder = device.command_encoder().context("mm2d encoder")?;
        call_quantized_matmul_mm2d_q4k(
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
        .context("mm2d matmul dispatch")?;
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
