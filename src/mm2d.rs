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
use std::sync::Arc;

use anyhow::{Context, Result};
use candle::op::BackpropOp;
use candle::quantized::k_quants::BlockQ4K;
use candle::quantized::{GgmlDType, QTensor};
use candle::{DType, MetalDevice, Storage, Tensor};
use candle_metal_kernels::metal::Buffer;
use candle_metal_kernels::{
    call_quantized_matmul_mm2d_q4k, call_quantized_matmul_mm2d_q4k_splitk,
};

/// Tensor-op route configuration. Constructed ONCE at the entrypoint
/// (usually via [`Mm2dConfig::from_env`]) and threaded to every
/// [`crate::quantized_linear::MixedLinear`] at construction — modules never
/// read the environment themselves. Defaults are this model's arbitrated
/// values; a new model re-tunes by constructing a different config.
#[derive(Clone, Debug)]
pub struct Mm2dConfig {
    /// Route master switch (default off until the verify refit ships it).
    pub enabled: bool,
    /// Minimum weight rows (n) for the tensor-op route. Small-n dispatches
    /// have too few threadgroups to hide the serial K-loop latency in a
    /// dependent layer chain; min_n isolates that effect (e.g. 100000 =
    /// lm_head only).
    pub min_n: usize,
    /// Minimum chunk rows for routing BODY linears (n below the head class)
    /// through the tensor op. With split-K (the default) the body wins at
    /// every m in [2,8]: suite mean 145.2 vs 137.5 tok/s at body_min_m=2 vs
    /// the head-only route (2026-07-14). Without split-K the crossover was
    /// m=5.
    pub body_min_m: usize,
    /// Head-class boundary: weights with n at/above this route as the vocab
    /// head (unsplit, any m); below it as body shapes (split-K, m-gated).
    /// The default fits this model's 248k vocab; small-vocab (32k) models
    /// need a lower boundary.
    pub head_min_n: usize,
    /// Split-K for body shapes. Default ON: the in-loop A/B (2026-07-14, d5
    /// warm) cut verify 13.36 -> 10.98 ms/round and the suite confirmed the
    /// mean win; off restores the single-dispatch kernel.
    pub splitk: bool,
    /// Split-K grid target (threadgroups). Suite-arbitrated at 128
    /// (2026-07-15): 384 cut the round 2ms (gate_up finally split) but lost
    /// the suite mean 182.6 -> 176.7 — the deeper split's
    /// accumulation-order perturbation flips enough drafter near-ties to
    /// eat the speed. Retest after acceptance rises.
    pub split_target_tgs: usize,
    /// Content-addressed plane cache directory; None disables caching.
    pub plane_cache_dir: Option<std::path::PathBuf>,
    /// Fused verify-head argmax. Opt-in: falsified in-loop 2026-07-15
    /// (byte-identical but 11.71 -> 11.82 ms/round — the saved logits write
    /// is ~0.02ms while the threadgroup argmax epilogue costs more than the
    /// 0.13ms fast_argmax pass it replaces). Kept for deep-chunk scenarios.
    pub fused_verify_argmax: bool,
}

impl Default for Mm2dConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_n: 0,
            body_min_m: 2,
            head_min_n: 100_000,
            splitk: true,
            split_target_tgs: 128,
            // Pure default: no caching. from_env (the entrypoint path)
            // supplies the ~/.cache location so tests stay deterministic.
            plane_cache_dir: None,
            fused_verify_argmax: false,
        }
    }
}

impl Mm2dConfig {
    /// The entrypoint's env resolution over the arbitrated defaults. The only
    /// place these variables are read (keys in `crate::env_keys`).
    pub fn from_env() -> Self {
        use crate::env_keys as k;
        let base = Self::default();
        let parse = |key: &str, default: usize| -> usize {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        Self {
            enabled: std::env::var(k::MM2D).is_ok_and(|v| v != "0"),
            min_n: parse(k::MM2D_MIN_N, base.min_n),
            body_min_m: parse(k::MM2D_BODY_MIN_M, base.body_min_m),
            head_min_n: parse(k::MM2D_HEAD_MIN_N, base.head_min_n),
            splitk: std::env::var(k::MM2D_SPLITK).map_or(base.splitk, |v| v != "0"),
            split_target_tgs: parse(k::MM2D_SPLIT_TGS, base.split_target_tgs),
            plane_cache_dir: if std::env::var(k::MM2D_PLANE_CACHE).is_ok_and(|v| v == "0") {
                None
            } else {
                std::env::var(k::MM2D_CACHE_DIR)
                    .map(std::path::PathBuf::from)
                    .ok()
                    .or_else(default_plane_cache_dir)
            },
            fused_verify_argmax: std::env::var(k::FUSED_VERIFY_ARGMAX).is_ok_and(|v| v == "1"),
        }
    }
}

fn default_plane_cache_dir() -> Option<std::path::PathBuf> {
    Some(std::path::PathBuf::from(std::env::var(crate::env_keys::HOME).ok()?).join(".cache/lmbrrr/mm2d"))
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

/// Plane layout version — bump whenever the fork's q4k_mm2d_planes layout
/// changes; encoded in cache filenames so stale entries miss.
const PLANE_CACHE_VERSION: u32 = 1;

impl Mm2dPlanes {
    /// Repacked planes for a q4_K weight: cache read (`cache_dir` from the
    /// config — the repack is a pure function of the block bytes, cached by
    /// their sha256), or CPU repack of the resident ggml blocks (cached for
    /// the next process), then upload.
    pub fn from_qtensor(
        weight: &QTensor,
        device: &MetalDevice,
        cache_dir: Option<&std::path::Path>,
    ) -> Result<Self> {
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

        // Plane sizes are a pure function of (n, k): nibble plane [K, Npad]
        // at one nibble each; dsc/dmm [Npad, K/32] fp16.
        let n_pad = n.div_ceil(64) * 64;
        let nib_len = k * n_pad / 2;
        let scale_len = n_pad * (k / 32) * 2;
        let total_len = nib_len + 2 * scale_len;

        let cache_path = cache_dir.and_then(|dir| {
            std::fs::create_dir_all(dir).ok()?;
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&*data);
            let digest = hasher.finalize();
            let short = digest[..8]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            Some(dir.join(format!("{short}-{n}x{k}.v{PLANE_CACHE_VERSION}.bin")))
        });

        let upload = |bytes: &[u8], label: &str| -> Result<Arc<Buffer>> {
            device
                .new_buffer_builder()
                .with_data(bytes)
                .with_label(label)
                .build()
                .with_context(|| format!("upload {label}"))
        };

        if let Some(path) = &cache_path {
            if let Ok(bytes) = std::fs::read(path) {
                if bytes.len() == total_len {
                    return Ok(Self {
                        nibbles: upload(&bytes[..nib_len], "mm2d_nibbles")?,
                        dsc: upload(&bytes[nib_len..nib_len + scale_len], "mm2d_dsc")?,
                        dmm: upload(&bytes[nib_len + scale_len..], "mm2d_dmm")?,
                        n,
                        n_pad,
                        k,
                    });
                }
                // Wrong size = stale layout for this version tag; rebuild.
            }
        }

        let blocks = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const BlockQ4K,
                data.len() / std::mem::size_of::<BlockQ4K>(),
            )
        };
        let planes = candle::quantized::metal::q4k_mm2d_planes(blocks, n, k)?;
        anyhow::ensure!(
            planes.n_pad == n_pad && planes.nibbles.len() == nib_len,
            "plane layout drifted from the cache's size model — bump PLANE_CACHE_VERSION"
        );
        let as_bytes = |p: *const u8, len: usize| unsafe { std::slice::from_raw_parts(p, len) };
        let dsc_bytes = as_bytes(planes.dsc.as_ptr() as *const u8, planes.dsc.len() * 2);
        let dmm_bytes = as_bytes(planes.dmm.as_ptr() as *const u8, planes.dmm.len() * 2);

        if let Some(path) = &cache_path {
            // Atomic publish; concurrent writers produce identical bytes.
            let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
            let write = || -> std::io::Result<()> {
                use std::io::Write;
                let mut f = std::fs::File::create(&tmp)?;
                f.write_all(&planes.nibbles)?;
                f.write_all(dsc_bytes)?;
                f.write_all(dmm_bytes)?;
                f.flush()?;
                std::fs::rename(&tmp, path)
            };
            if let Err(err) = write() {
                let _ = std::fs::remove_file(&tmp);
                eprintln!("warning: mm2d plane cache write failed ({err})");
            }
        }

        Ok(Self {
            nibbles: upload(&planes.nibbles, "mm2d_nibbles")?,
            dsc: upload(dsc_bytes, "mm2d_dsc")?,
            dmm: upload(dmm_bytes, "mm2d_dmm")?,
            n: planes.n,
            n_pad: planes.n_pad,
            k: planes.k,
        })
    }
}

/// Whether this activation shape is eligible for the tensor-op route.
pub fn mm2d_eligible(xs: &Tensor, cfg: &Mm2dConfig) -> bool {
    if !cfg.enabled || MM2D_BROKEN.load(Ordering::Relaxed) {
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
pub fn mm2d_q4k_forward(xs: &Tensor, planes: &Mm2dPlanes, cfg: &Mm2dConfig) -> Result<Tensor> {
    anyhow::ensure!(planes.n >= cfg.min_n, "below the configured mm2d min_n");
    if planes.n < cfg.head_min_n {
        let dims = xs.dims();
        let m: usize = dims[..dims.len() - 1].iter().product();
        anyhow::ensure!(m >= cfg.body_min_m, "body shape below body_min_m");
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
    // the grid reaches the target threadgroup count. The 128 target leaves
    // gate_up (N=7168, 112 TGs) unsplit at 47 GB/s — 24 plain dispatches
    // ~2ms/round in the labeled trace (2026-07-15); 256 splits it 2x but is
    // margin-class (split_target_tgs re-arbitrates per arm).
    let n_splits = if planes.n < cfg.head_min_n && cfg.splitk {
        (cfg.split_target_tgs / (planes.n_pad / 64)).clamp(1, planes.k / 32)
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
/// apply (caller falls back); gated by cfg.fused_verify_argmax (see its
/// falsification note).
pub fn mm2d_head_argmax(
    xs: &Tensor,
    head: &crate::quantized_linear::MixedLinear,
    cfg: &Mm2dConfig,
) -> Result<Option<Tensor>> {
    if !cfg.fused_verify_argmax || !cfg.enabled || MM2D_BROKEN.load(Ordering::Relaxed) {
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
