//! Process-level runtime configuration: the composition root.
//!
//! Every tunable and route gate is resolved ONCE at the entrypoint —
//! `RuntimeConfig::from_env()` — and threaded down the ownership tree at
//! construction. Modules hold their domain slice (`Arc`) and never read the
//! environment; the env vars remain the inter-process tunable surface for
//! the bench/suite scripts, but WHERE they are read is exactly here (each
//! domain struct's `from_env`, invoked only by this root).
//!
//! Portability note: porting to a new model means constructing different
//! configs (defaults live on each struct's `Default` with their arbitration
//! receipts in field docs) — not grepping for env reads.

use std::sync::Arc;

use crate::mm2d::Mm2dConfig;
use crate::model_ctx::ModelCtx;

/// Kernel-fusion / route gates for the text model, resolved at the
/// entrypoint and stored on each layer at construction (see qwen35.rs).
/// Fields are POSITIVE-SENSE (`fused_* = true` means the fused kernel
/// runs); the falsification/bit-identity receipts live in the field docs.
/// `Default` is the production configuration.
#[derive(Clone, Debug)]
pub struct KernelRouteConfig {
    /// Fused single-dispatch rmsnorm (F32 accumulate). Off = the reference
    /// unfused chain, for drift attribution.
    pub fused_rmsnorm: bool,
    /// Fused SDPA (vs. the reference softmax attention).
    pub fused_sdpa: bool,
    /// Fused single-dispatch DeltaNet rollback-state reconstruction (vs the
    /// f32 broadcast/exp/GEMM host chain).
    pub fused_reconstruct: bool,
    /// Fused DeltaNet decode/chunk kernels (vs the tensor-op reference).
    pub fused_deltanet: bool,
    /// Fused q/k head-norm + partial rope + direct KV-cache write for the
    /// full-attention layers (bit-identical to the unfused chain).
    pub fused_attn_prep: bool,
    /// Fused MTP pre-fc chain (2x rmsnorm + concat + fc GEMV in one
    /// dispatch; bit-identical at m == 1).
    pub fused_mtp_fc: bool,
    /// v2 fused DeltaNet (re-gridded decode/chunk kernels, transposed
    /// state layout) vs the v1 kernels.
    pub deltanet_v2: bool,
    /// Sequential (per-step) DeltaNet recurrence instead of the chunked
    /// WY/UT path — a reference fallback for seq_len > 1.
    pub deltanet_sequential_fallback: bool,
}

impl Default for KernelRouteConfig {
    fn default() -> Self {
        Self {
            fused_rmsnorm: true,
            fused_sdpa: true,
            fused_reconstruct: true,
            fused_deltanet: true,
            fused_attn_prep: true,
            fused_mtp_fc: true,
            deltanet_v2: true,
            deltanet_sequential_fallback: false,
        }
    }
}

impl KernelRouteConfig {
    /// Entrypoint env resolution over the production defaults (the only
    /// place these vars are read). `LMBRRR_UNFUSED_*` invert their gate;
    /// `LMBRRR_FUSED_*`/`_DELTANET_V2` opt out with "0".
    pub fn from_env() -> Self {
        let base = Self::default();
        let unfused = |key: &str| std::env::var(key).is_ok_and(|v| v == "1");
        let opt_out = |key: &str, default: bool| {
            std::env::var(key).map_or(default, |v| v != "0")
        };
        Self {
            fused_rmsnorm: !unfused("LMBRRR_UNFUSED_RMSNORM"),
            fused_sdpa: !unfused("LMBRRR_UNFUSED_SDPA"),
            fused_reconstruct: !unfused("LMBRRR_UNFUSED_RECONSTRUCT"),
            fused_deltanet: !unfused("LMBRRR_UNFUSED_DELTANET"),
            fused_attn_prep: opt_out("LMBRRR_FUSED_ATTN_PREP", base.fused_attn_prep),
            fused_mtp_fc: opt_out("LMBRRR_FUSED_MTP_FC", base.fused_mtp_fc),
            deltanet_v2: opt_out("LMBRRR_DELTANET_V2", base.deltanet_v2),
            // Any presence enables the fallback (historical `is_ok()` sense).
            deltanet_sequential_fallback: std::env::var("LMBRRR_DELTANET_SEQUENTIAL").is_ok(),
        }
    }
}

/// GGML-ready weight pack (sidecar) gate.
#[derive(Clone, Copy, Debug)]
pub struct PackConfig {
    /// Use + write the pack sidecar (skips the per-start requantize). Off
    /// forces cold requantization every start.
    pub enabled: bool,
}

impl Default for PackConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl PackConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: std::env::var("LMBRRR_PACK").map_or(true, |v| v != "0"),
        }
    }
}

/// Decode-loop path selection (host-side, model-agnostic).
#[derive(Clone, Copy, Debug)]
pub struct DecodeConfig {
    /// Async event-driven readback for the greedy device-chain (vs the
    /// batched-flush path); the batched path is always used off Metal.
    pub async_readback: bool,
    /// Fused GEMV+argmax head (vs materializing logits then argmax), when
    /// the model supports it.
    pub fused_argmax: bool,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            async_readback: true,
            fused_argmax: true,
        }
    }
}

impl DecodeConfig {
    pub fn from_env() -> Self {
        let opt_out = |key: &str, default: bool| {
            std::env::var(key).map_or(default, |v| v != "0")
        };
        Self {
            async_readback: opt_out("LMBRRR_ASYNC_READBACK", true),
            fused_argmax: opt_out("LMBRRR_FUSED_ARGMAX", true),
        }
    }
}

/// Composition root: resolved once per command entry (the sole env reader),
/// then its layers thread down to their consumers. LAYERED — `model` is the
/// construction-time context threaded into the model tree; the rest are
/// run-time / command-scope knobs consumed at their point of use.
#[derive(Clone, Debug, Default)]
pub struct RuntimeConfig {
    /// Construction-time context (mm2d + fusion routes) + component
    /// factories; threaded by `&ctx` into every model constructor.
    pub model: ModelCtx,
    /// Weight-pack sidecar gate (see pack.rs).
    pub pack: PackConfig,
    /// Decode-loop path selection (see generate.rs).
    pub decode: DecodeConfig,
    /// MTP-quantization bisection hook (LMBRRR_MTP_Q_ONLY =
    /// fc|qkv|o|gate_up|down): quantize a single head linear to isolate
    /// per-path damage. Diagnostics only; None in production.
    pub mtp_quantize_only: Option<String>,
}

impl RuntimeConfig {
    /// The entrypoint's env resolution. The only call sites are `main()`
    /// and standalone command entries (tests use `Default`).
    pub fn from_env() -> Self {
        Self {
            model: ModelCtx {
                mm2d: Arc::new(Mm2dConfig::from_env()),
                routes: Arc::new(KernelRouteConfig::from_env()),
            },
            pack: PackConfig::from_env(),
            decode: DecodeConfig::from_env(),
            mtp_quantize_only: std::env::var("LMBRRR_MTP_Q_ONLY").ok(),
        }
    }
}
