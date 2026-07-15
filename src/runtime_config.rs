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

#[derive(Clone, Debug, Default)]
pub struct RuntimeConfig {
    /// Tensor-op (matmul2d) route: master switch, routing thresholds,
    /// split-K geometry, plane cache. See mm2d.rs for field receipts.
    pub mm2d: Arc<Mm2dConfig>,
    /// Text-model kernel-fusion route gates (see qwen35.rs).
    pub routes: Arc<KernelRouteConfig>,
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
            mm2d: Arc::new(Mm2dConfig::from_env()),
            routes: Arc::new(KernelRouteConfig::from_env()),
            mtp_quantize_only: std::env::var("LMBRRR_MTP_Q_ONLY").ok(),
        }
    }
}
