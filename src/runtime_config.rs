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
    /// Route long prefill (l > the fused-chunk cap) through the fused chunk
    /// kernel, looped over sub-chunks carrying state, instead of the unfused
    /// tensor-path scan. Opt-in until the byte-parity gate ships it on.
    pub deltanet_prefill_fused: bool,
    /// Sub-chunk size for the fused-prefill loop (experiment knob: sweeps the
    /// dispatch-count vs intra-chunk-work tradeoff). Clamped to the kernel's
    /// GDC_MAX_L=12; 0/unset uses 12.
    pub deltanet_prefill_cap: usize,
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
            deltanet_prefill_fused: false,
            deltanet_prefill_cap: 12,
        }
    }
}

impl KernelRouteConfig {
    /// Entrypoint env resolution over the production defaults (the only
    /// place these vars are read). `LMBRRR_UNFUSED_*` invert their gate;
    /// `LMBRRR_FUSED_*`/`_DELTANET_V2` opt out with "0".
    pub fn from_env() -> Self {
        use crate::env_keys as k;
        let base = Self::default();
        let unfused = |key: &str| std::env::var(key).is_ok_and(|v| v == "1");
        let opt_out = |key: &str, default: bool| std::env::var(key).map_or(default, |v| v != "0");
        Self {
            fused_rmsnorm: !unfused(k::UNFUSED_RMSNORM),
            fused_sdpa: !unfused(k::UNFUSED_SDPA),
            fused_reconstruct: !unfused(k::UNFUSED_RECONSTRUCT),
            fused_deltanet: !unfused(k::UNFUSED_DELTANET),
            fused_attn_prep: opt_out(k::FUSED_ATTN_PREP, base.fused_attn_prep),
            fused_mtp_fc: opt_out(k::FUSED_MTP_FC, base.fused_mtp_fc),
            deltanet_v2: opt_out(k::DELTANET_V2, base.deltanet_v2),
            // Any presence enables the fallback (historical `is_ok()` sense).
            deltanet_sequential_fallback: std::env::var(k::DELTANET_SEQUENTIAL).is_ok(),
            deltanet_prefill_fused: std::env::var(k::DELTANET_PREFILL_FUSED).is_ok(),
            deltanet_prefill_cap: std::env::var(k::DELTANET_PREFILL_CAP)
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&c| (2..=12).contains(&c))
                .unwrap_or(12),
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
            enabled: std::env::var(crate::env_keys::PACK).map_or(true, |v| v != "0"),
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
        use crate::env_keys as k;
        let opt_out = |key: &str, default: bool| std::env::var(key).map_or(default, |v| v != "0");
        Self {
            async_readback: opt_out(k::ASYNC_READBACK, true),
            fused_argmax: opt_out(k::FUSED_ARGMAX, true),
        }
    }
}

/// Spec-run command knobs: diagnostics and reference-path selectors resolved
/// once at each spec-run command entry. COMMAND-SCOPE — deliberately not a
/// field of [`RuntimeConfig`]: these configure a run, not a model component,
/// and never thread into the model tree. All default off.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpecRunConfig {
    /// Re-enable per-phase `synchronize()` so draft/verify/rollback buckets
    /// measure GPU time. Off: the round pays exactly two readback waits and
    /// the buckets attribute encode+queue time only.
    pub loop_timing: bool,
    /// Restore the legacy restore + re-advance rollback (reference path for
    /// the state-selection mechanism).
    pub readvance_rollback: bool,
    /// Per-bucket verify/draft sync walls (fenced instrumentation).
    pub fenced_timing: bool,
    /// Adaptive MTP draft depth.
    pub adaptive_depth: bool,
    /// Per-segment fenced propose timing (DSpark drafter attribution).
    pub propose_timing: bool,
}

impl SpecRunConfig {
    pub fn from_env() -> Self {
        use crate::env_keys as k;
        let on = |key: &str| std::env::var(key).is_ok_and(|v| v == "1");
        Self {
            loop_timing: on(k::LOOP_TIMING),
            readvance_rollback: on(k::READVANCE_ROLLBACK),
            fenced_timing: on(k::SPEC_FENCED_TIMING),
            adaptive_depth: on(k::MTP_ADAPTIVE_DEPTH),
            propose_timing: on(k::PROPOSE_TIMING),
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
    /// and standalone command entries (tests use `Default`). Also warns on
    /// any `LMBRRR_*` var in the environment that no resolver consumes — the
    /// typo guard for the bench/suite scripts.
    pub fn from_env() -> Self {
        warn_unknown_lmbrrr_keys(std::env::vars().map(|(k, _)| k));
        Self {
            model: ModelCtx {
                mm2d: Arc::new(Mm2dConfig::from_env()),
                routes: Arc::new(KernelRouteConfig::from_env()),
            },
            pack: PackConfig::from_env(),
            decode: DecodeConfig::from_env(),
            mtp_quantize_only: std::env::var(crate::env_keys::MTP_Q_ONLY).ok(),
        }
    }
}

/// Emit a warning for every `LMBRRR_*` environment key not in the known-key
/// registry ([`crate::env_keys::KNOWN_LMBRRR_KEYS`]). Compile-time keys
/// (`LMBRRR_GIT_REV`, `LMBRRR_CANDLE_PIN`) are excluded — they are `env!`
/// build inputs, never resolved at runtime. Factored out so it is unit
/// testable without mutating the process environment.
fn warn_unknown_lmbrrr_keys(env_keys_present: impl Iterator<Item = String>) {
    for unknown in unknown_lmbrrr_keys(env_keys_present) {
        eprintln!(
            "warning: {unknown} is set but unknown to lmbrrr — typo? \
             (no config resolver reads it)"
        );
    }
}

/// The `LMBRRR_*` keys present in `env_keys_present` that are neither in the
/// runtime registry nor a compile-time (`env!`) key. Pure; the testable core
/// of [`warn_unknown_lmbrrr_keys`].
fn unknown_lmbrrr_keys(env_keys_present: impl Iterator<Item = String>) -> Vec<String> {
    const COMPILE_TIME: &[&str] = &["LMBRRR_GIT_REV", "LMBRRR_CANDLE_PIN"];
    env_keys_present
        .filter(|k| {
            k.starts_with("LMBRRR_")
                && !crate::env_keys::KNOWN_LMBRRR_KEYS.contains(&k.as_str())
                && !COMPILE_TIME.contains(&k.as_str())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_keys_flag_typos_not_known_or_compile_time_keys() {
        let present = [
            crate::env_keys::MM2D.to_string(), // known -> ok
            "LMBRRR_GIT_REV".to_string(),      // compile-time -> ok
            "LMBRRR_MM2D_TYPOO".to_string(),   // typo -> flagged
            "PATH".to_string(),                // non-lmbrrr -> ignored
        ];
        let unknown = unknown_lmbrrr_keys(present.into_iter());
        assert_eq!(unknown, vec!["LMBRRR_MM2D_TYPOO".to_string()]);
    }

    #[test]
    fn every_known_key_is_a_distinct_lmbrrr_name() {
        let keys = crate::env_keys::KNOWN_LMBRRR_KEYS;
        for k in keys {
            assert!(k.starts_with("LMBRRR_"), "{k} is not an LMBRRR_ key");
        }
        let mut sorted = keys.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), keys.len(), "duplicate key in the registry");
    }
}
