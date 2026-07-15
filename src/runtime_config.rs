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

#[derive(Clone, Debug, Default)]
pub struct RuntimeConfig {
    /// Tensor-op (matmul2d) route: master switch, routing thresholds,
    /// split-K geometry, plane cache. See mm2d.rs for field receipts.
    pub mm2d: Arc<Mm2dConfig>,
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
            mtp_quantize_only: std::env::var("LMBRRR_MTP_Q_ONLY").ok(),
        }
    }
}
