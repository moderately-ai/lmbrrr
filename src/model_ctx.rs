//! Model construction context — the threaded DI seam and component factory.
//!
//! `ModelCtx` is the construction-time layer of [`crate::runtime_config::RuntimeConfig`],
//! resolved once at the entrypoint and threaded by `&ModelCtx` through every
//! model constructor. It carries the config slices the model tree needs
//! (`mm2d`, `routes`) and exposes factory methods for the components whose
//! construction otherwise crosses module boundaries.
//!
//! Portability: porting to a different model means constructing a different
//! `ModelCtx` (different `Mm2dConfig`/`KernelRouteConfig`) at the root — not
//! grepping the tree for env reads. Nothing below the root reads the
//! environment.

use std::sync::Arc;

use candle::quantized::QTensor;

use crate::mm2d::Mm2dConfig;
use crate::quantized_linear::MixedLinear;
use crate::runtime_config::KernelRouteConfig;

/// Threaded construction context + component factory. Cheap to clone (two
/// `Arc`s); `Default` is the production configuration.
#[derive(Clone, Debug, Default)]
pub struct ModelCtx {
    /// Tensor-op (matmul2d) route config; stored on every quantized linear
    /// this ctx builds.
    pub mm2d: Arc<Mm2dConfig>,
    /// Kernel-fusion route gates; stored (narrowly) on the layers that read
    /// them at forward time.
    pub routes: Arc<KernelRouteConfig>,
}

impl ModelCtx {
    /// Factory: build a quantized [`MixedLinear`] wired with this ctx's mm2d
    /// route. Call sites use this instead of naming `Mm2dConfig` — the
    /// construction policy is inverted to the ctx.
    pub fn quantized_linear(&self, weight: QTensor) -> candle::Result<MixedLinear> {
        MixedLinear::from_qtensor(weight, self.mm2d.clone())
    }
}
