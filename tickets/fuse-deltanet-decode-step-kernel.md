---
id: fuse-deltanet-decode-step-kernel
title: Fuse DeltaNet decode step into one Metal kernel
status: todo
priority: p1
dependencies: []
related: [measure-metal-roofline-and-dispatch-overhead, optimize-deltanet-chunked-prefill-and-verify-throughput]
scopes: [runtime/candle, runtime/metal]
shared_scopes: [docs/research]
paths: [src/qwen35.rs, src/**, docs/research/fused-deltanet-decode-kernel.md]
tags: [performance, kernels, deltanet, campaign-1000]
---
## Goal

Collapse the single-token GatedDeltaNet decode step — conv state update + silu, gates (sigmoid/softplus-style), l2norm, recurrent delta rule, output RMSNorm + silu gate — from ~30 small tensor dispatches per layer into one (or two) custom Metal kernel dispatches per layer. 18 DeltaNet layers dominate the per-token dispatch budget at hidden 1024.

## Acceptance

- Implement as a Candle `CustomOp` with `metal_fwd`, compiling embedded MSL at runtime via `MetalDevice::new_library_with_source` (no candle-core fork required; UgIOp1 in candle-core/src/custom_op.rs is the template). State tensors updated in place via the inplace-op variant.
- Numerics: advisory drift report against the unfused path (logits parity + long-generation text diff) published with the change; drift is accepted under the campaign quality bar but must be visible.
- Measure decode tok/s before/after on the standard bench matrix; report the dispatch-count reduction per token.
- Keep the unfused path behind a flag for oracle comparisons.
