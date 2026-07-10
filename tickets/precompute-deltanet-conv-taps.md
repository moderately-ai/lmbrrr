---
id: precompute-deltanet-conv-taps
title: Precompute depthwise conv tap tensors instead of narrow+reshape per token
status: done
priority: p2
dependencies: []
related: []
scopes: [runtime/candle]
shared_scopes: []
paths: []
tags: [performance, decode-audit-2026-07-10]
---
## Outcome (2026-07-10)

Landed in lmbrrr 4c03195: taps precomputed as (1, conv_dim, 1) tensors in GatedDeltaNet::new, per-step squeeze/narrow/reshape removed. Parity bit-identical. Bundled with the sync-removal A/B (~+1% combined).

## Goal

src/qwen35.rs:856-861: `weight.narrow(1, k, 1)?.reshape((1, c, 1))` — the narrow of the (6144,4) squeezed weight is non-contiguous, so the reshape runs a copy kernel. 4 copies + 4 allocs per layer per token = 72 dispatches/token producing byte-identical constants. `weight.squeeze(1)` at line 836 also recomputes a view each call.

## Acceptance

- Precompute the 4 `(1, conv_dim, 1)` tap tensors in `GatedDeltaNet::new`; reuse per step.
- Strict logits parity (bit-identical expected); folded dispatch-count delta noted in docs/research.
- Superseded by the full fused decode kernel when it lands, but worth taking now — trivial and immediate.
