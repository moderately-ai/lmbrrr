---
id: optimize-deltanet-chunked-prefill-and-verify-throughput
title: Optimize DeltaNet chunked prefill and verify throughput
status: todo
priority: p1
dependencies: [profile-dspark-verification-throughput-table]
related: []
scopes: [runtime/candle, runtime/metal, inference/speculative]
shared_scopes: [docs/research]
paths: []
tags: [dspark, performance, kernels]
---
## Goal

Remove the two sequential-loop bottlenecks in `GatedDeltaNet` for seq_len > 1 so that verifying a gamma-token DSpark block approaches one-forward-pass cost instead of gamma sequential recurrent steps per layer.

## Status: campaign critical path (measured 2026-07-10)

The verification throughput table (docs/research/dspark-verification-throughput-table.md) measured T_verify(gamma) ~= 11 ms + 6.3 ms per token: the marginal verify token costs 37% of a full decode step, capping chain-DSpark at ~72 tok/s even at tau = 5. No speculation configuration pays until this ticket lands. Success target: marginal verify token <= 1.5 ms (per-token efficiency >= 8x at gamma = 16), re-measured via `verify-table`.

## Context (from full-source audit, 2026-07-10)

- `recurrent_delta_rule` (src/qwen35.rs) processes seq_len > 1 with a per-token Rust loop issuing ~8 small tensor ops per token per DeltaNet layer. Prefill and speculative verify chunks pay this on every round.
- `depthwise_causal_conv` (src/qwen35.rs) is an O(seq_len * kernel) loop of narrow/broadcast_mul/add ops with a per-position `Tensor::zeros` allocation, instead of a single grouped conv1d.
- A prior conv shortcut was rejected because BF16 reduction-order drift changed generated tokens (docs/research/deltanet-decode-optimization.md); any replacement must be gated the same way.

## Acceptance

- Replace the seq_len > 1 depthwise conv loop with a grouped `conv1d` (or a fork kernel) that preserves the decode sample and strict logits parity.
- Implement a chunked/parallel delta-rule path for seq_len > 1 (chunkwise GLA/DeltaNet formulation), in Candle ops first; if Metal dispatch overhead still dominates, add a fused Metal kernel in the candle fork (~/workspace/github.com/huggingface/candle) and pin lmbrrr to the fork rev.
- Gate every variant with `logits --fail-on-mismatch` plus a long-generation exact-greedy oracle; document any accepted arithmetic-order deltas rather than hiding them behind tolerances.
- Measure verify-chunk tokens/sec for gamma in {2, 4, 8, 16} before/after against the DSpark verification throughput table, and update the table artifact the scheduler consumes.
