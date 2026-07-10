---
id: implement-speculative-state-rollback
title: Implement speculative state rollback and multi-round loop
status: done
priority: p1
dependencies: []
related: [integrate-dspark-block-runner, optimize-deltanet-chunked-prefill-and-verify-throughput]
scopes: [runtime/candle, inference/speculative]
shared_scopes: [docs/research]
paths: [src/qwen35.rs, src/main.rs, docs/research/speculative-state-rollback.md]
tags: [speculative, dspark, campaign-1000]
---
## Goal

The multi-round speculative decoding loop with full state rollback, proven with a stub drafter (pre-computed greedy tokens with injected corruptions) so the machinery is verified before any trained DSpark checkpoint exists. Split out of integrate-dspark-block-runner, which retains drafter loading/inference and gains this ticket as a dependency.

## Context

The audit confirmed GatedDeltaNet mutates conv_state/recurrent_state by assignment (snapshot = cheap Arc clone; old tensors are never mutated in place) while the candle KvCache slice_sets into a preallocated buffer with no truncate — and candle's KvCache preallocates max_position_embeddings (262k) per layer, which is also a memory problem. Chunk semantics follow DeepSpec: verify chunks are [anchor, d1..dgamma] (the anchor token is fed; its logits verify d1), the bonus token becomes the next anchor.

## Acceptance

- Replace candle KvCache in FullAttention with an in-repo truncatable cache (grow-on-demand capacity, slice_set append, truncate(len) rewinds); strict logits parity must be unchanged.
- Qwen35TextModel::snapshot_decode_state / restore_decode_state covering per-layer DeltaNet conv/recurrent tensors (Arc clone) and KV lengths.
- Multi-round loop in a `dspark-run` subcommand (stub mode): prefill once, then rounds of stub-propose -> verify chunk [anchor, drafts] -> exact-match accept -> restore + re-advance accepted prefix on partial accept (fast path keeps state on full accept) -> bonus becomes next anchor; no prompt re-prefill ever.
- Stub drafter takes the precomputed greedy continuation with `--stub-corrupt-every N` corrupting every Nth draft token to force rejections at controlled positions.
- BLOCKING oracle: with corruptions injected, the multi-round output must exactly equal plain greedy generation over >= 128 tokens on multiple prompts.
- Report per-round accepted lengths, verify/rollback/re-advance timings, and effective tok/s vs the greedy baseline.
