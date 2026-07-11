---
id: verify-chunk-masked-sdpa
title: Route the masked verify-chunk attention through SDPA (kill per-chunk repeat_kv)
status: todo
priority: p1
dependencies: []
related: [remeasure-spec-round-cost-model]
scopes: [runtime/candle]
shared_scopes: [docs/research]
paths: []
tags: [speculative, performance, kernels]
---
## Goal

SDPA landed only for l == 1 && mask.is_none() (src/qwen35.rs:550); every spec verify chunk (l <= 9, masked) still materializes repeat_kv(k)/repeat_kv(v) .contiguous() over the WHOLE cache plus the k_t transpose copy, per full-attention layer per round (src/qwen35.rs:567-571) — cost grows with context. Route the masked l > 1 path through SDPA-full with the causal mask (the drafter side already runs SDPA-full unmasked, commit 5218684 — pattern in-repo; check the sdpa mask shape contract (bs, qhead, seq, kv_seq) vs our (1,1,l,total) broadcast mask).

## Acceptance

- Masked chunk path through sdpa (or a documented reason it cannot take the mask shape + the fallback chosen).
- Gates: 33/33 tests, fixture, state-integrity oracle both prompts. NOTE: rederive the trajectory-oracle noise bound if it creeps past 0.75 — envelope is already at 0.5 with stacked numerics changes.
- In-loop verify ms/round before/after into the cost-model artifact.
