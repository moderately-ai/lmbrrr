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
## Expanded scope: the L2 verify-residuals bundle (spec-loop analysis)

The l>=2 chunk verify costs ~15.6 ms vs a 6.9 ms decode token; itemized cuts beyond the repeat_kv fix below: (1) instrumentation-only syncs at main.rs:4722 (post-draft), 4750 (post-verify), and the unconditional rollback sync at 4830 — production round needs exactly 2 readbacks and zero bare syncs; gate timing syncs behind a flag (-1.5-2.5 ms). (2) One committed buffer per round: enqueue verify + lm_head + argmax + capture-cat + SPECULATIVE full-l drafter ctx append (chunk captures are prefix-valid regardless of accepted; truncate the drafter ctx after readback) (-1-1.5 ms host gaps). (3) Device-resident acceptance: eq+cumprod on GPU, one packed [accepted, targets...] readback folding the argmax sync into the verify readback (-0.5-1 ms). (4) Rollback reconstruction is 2.15 ms per occurrence host-composed — drop its sync, consider a small fused select-state kernel. Target: mean verify (incl. width-0 rounds) ~9.3 ms.

## Goal

SDPA landed only for l == 1 && mask.is_none() (src/qwen35.rs:550); every spec verify chunk (l <= 9, masked) still materializes repeat_kv(k)/repeat_kv(v) .contiguous() over the WHOLE cache plus the k_t transpose copy, per full-attention layer per round (src/qwen35.rs:567-571) — cost grows with context. Route the masked l > 1 path through SDPA-full with the causal mask (the drafter side already runs SDPA-full unmasked, commit 5218684 — pattern in-repo; check the sdpa mask shape contract (bs, qhead, seq, kv_seq) vs our (1,1,l,total) broadcast mask).

## Acceptance

- Masked chunk path through sdpa (or a documented reason it cannot take the mask shape + the fallback chosen).
- Gates: 33/33 tests, fixture, state-integrity oracle both prompts. NOTE: rederive the trajectory-oracle noise bound if it creeps past 0.75 — envelope is already at 0.5 with stacked numerics changes.
- In-loop verify ms/round before/after into the cost-model artifact.
