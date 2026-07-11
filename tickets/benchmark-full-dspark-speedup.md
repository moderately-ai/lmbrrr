---
id: benchmark-full-dspark-speedup
title: Benchmark full DSpark speedup
status: todo
priority: p1
dependencies: [integrate-dspark-block-runner, implement-dspark-hardware-aware-prefix-scheduler]
related: []
scopes: [inference/speculative, runtime/candle, evals]
shared_scopes: [docs/research]
paths: [src/main.rs, evals/dspark/**, docs/research/full-dspark-speedup-benchmark.md]
tags: [speculative, dspark, benchmark, performance]
---
## Goal

Measure end-to-end DSpark speedup with the full drafter plus calibrated hardware-aware scheduling, and compare it against greedy, recurrent EAGLE, and fixed-length verification.

## Acceptance

- Run matched greedy, recurrent EAGLE, DSpark fixed-length, and DSpark scheduled benchmarks on the same prompt matrix, dtype, device, and max-token caps.
- Report median and spread over multiple iterations for prefill, draft, verify, decode/output token rate, accepted length, verifier waste, and exact greedy reconstruction.
- Demonstrate whether DSpark beats recurrent EAGLE locally and identify prompt classes where it does or does not.
- Require a material speedup gate before claiming success: beat the 136-146 tok/s fused greedy baseline outright; 10%+ over it for prototype value, 20%+ before architecture direction changes.
- Measured context (REWRITTEN 2026-07-10 evening; the old bullet cited pre-fusion ghosts): greedy is 136-146 tok/s (6.9 ms/token); the spec round is propose 8.7 + verify 13.2 + rollback <1 ms at tau~2.1 -> 87 math / 77 tides. Break-even is no longer a single tau number: it is tokens_per_round / round_ms > 0.146, recomputed from remeasure-spec-round-cost-model's table. The perfect-drafter ceiling must be re-derived from that table before quoting any ceiling.
- Document failure modes if DSpark does not beat EAGLE on this hardware/model pair.
