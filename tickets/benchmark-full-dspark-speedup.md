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
- Require a material speedup gate before claiming success: at least 10% over greedy for prototype value, and preferably 20%+ before architecture direction changes.
- Document failure modes if DSpark does not beat EAGLE on this hardware/model pair.
