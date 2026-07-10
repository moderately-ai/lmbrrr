---
id: aggregate-throughput-bench-harness
title: Aggregate throughput bench harness
status: todo
priority: p2
dependencies: [batched-multi-stream-decode-runner]
related: []
scopes: [evals, runtime/candle]
shared_scopes: [docs/research]
paths: [src/main.rs, evals/**, docs/research/aggregate-throughput-benchmark.md]
tags: [measurement, batching, campaign-1000]
---
## Goal

Extend `lmbrrr bench` with an N-stream mode that measures aggregate and per-stream tok/s under the repo's measurement gate (warmups, >= 5 iterations, medians and spread), so the aggregate-1000 milestone is claimed by the standard harness, not a one-off script.

## Acceptance

- `bench --streams N` running the batched runner over the standard prompt matrix with mixed prompt lengths per batch.
- JSONL rows carry N, per-stream rates, aggregate rate, batch-fill/EOS behaviour, and the active quantization policy.
- MILESTONE GATE: >= 1000 tok/s aggregate recorded with full config disclosure in docs/research.
