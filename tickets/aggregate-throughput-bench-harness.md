---
id: aggregate-throughput-bench-harness
title: Aggregate throughput bench harness
status: done
priority: p2
dependencies: [batched-multi-stream-decode-runner]
related: []
scopes: [evals, runtime/candle]
shared_scopes: [docs/research]
paths: [src/main.rs, evals/**, docs/research/aggregate-throughput-benchmark.md]
tags: [measurement, batching, campaign-1000]
---
## Outcome (2026-07-11, commit 853a542 — MILESTONE ACHIEVED)

multi-bench --streams sweep mode (one model load, per-N rows). BF16 N=8/16/24/32 = 564/915/1242/1530 tok/s aggregate: the campaign Stage-4 gate (>= 1000 aggregate) clears at N=24 and reaches 1530 at N=32 — BF16, static batching, no speculation. q4k saturates ~845 (quantized mm does not batch; BF16 tile gemm works at m >= 16). Output integrity verified per-N (exact token counts, coherent text; one early tie-flip on stream 0, advisory).

## Goal

Extend `lmbrrr bench` with an N-stream mode that measures aggregate and per-stream tok/s under the repo's measurement gate (warmups, >= 5 iterations, medians and spread), so the aggregate-1000 milestone is claimed by the standard harness, not a one-off script.

## Acceptance

- `bench --streams N` running the batched runner over the standard prompt matrix with mixed prompt lengths per batch.
- JSONL rows carry N, per-stream rates, aggregate rate, batch-fill/EOS behaviour, and the active quantization policy.
- MILESTONE GATE: >= 1000 tok/s aggregate recorded with full config disclosure in docs/research.
