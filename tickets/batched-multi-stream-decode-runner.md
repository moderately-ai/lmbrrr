---
id: batched-multi-stream-decode-runner
title: Batched multi stream decode runner
status: todo
priority: p1
dependencies: []
related: [aggregate-throughput-bench-harness]
scopes: [runtime/candle, evals]
shared_scopes: [docs/research]
paths: [src/qwen35.rs, src/minicpm.rs, src/main.rs, evals/**, docs/research/batched-decode-runner.md]
tags: [performance, batching, campaign-1000]
---
## Goal

Run N independent generation streams through one batched decode forward. Batching amortizes exactly the per-dispatch overhead this small model suffers from and is the near-term path to the aggregate-1000 milestone (e.g. 8 streams x 125 tok/s quantized).

## Acceptance

- Batch dimension threaded through the whole text decode path: batched DeltaNet conv/recurrent states, per-stream KV via a batched cache, batched greedy sampling, per-stream EOS handling. Static batching of N in {2, 4, 8, 16}; no continuous batching needed for the milestone. The in-repo TruncatableKvCache (landed with implement-speculative-state-rollback) already carries batch dims and replaces candle's fixed-262k-preallocation cache — extend it rather than reintroducing candle_nn's.
- Remove the batch==1 assumptions found in the audit (trace recorder bail, per-row scalar readbacks).
- Left-pad or per-stream offset handling documented and oracle-checked: each stream's batched output must exactly match its single-stream greedy output (this gate blocks — batching must not change per-stream text).
- Report aggregate and per-stream tok/s vs N on the standard prompt matrix.
