---
id: optimize-generation-loop-overhead
title: Remove generation-loop overhead from token-rate measurements
status: done
priority: p1
dependencies: [benchmark-runner-token-rate]
related: [implement-naive-text-inference-runner]
scopes: [runtime/candle]
shared_scopes: []
paths: [src/main.rs, docs/research/benchmark-runner.md, docs/research/generation-loop-overhead.md]
tags: [performance, generation]
---
## Goal
Make sure benchmark results reflect model execution rather than avoidable CLI and sampling costs.

## Acceptance
- Add a greedy fast path that avoids unnecessary probability materialization and host transfers.
- Add buffered or disabled streaming for benchmark mode.
- Keep interactive streaming behavior available for normal runs.
- Show before/after benchmark deltas using the reproducible harness.
