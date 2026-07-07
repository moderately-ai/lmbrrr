---
id: optimize-generation-loop-overhead
title: Remove generation-loop overhead from token-rate measurements
status: todo
priority: p1
dependencies: [benchmark-runner-token-rate]
related: [implement-naive-text-inference-runner]
scopes: [runtime/candle]
shared_scopes: []
paths: []
tags: [performance, generation]
---
## Goal
Make sure benchmark results reflect model execution rather than avoidable CLI and sampling costs.

## Acceptance
- Add a greedy fast path that avoids unnecessary probability materialization and host transfers.
- Add buffered or disabled streaming for benchmark mode.
- Keep interactive streaming behavior available for normal runs.
- Show before/after benchmark deltas using the reproducible harness.
