---
id: benchmark-runner-token-rate
title: Add reproducible token-rate benchmark harness
status: done
priority: p1
dependencies: [scaffold-baseline-harness]
related: [implement-naive-text-inference-runner]
scopes: [runtime/candle, evals]
shared_scopes: []
paths: []
tags: [benchmark, performance, metal]
---
## Goal
Measure the runner in a way that separates user-visible streaming behavior from raw decode throughput.

## Acceptance
- Add a benchmark path or command that records prefill time, TTFT, decode tok/s, total tokens, prompt tokens, device, dtype, model revision, and generation settings.
- Include warmup, greedy no-stream mode, and JSONL output suitable for before/after comparisons.
- Provide at least three prompt profiles: short, medium, and longer reasoning trace.
- Document release/Metal invocation and how to compare runs.
