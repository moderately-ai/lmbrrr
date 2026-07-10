---
id: profile-dspark-verification-throughput-table
title: Profile DSpark verification throughput table
status: in-progress
priority: p1
dependencies: [design-full-dspark-drafter]
related: []
scopes: [runtime/candle, runtime/metal, inference/speculative]
shared_scopes: [docs/research]
paths: [src/main.rs, docs/research/dspark-verification-throughput-table.md]
tags: [speculative, dspark, benchmark]
claimed_from: todo
assignee: claude
lease_expires_at: 1783703003
---
## Goal

Measure the local target-verification throughput table DSpark needs for hardware-aware scheduling.

## Acceptance

- Benchmark target verification chunks across draft lengths and effective batch/concurrency sizes available in this runner.
- Report SPS or token/s as a function of verification token budget on Apple Metal.
- Identify throughput cliffs where verifying extra low-confidence tokens becomes harmful.
- Emit a JSON artifact that the DSpark scheduler can consume.
