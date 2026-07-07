---
id: profile-metal-decode-hot-path
title: Profile Metal decode hot path and rank bottlenecks
status: done
priority: p1
dependencies: [benchmark-runner-token-rate]
related: [port-qwen35-text-decoder]
scopes: [runtime/candle, runtime/metal, docs/research]
shared_scopes: []
paths: [src/main.rs, src/qwen35.rs, docs/research/metal-decode-hot-path-profile.md, docs/research/benchmark-runner.md]
tags: [profiling, performance, metal]
---
## Goal
Identify the real decode bottlenecks before choosing optimization work.

## Acceptance
- Profile text-only decode on Metal using the benchmark harness.
- Attribute time to sampling/CPU transfers, DeltaNet recurrent path, full attention layers, KV cache work, matmuls, and kernel launch overhead where possible.
- Produce a ranked optimization backlog with measured evidence in docs/research.
- Include one recommended first optimization and one thing that looked tempting but was not yet justified by measurements.
