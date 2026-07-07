---
id: optimize-deltanet-metal-decode
title: Optimize Qwen3.5 DeltaNet decode on Metal
status: in-progress
priority: p1
dependencies: [build-transformers-parity-oracle, profile-metal-decode-hot-path]
related: [port-qwen35-text-decoder]
scopes: [runtime/candle, runtime/metal, model/qwen]
shared_scopes: []
paths: [src/qwen35.rs, docs/research/deltanet-decode-optimization.md]
tags: [performance, metal, qwen, deltanet]
---
## Goal
Replace correctness-first DeltaNet decode code with a faster implementation once parity and profiling show it is worth doing.

## Acceptance
- Preserve oracle parity on covered prompts.
- Reduce per-token decode latency on the measured hot path.
- Prefer grouped Candle ops first when sufficient; use a custom Metal kernel only when the op graph or launch count makes that necessary.
- Document the state layout and any numerical tradeoffs.
