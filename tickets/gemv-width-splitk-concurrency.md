---
id: gemv-width-splitk-concurrency
title: GEMV width fusion, split-K, and barrier-minimal encoding
status: todo
priority: p1
dependencies: []
related: []
scopes: [runtime/metal, candle-fork, model/minicpm]
shared_scopes: []
paths: []
tags: [kernels, frontier-survey]
---
## Goal
Three residual-lane kernel items after the DeltaNet re-grid: (1) concatenate gate+up (and attention q/k/v/gate) projections - own roofline data shows 6144-row GEMV at 179 GB/s vs 3584-row at 83; (2) MLX-style qmv_split_k for skinny (<=3584-row) quantized shapes; (3) llama.cpp-style hazard-tracked concurrent encoding (barrier granularity - distinct from the falsified COMMIT granularity).

## Acceptance
- Expect +8-12% (width+split-k) and +5-8% (concurrency) on the post-regrid baseline; measure per protocol; aggregate lane re-verified.
- Also investigate: quantized aggregate 860 vs BF16 1530 - the multi-column kernels do not batch-scale (from the frontier survey); diagnose alongside split-k.
