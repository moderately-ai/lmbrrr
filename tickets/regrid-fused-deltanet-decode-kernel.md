---
id: regrid-fused-deltanet-decode-kernel
title: Re-grid fused DeltaNet decode kernel to full-occupancy geometry
status: in-progress
priority: p1
dependencies: [ngram-draft-source-mux]
related: []
scopes: [runtime/metal, candle-fork]
shared_scopes: [docs/research]
paths: []
tags: [kernels, campaign-1000, frontier-survey]
---
## Goal
The fused GatedDeltaNet decode kernel launches 16 threadgroups (2,048 threads) on a 32-core GPU: half the machine idle, state slab read twice, ~19 GB/s effective, ~2.8 ms of the 4.25 ms token. Re-grid to MLX gated_delta.py geometry: grid (32, dv, B*heads) = 512 threadgroups, TG (32,4,1), 32 threads simd-cooperate over dk per (head, dv-column), state read once into registers, fp32 state/accumulate throughout.

## Acceptance
- VERIFY the agent diagnosis by reading the kernel in full before changing it (metal_src/gated_delta.metal, decode variant).
- Rewritten kernel passes the existing parity gates + decode-level state-integrity oracle; batched (aggregate) variant preserved.
- Measured end-to-end per protocol: expect deltanet block 2.8 -> 0.3-0.5 ms, single-stream 235 -> ~520-600 tok/s; report actuals vs the ~830 roofline.
- Reference: mlx-lm gated_delta.py; ollama #15865 (bf16-state bug precedent -> keep fp32 state).
