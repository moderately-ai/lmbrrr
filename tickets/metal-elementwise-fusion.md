---
id: metal-elementwise-fusion
title: Elementwise fusion of the norm/add/silu confetti (post wave-1/2)
status: todo
priority: p3
dependencies: [metal-wave2-host-encode-rlist]
related: []
scopes: [candle-fork, runtime/candle]
shared_scopes: [docs/research]
paths: []
tags: [kernels, campaign-1000]
---
Census (dossier §5-final): per decode step the elementwise soup is rmsnorm 61 + badd 48 + bmul 30 + silu 24 + sigmoid 6 + rope 12 ≈ 180 dispatches at 1-20μs each; every one costs a barrier drain + ramp at <17% occupancy (the Active-Cores comb). Global barriers provably serialize everything (barrier-probe ratios 0.85-1.0), so the only comb fix is FEWER, FATTER kernels.

Candidates, by adjacency: fused SwiGLU (silu·mul, 24+24→24... covers MLP), residual-add+rmsnorm fused (48+48 pairs), attention gate sigmoid·mul. Upstream direction: #3467 lazy backend (op-graph capture for automatic fusion — watch it; don't block on it). RE-MEASURE FIRST after waves 1-2 land: ~100 casts and ~78 copies disappear there, which changes which fusions still matter. Procedure: dossier §7.
