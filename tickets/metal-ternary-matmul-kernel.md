---
id: metal-ternary-matmul-kernel
title: "FEATURE: Metal ternary matmul kernel (ternary weight x activation, BitNet-style)"
status: todo
priority: p2
dependencies: [spike-ternary-type42-block-format, ternary-type42-dequant]
related: [ternary-bonsai-27b-support, eval-matmul2d-uint4b-tensor-op, bf16-activation-quantized-matmul-metal]
scopes: [candle-fork]
shared_scopes: []
paths: []
tags: [ternary-bonsai, model-compat, fork]
---
## WHY

The point of a ternary target is the byte diet: ~2 bpw weights vs q4k's ~4.5 → roughly half the decode bandwidth of our current MiniCPM stack, on a 27B model where bandwidth is the wall. Dequant-to-bf16-then-GEMV throws that away (materializes bf16, moves 8x the bytes). The win needs a kernel that consumes the packed ternary weights directly — a ternary GEMV/GEMM (weight ∈ {-1,0,+1} × bf16 activation, or int8-activation BitNet-style) — analogous to our existing `mm2d` uint4b tensor-op path, but for type 42.

## WORK ITEMS

1. Design the ternary GEMV: unpack trits in-kernel, accumulate `sum(sign * act)` per output with the per-block scale, threadgroup-staged like the mm2d/`mul_mv` path. Decide activation precision (bf16 direct vs int8-quantized activations for a true BitNet int matmul) — measure both.
2. Route it at the decode shapes of the 27B (hidden 5120, ffn 17408, qkv/gate) — the body/head split like the existing routing table.
3. Correctness gate: kernel output == the [[ternary-type42-dequant]] reference within the stub-oracle noise bound (reuse the mm2d oracle harness).
4. Bench vs dequant-to-bf16-GEMV and vs q4k on the same shapes; confirm the bandwidth win is realized in-loop (not just isolated latency — the split-K/t32 lesson: in-loop-arbitrated).

## DONE-WHEN

Ternary decode runs on Metal consuming packed type-42 weights, bit-close to the dequant reference, and beats dequant-to-bf16 on measured decode bandwidth for the 27B shapes. Candidate for upstreaming alongside the other fork kernels ([[upstream-fork-kernels]]).
