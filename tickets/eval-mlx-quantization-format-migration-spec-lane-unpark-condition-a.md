---
id: eval-mlx-quantization-format-migration-spec-lane-unpark-condition-a
title: "eval: MLX quantization format migration (spec-lane unpark condition A)"
status: todo
priority: p1
dependencies: []
related: []
scopes: [evals, candle-fork]
shared_scopes: []
paths: [docs/performance.md]
tags: [eval-wave, kernels, mlx]
---

WHY (condition A): the open question is whether MLX's *format* (affine-2bit, group_size 128, ~2.19 bpw, unpacked fp16 scale+bias) is worth migrating lmbrrr's Q2_0 ternary onto. This is the spec-lane unpark gate: if MLX's e2e decode is materially faster at comparable bpw/quality, a format migration (requantize-at-load into MLX affine-2b) becomes the highest-leverage decode lever, ahead of more Q2_0 kernel tuning.

CONDITION-A EVIDENCE (measured 2026-07-18, e2e decode, greedy, 96 tok, same Ternary-Bonsai-27B base, both hosts — full table + caveats in docs/performance.md#reference-engine-comparison): **prism MLX fork (affine-2bit) is the raw-decode throughput leader on BOTH hosts** — 16.8 tok/s M3 / 40.3 M4, vs lmbrrr Q2_0 14.5 / 33.1 (MLX +16% M3, +22% M4) and the prism llama.cpp Q2_0 fork 13.3 / 28.2. lmbrrr's *speculative* path edges MLX decode only on the bandwidth-starved M3 (17.4 vs 16.8) and loses on the M4. Kernel-level context (does not contradict): lmbrrr's mm2d *verify* GEMM (m=8) still beats MLX qmm — the MLX win is on the m=1 *decode* qmv, a different kernel (see [[eval-apples-to-apples-qmv]], closed).

⇒ **Condition A is MET**: MLX's format+kernel unit is faster e2e. This unparks the migration lane. But the decision is NOT clean — the e2e gap conflates two effects and BOTH must be separated before committing:
1. **Format vs bpw**: MLX carries ~3% more bits (2.19 vs 2.125) and a 4-level codebook vs 3-level ternary. Part of the speed is a richer operating point, not a faster engine. The clean apples-to-apples row (lmbrrr vs llama.cpp, identical Q2_0) has lmbrrr winning — so the ternary *kernel* is not the loser; the affine *format* is the lever.
2. **Quality unmeasured**: no cross-engine PPL. A migration to affine-2b is NOT bit-preserving (ternary→4-level affine, group 128) → needs a quant-quality ladder + margin/quality gates before it can replace Q2_0.

NEXT (if this lane is pursued): (a) requant-at-load prototype into MLX affine-2b within lmbrrr/candle (or measure MLX-lm affine-2b PPL vs lmbrrr Q2_0 greedy on the quality battery); (b) confirm the format gap survives at matched bpw (build an affine-2b at ~2.125 bpw, group 128→wider, and re-time); (c) if quality holds and the gap survives bpw-matching, promote format migration as the primary decode lever. MLX spec (`spec_decode_verify` in the prism fork) is also unmeasured and would likely extend MLX's lead — measure before concluding lmbrrr's spec is competitive.
