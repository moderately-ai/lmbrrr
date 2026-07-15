---
id: spike-ternary-type42-block-format
title: "SPIKE: reverse-engineer the prism-ml ternary ggml type-42 block layout"
status: done
priority: p1
dependencies: []
related: [ternary-bonsai-27b-support, ternary-type42-dequant]
scopes: [candle-fork]
shared_scopes: [docs/research]
paths: [docs/research/ternary-type42-format.md]
tags: [ternary-bonsai, model-compat]
---
## WHY

Blocks the whole ternary track. `Ternary-Bonsai-27B-Q2_0.gguf` stores 498 weight matrices as custom ggml type `42` (`file_type 41`), with no upstream reference (mainline llama.cpp/candle/gguf-python types stop ~39). Nothing can dequantize or matmul these weights until the block struct is known: block element count, bit packing of the ternary trits (repo tags say ~2 bpw), and how the per-block scale(s) are stored (fp16? per-group?). There is also a sibling `PQ2_0` (7.17 GB, likely a repacked layout) and `Q2_g64` (group-64) — determine whether these are distinct types or packings.

## WORK ITEMS

1. **Route A (preferred): find prism-ml's llama.cpp fork / quantizer source.** `library_name: llama.cpp` + `metal`/`cuda` tags imply a published fork defining the type-42 `block_*` struct and its `dequantize_row_*` / `vec_dot_*`. Locate it (repo README/NOTICE/LICENSE, prism-ml org), read the block struct + quant/dequant IN FULL. Cheapest ground truth.
2. **Route B (fallback): raw-byte reverse-engineering.** Range-fetch one type-42 tensor's data region (offset from the header tensor-info) for a known-shape weight (e.g. `output_norm`-adjacent small matrix), and infer block size + scale layout from the byte pattern, cross-checked against the F16 reference `Ternary-Bonsai-27B-F16.gguf` for the same tensor (dequant must reproduce F16 within ternary error).
3. Write a **Python reference dequant** for type 42 and validate: dequant(type42 tensor) ≈ corresponding F16 tensor (per-tensor cosine / max-abs error within expected ternary rounding).
4. Characterize `PQ2_0` vs `Q2_0` vs `Q2_g64` (same type code or distinct?).

## DELIVERABLE

`docs/research/ternary-type42-format.md`: the block struct (bytes/block, trits/block, scale encoding), a validated Python reference dequant, and the code-map to prism-ml's fork (if found). Unblocks [[ternary-type42-dequant]] and [[metal-ternary-matmul-kernel]].

## DONE-WHEN

We can dequantize an arbitrary type-42 tensor to bf16 and match the F16 reference within ternary rounding error, with the byte layout documented.

## RESOLVED (2026-07-15) — format fully determined; see `docs/research/ternary-type42-format.md`

Type 42 = **`Q2_0` at QK=128** (README "Q2_0_g128"). Ground truth from prism-ml's now-cloned llama.cpp fork (`~/workspace/github.com/PrismML-Eng/llama.cpp`, branch `prism`) + confirmed against the file's own byte arithmetic (every type-42 tensor is exactly 34.00 B/128 elements):
- `block_q2_0 { ggml_half d; uint8_t qs[32]; }` = 34 B / 128 weights (2.125 bpw deployed, 1.71 ideal). `GGML_TYPE_Q2_0 = 42`.
- Code→value `00→−1 01→0 10→+1 11→+2`, `w=(q−1)·d`, LSB-first (weight j at byte j/4, bit (j%4)*2). Python reference dequant written (in the doc).
- Reference **Metal** kernel located: `ggml-metal.metal` `kernel_mul_mv_q2_0_f32_impl` (nr1 multi-column weight-reuse = the verify path) → the direct template for [[metal-ternary-matmul-kernel]]. MLX fork uses a *different* affine-2-bit scheme (weaker ref).
- `PQ2_0`/`Q2_g64` characterized (g128 repack / group-64).

VALIDATED (2026-07-15): numerical cross-check done — range-fetched `blk.0.ssm_alpha.weight` from both `Q2_0` and `F16` and dequantized. **cosine(dequant_Q2, F16) = 1.00000**, value-for-value ({−0.0137, 0, +0.0137} = {−1,0,+1}·d). The reference dequant is exact. (Aside: the F16 tensor is itself perfectly ternary → Bonsai is ternary-native, F16 is just a wide container.) Spike closed; [[ternary-type42-dequant]] + [[metal-ternary-matmul-kernel]] fully unblocked with a certain, validated format.
