---
id: spike-ternary-type42-block-format
title: "SPIKE: reverse-engineer the prism-ml ternary ggml type-42 block layout"
status: todo
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
