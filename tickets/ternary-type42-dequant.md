---
id: ternary-type42-dequant
title: "FEATURE: ternary type-42 dequantization in the candle fork (QTensor path)"
status: todo
priority: p2
dependencies: [spike-ternary-type42-block-format]
related: [ternary-bonsai-27b-support, metal-ternary-matmul-kernel, gguf-loader-qwen35-hybrid]
scopes: [candle-fork]
shared_scopes: []
paths: []
tags: [ternary-bonsai, model-compat, fork]
---
## WHY

Once the type-42 block layout is known ([[spike-ternary-type42-block-format]]), the fork must be able to READ it: register the ternary type so `gguf_file`/`ggml_file` ingest doesn't reject code 42, and provide a correct CPU dequant (type42 → f32/bf16). This is the reference path that gates the loader and is the correctness oracle for the Metal kernel.

## WORK ITEMS

1. Add the ternary type to the fork's `GgmlDType` (or a fork-local extension) with its block size / type-size, so `qtensor_from_ggml` and the GGUF reader accept it. Fail loudly on unknown variants.
2. Implement `dequantize` (block → f32/bf16) matching the Python reference from the spike bit-for-bit (or within documented rounding).
3. A fork test: dequant a fixture type-42 block == reference vector; and a whole-tensor test vs the F16 reference for one Bonsai weight.
4. Decide the QTensor representation: keep ternary packed (for the Metal kernel to consume directly) vs eager-dequant-to-bf16 at load. Packed is required for the memory/bandwidth win; expose both so the loader can start on dequant-to-bf16 while the kernel lands.

## DONE-WHEN

The fork loads a type-42 GGUF tensor and produces a bf16 tensor matching the F16 reference within ternary error; `cargo nextest` fixture + whole-tensor tests pass. Feeds [[gguf-loader-qwen35-hybrid]] (functional path) and [[metal-ternary-matmul-kernel]] (oracle).
