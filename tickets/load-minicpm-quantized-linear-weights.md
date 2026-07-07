---
id: load-minicpm-quantized-linear-weights
title: Load MiniCPM quantized linear weights
status: done
priority: p2
dependencies: [convert-minicpm-mixed-precision-weights]
related: []
scopes: [runtime/candle, quantization]
shared_scopes: []
paths: [src/main.rs, src/lib.rs, src/qwen35.rs, src/minicpm.rs, src/quantized_linear.rs, docs/research/quantized-linear-loader.md]
tags: [quantization, runtime, implementation]
---
Teach the Candle runner to load quantized MiniCPM linear weights beside BF16 tensors.

Acceptance:

- Add a quantized-linear abstraction that can hold BF16 `Linear` or Candle `QMatMul`.
- Load mixed-precision artifacts according to the conversion manifest.
- Keep embeddings, norms, vision, merger, and DeltaNet state tensors on the protected BF16/F32 path initially.
- Run logits parity and benchmark commands against at least one Q8 text-linear artifact.
