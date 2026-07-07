---
id: store-quantized-weights-as-qtensor
title: Store MiniCPM quantized linears as Candle QTensor
status: todo
priority: p1
dependencies: [load-minicpm-quantized-linear-weights, benchmark-metal-quantized-matmul-kernels]
related: []
scopes: [runtime/candle, quantization]
shared_scopes: []
paths: [src/quantized_linear.rs, src/qwen35.rs, src/minicpm.rs, src/quant_convert.rs, docs/research/quantized-linear-loader.md]
tags: [quantization, performance, implementation]
---
## Goal

Replace the current dequantized QMatMul correctness fallback with real Candle QTensor storage for supported MiniCPM text linear weights.

## Acceptance

- q8 MiniCPM quantized artifacts load without materializing quantized tensors as dense model weights.
- The loader reports quantized tensor count, dense fallback count, and approximate memory saved.
- A short Metal logits or generation smoke still produces coherent output.
- Remaining q4k/q5k format gaps are documented.
