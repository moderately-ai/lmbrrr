---
id: benchmark-metal-quantized-matmul-kernels
title: Benchmark Metal quantized matmul kernels
status: done
priority: p2
dependencies: [load-minicpm-quantized-linear-weights]
related: []
scopes: [runtime/metal, quantization]
shared_scopes: [docs/research]
paths: [src/main.rs, docs/research/metal-quantized-matmul-benchmark.md]
tags: [quantization, metal, benchmark]
---
Benchmark Candle's existing Metal quantized matmul paths on MiniCPM/Qwen3.5 shapes before writing custom kernels.

Acceptance:

- Measure decode MV and prefill/chunk MM separately for Q8, Q4K, and high-bit dynamic candidates.
- Include activation dtype/cast costs, especially BF16/F16 versus F32 behavior.
- Compare against BF16 baseline linear matmuls for representative DeltaNet, MLP, full-attention, and LM-head shapes.
- Document whether custom BF16/F16 activation x quantized-weight Metal kernels are justified.
