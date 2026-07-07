---
id: benchmark-real-quantized-inference
title: Benchmark real quantized MiniCPM inference
status: done
priority: p1
dependencies: [store-quantized-weights-as-qtensor]
related: []
scopes: [runtime/candle, quantization, evals]
shared_scopes: []
paths: [src/main.rs, src/quantized_linear.rs, docs/research/real-quantized-inference-benchmark.md]
tags: [quantization, benchmark, performance]
---
## Goal

Measure dense BF16/F16 versus real quantized MiniCPM text inference on this hardware.

## Acceptance

- Run matched text prompts for dense, q8, and any working q4/q5 policy.
- Report load time, prefill tok/s, decode tok/s, memory/weight footprint, and output sanity.
- Record whether quantized matmul speedups survive full-model overheads.
- Produce a decision note for the next kernel or policy change.
