---
id: benchmark-mixed-q4-q8-quant-policy
title: Benchmark mixed q4 q8 quantization policy
status: todo
priority: p1
dependencies: [calibrate-q4-quantization-quality-gates]
related: []
scopes: [quantization, runtime/candle, evals]
shared_scopes: [docs/research]
paths: [src/main.rs, src/quant_convert.rs, src/quantized_linear.rs, evals/**, docs/research/mixed-q4-q8-quantization-policy.md]
tags: [quantization, benchmark, performance]
---
## Goal

Test a mixed policy that keeps attention and DeltaNet-sensitive tensors at q8 while using q4 for selected MLP tensors.

## Acceptance

- Add a named conversion policy for mixed q4 MLP plus q8 attention/DeltaNet coverage.
- Benchmark dense, q8 text, q4 MLP-only, q4 text-safe, and mixed q4/q8 under the same prompt matrix.
- Report decode speed, prefill speed, memory avoided, and quality-gate result side by side.
- Decide whether mixed q4/q8 should become the default performance policy.
