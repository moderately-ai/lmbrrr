---
id: calibrate-q4-quantization-quality-gates
title: Calibrate q4 quantization quality gates
status: done
priority: p1
dependencies: [benchmark-real-quantized-inference]
related: []
scopes: [quantization, evals, runtime/candle, coordination]
shared_scopes: [docs/research]
paths: [src/main.rs, src/quant_convert.rs, src/quantized_linear.rs, evals/**, docs/research/q4-quantization-quality-gates.md]
tags: [quantization, evals, quality]
---
## Goal

Add generation-level quality checks for q4 MiniCPM quantization policies so speed gains are not accepted on top-1 fixture parity alone.

## Acceptance

- Add a reproducible text generation eval matrix that compares dense, q8, q4 MLP-only, and q4 text-safe outputs.
- Report exact-match/prefix, token divergence point, length, and simple lexical similarity metrics per prompt.
- Document pass/fail gates for q4 policies before broadening quantization coverage.
- Keep existing logits parity and quantized inference smoke checks passing.
