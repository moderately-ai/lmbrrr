---
id: convert-minicpm-mixed-precision-weights
title: Convert MiniCPM mixed precision weights
status: todo
priority: p2
dependencies: [score-minicpm-quantization-sensitivity]
related: []
scopes: [quantization, runtime/candle]
shared_scopes: [docs/research]
paths: []
tags: [quantization, conversion, implementation]
---
Convert native MiniCPM-V-4.6 safetensors into a mixed-precision artifact using a policy manifest.

Acceptance:

- Start with Q8 and Q4K text linear policies from `docs/research/dynamic-quantization-lab.md`.
- Emit a manifest recording source checkpoint, quantization format per tensor, protected tensors, and expected weight bytes.
- Preserve BF16/F32 protected modules exactly.
- Keep conversion deterministic and reproducible from the sensitivity artifact.
