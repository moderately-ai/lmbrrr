---
id: design-dynamic-quantization-lab
title: Design dynamic quantization experiment lab
status: todo
priority: p2
dependencies: [research-minicpm-surface, define-experiment-pivot-gates]
related: []
scopes: [docs/research, quantization]
shared_scopes: []
paths: []
tags: [research, quantization]
---
## Goal

Translate MLX learned quantization and Unsloth dynamic 4-bit ideas into an experiment plan for MiniCPM-V-4.6 on Candle/Metal.

## Work

- Identify candidate sensitivity metrics and calibration data.
- Define per-module precision policies for text, vision, multimodal bridge, embeddings, and MTP.
- Specify benchmark tasks that catch text and VLM regressions.
- Decide which quantized kernels are needed first.

## Acceptance

- A quantization design note proposes the first measurable variants and required Candle/kernel support.
- Follow-up implementation tickets can be created for calibration, conversion, and kernels.
