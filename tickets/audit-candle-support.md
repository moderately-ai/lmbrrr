---
id: audit-candle-support
title: Audit Candle support gaps for MiniCPM and Qwen3.5
status: done
priority: p1
dependencies: [research-minicpm-surface]
related: []
scopes: [docs/research, runtime/candle]
shared_scopes: []
paths: []
tags: [research, candle]
---
## Goal

Compare MiniCPM-V-4.6 and Qwen3.5 requirements against Candle's current model, cache, quantization, and Metal support.

## Work

- Inspect Candle examples and model implementations relevant to Qwen, SigLIP, linear attention, MTP, and quantization.
- Identify missing operators, state/cache abstractions, tokenizer/processor needs, and Metal kernel gaps.
- Separate upstreamable Candle work from lmbrrr-specific experiment code.

## Acceptance

- A support-gap note lists existing Candle pieces, missing pieces, likely implementation files, and risk areas.
- Follow-up tickets can be created for concrete ports or kernels.
