---
id: port-qwen35-text-decoder
title: Port Qwen3.5 text decoder path
status: in-progress
priority: p1
dependencies: [audit-candle-support, scaffold-baseline-harness]
related: []
scopes: [model/qwen, runtime/candle]
shared_scopes: []
paths: []
tags: [implementation, qwen]
---
## Goal

Implement or prototype the Qwen3.5-style text decoder path needed by MiniCPM-V-4.6.

## Work

- Map HF module names and tensor names to Candle modules.
- Implement the hybrid linear-attention/full-attention layer sequence and cache/state handling.
- Validate short-prompt logits against Transformers before performance optimization.
- Expose hidden states needed later by MTP or speculative decoding experiments.

## Acceptance

- The text decoder can load compatible weights or fixtures and produce matching logits on short validation prompts.
- Known accuracy/performance gaps are documented.
