---
id: design-speculative-decoding-lab
title: Design MTP EAGLE DFlash DSpark experiment lab
status: todo
priority: p2
dependencies: [research-minicpm-surface, define-experiment-pivot-gates]
related: []
scopes: [docs/research, inference/speculative]
shared_scopes: []
paths: []
tags: [research, speculative]
---
## Goal

Define a staged speculative-decoding research plan for MiniCPM/Qwen on local Apple Silicon.

## Work

- Compare built-in MTP-1, EAGLE-style heads, DFlash, and DSpark requirements.
- Define accepted-length, verifier-waste, draft-latency, and quality metrics.
- Separate single-user local generation experiments from high-concurrency serving scheduler experiments.
- Identify what hidden states, logits, caches, and batching support the runtime must expose.

## Acceptance

- A speculative-decoding design note ranks the approaches by implementation cost and expected value.
- Follow-up implementation tickets can be created for MTP verification, EAGLE prototype, and DSpark scheduler experiments.
