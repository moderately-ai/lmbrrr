---
id: validate-minicpm-image-parity
title: Validate MiniCPM image processor and vision parity
status: todo
priority: p1
dependencies: [build-transformers-parity-oracle]
related: [port-minicpm-v46-full-path]
scopes: [evals, model/vision, model/minicpm, docs/research]
shared_scopes: []
paths: []
tags: [parity, vision, minicpm]
---
## Goal
Verify that image preprocessing, patch packing, placeholder expansion, and vision/text handoff match upstream MiniCPM-V-4.6 closely enough for experiments.

## Acceptance
- Compare processor outputs against Transformers for representative image sizes and aspect ratios.
- Validate image feature counts and text placeholder positions.
- Compare at least one multimodal next-token result or hidden-state checkpoint with documented tolerance.
- Record unresolved differences before using multimodal results as experimental evidence.
