---
id: validate-minicpm-image-parity
title: Validate MiniCPM image processor and vision parity
status: done
priority: p1
dependencies: [build-transformers-parity-oracle]
related: [port-minicpm-v46-full-path]
scopes: [evals, model/vision, model/minicpm, docs/research]
shared_scopes: []
paths: [src/image_processor.rs, tests/minicpm_v46_text_parity.rs, evals/minicpm_v46_image_oracle.py, evals/fixtures/minicpm_v46_transformers_image_processor.json, docs/research/minicpm-v46-image-parity.md]
tags: [parity, vision, minicpm]
---
## Goal
Verify that image preprocessing, patch packing, placeholder expansion, and vision/text handoff match upstream MiniCPM-V-4.6 closely enough for experiments.

## Acceptance
- Compare processor outputs against Transformers for representative image sizes and aspect ratios.
- Validate image feature counts and text placeholder positions.
- Compare at least one multimodal next-token result or hidden-state checkpoint with documented tolerance.
- Record unresolved differences before using multimodal results as experimental evidence.
