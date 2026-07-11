---
id: validate-minicpm-vision-feature-parity
title: Validate MiniCPM vision feature parity
status: done
priority: p3
dependencies: [port-minicpm-v46-full-path, validate-minicpm-image-parity]
related: []
scopes: [model/minicpm, model/vision, evals]
shared_scopes: []
paths: [src/minicpm.rs, tests/minicpm_v46_text_parity.rs, evals/minicpm_v46_image_oracle.py, docs/research/minicpm-v46-image-parity.md]
tags: [parity, vision, minicpm]
---
## Goal

Compare the Candle vision tower and merged image features against Transformers for deterministic synthetic images.

## Acceptance

- Export Transformers image feature shape and sampled values after the MiniCPM image feature path.
- Add a Rust parity check for shapes and sampled values with documented tolerance.
- Keep the existing end-to-end image smoke passing.
- Document any architecture-level mismatch that blocks exact parity.
