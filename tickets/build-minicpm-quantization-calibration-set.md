---
id: build-minicpm-quantization-calibration-set
title: Build MiniCPM quantization calibration set
status: todo
priority: p1
dependencies: [design-dynamic-quantization-lab]
related: []
scopes: [evals]
shared_scopes: [docs/research]
paths: []
tags: [quantization, calibration, evals]
---
Create deterministic calibration fixtures for MiniCPM-V-4.6 quantization, following `docs/research/dynamic-quantization-lab.md`.

Acceptance:

- Add text calibration prompts covering short factual, arithmetic, long reasoning, code, tool-style, and thinking enabled/disabled.
- Add VLM calibration metadata for representative image/OCR shapes once multimodal eval paths are ready.
- Store calibration metadata under `evals/calibration/` without embedding large media bytes in JSON.
- Document how to regenerate or extend the calibration set.
