---
id: port-minicpm-v46-full-path
title: Port full MiniCPM-V-4.6 multimodal path
status: in-progress
priority: p1
dependencies: [port-qwen35-text-decoder]
related: []
scopes: [model/minicpm, model/vision, runtime/candle]
shared_scopes: []
paths: []
tags: [implementation, minicpm, vlm]
---
## Goal

Implement the MiniCPM-V-4.6 multimodal path after the Qwen3.5 text decoder is proven.

## Work

- Implement or adapt the MiniCPM image processor behavior: slicing, scale resolution, patch sizing, image IDs, and normalization.
- Port the vision encoder and multimodal bridge/insertion logic.
- Validate image-token placement and layer-6 insertion against Transformers.
- Add basic OCR/image prompt fixtures to the eval harness.

## Acceptance

- Full MiniCPM-V-4.6 can run at least one image-conditioned prompt with validated tensor shapes and plausible output.
- Remaining parity or performance gaps are documented as follow-up tickets.
