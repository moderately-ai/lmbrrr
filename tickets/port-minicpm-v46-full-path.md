---
id: port-minicpm-v46-full-path
title: Port full MiniCPM-V-4.6 multimodal path
status: todo
priority: p1
dependencies: [port-qwen35-text-decoder]
related: []
scopes: [model/minicpm, model/vision, runtime/candle]
shared_scopes: []
paths: [src/minicpm.rs, src/image_processor.rs, src/prompt.rs, src/main.rs, tests/minicpm_v46_text_parity.rs, docs/research/minicpm-v46-image-parity.md]
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


## Current Status

The image processor, placeholder expansion, vision tower, merger, and image embedding replacement paths are implemented. Remaining work is to validate at least one end-to-end image-conditioned run and document any parity/performance gaps before closing this ticket.
