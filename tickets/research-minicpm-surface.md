---
id: research-minicpm-surface
title: Map MiniCPM-V-4.6 implementation surface
status: done
priority: p1
dependencies: []
related: []
scopes: [docs/research]
shared_scopes: [docs/papers, docs/artifacts]
paths: []
tags: [research, minicpm]
---
## Goal

Understand the exact Hugging Face implementation surface for MiniCPM-V-4.6 and translate it into a Candle/Metal port map.

## Work

- Vendor relevant model metadata and papers into docs.
- Inspect MiniCPM processor, config, model code, tensor names, and generation settings.
- Identify text-only, vision, multimodal bridge, cache/state, and MTP components.
- Document what can be implemented directly in Candle and what needs new kernels or runtime support.

## Acceptance

- A repo-local research note exists with source links, vendored artifacts, and an ordered implementation map.
- Open questions are explicit enough to turn into implementation tickets.
