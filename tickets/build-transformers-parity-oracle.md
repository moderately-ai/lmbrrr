---
id: build-transformers-parity-oracle
title: Build Transformers parity oracle for MiniCPM-V-4.6 runner
status: done
priority: p1
dependencies: [scaffold-baseline-harness]
related: [implement-naive-text-inference-runner, port-qwen35-text-decoder, port-minicpm-v46-full-path]
scopes: [evals, docs/research]
shared_scopes: []
paths: [tests/minicpm_v46_text_parity.rs]
tags: [parity, evals, minicpm]
---
## Goal
Create a repeatable Transformers-based oracle for the current Candle MiniCPM-V-4.6 runner.

## Acceptance
- Capture prompt formatting, token ids, image placeholder expansion, and selected logits/next-token outputs from the upstream Transformers implementation.
- Add Rust-side comparison tests or fixtures that can run without ambiguity about model revision and input assets.
- Cover text-only, single-image, and at least one longer-context reasoning prompt.
- Document tolerances, known mismatches, and exact commands in docs/research.
