---
id: integrate-dspark-block-runner
title: Integrate DSpark block runner
status: todo
priority: p1
dependencies: [train-dspark-semi-autoregressive-drafter]
related: []
scopes: [inference/speculative, runtime/candle, evals]
shared_scopes: [docs/research]
paths: [src/main.rs, src/minicpm.rs, src/qwen35.rs, evals/dspark/**, docs/research/dspark-block-runner.md]
tags: [speculative, dspark, performance]
---
## Goal

Load the trained DSpark drafter in Rust and run a real speculative cycle: one target anchor, DSpark block draft, scheduled target verification, exact greedy reconstruction.

## Acceptance

- Load the DSpark backbone, Markov head, and confidence head from safetensors.
- Propose a draft block before target verification without computing target hidden states for future positions.
- Verify the scheduled prefix in one target chunk and reconstruct exact greedy output.
- Report draft latency, verify latency, accepted length, verifier waste, confidence scores, and target calls saved.
- Compare directly against the recurrent EAGLE runner on the same prompts and draft widths.
