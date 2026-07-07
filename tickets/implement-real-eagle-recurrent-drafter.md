---
id: implement-real-eagle-recurrent-drafter
title: Implement real EAGLE recurrent drafter
status: todo
priority: p1
dependencies: [integrate-dspark-adaptive-scheduler]
related: []
scopes: [inference/speculative, runtime/candle, evals]
shared_scopes: [docs/research]
paths: [src/main.rs, src/minicpm.rs, src/qwen35.rs, evals/eagle/**, docs/research/real-eagle-recurrent-drafter.md]
tags: [speculative, eagle, performance]
---
## Goal

Move from the current live EAGLE probe to a drafter that can propose future tokens without first computing target hidden states for those future positions.

## Acceptance

- Define the recurrent or feature-prediction state carried by the drafter between proposed tokens.
- Train/export a smoke drafter with a full-vocabulary or defensible candidate-vocabulary head.
- Add a runner path that drafts multiple tokens before target verification rather than after target forward.
- Report accepted length, target calls saved, drafter overhead, exact greedy reconstruction, and end-to-end token-rate impact.
