---
id: design-full-dspark-drafter
title: Design full DSpark drafter implementation
status: done
priority: p1
dependencies: [implement-real-eagle-recurrent-drafter]
related: []
scopes: [inference/speculative]
shared_scopes: [docs/research]
paths: [docs/research/full-dspark-drafter-design.md]
tags: [speculative, dspark, design]
---
## Goal

Translate the DSpark paper into an implementation plan for this repo that targets the full method, not just confidence scheduling over an EAGLE-style drafter.

## Acceptance

- Specify the DSpark parallel backbone, semi-autoregressive Markov/RNN head, confidence head, and verification scheduler as separate runtime/training components.
- Define the local MiniCPM/Qwen hidden-state features, target distribution labels, draft distribution labels, and calibration data required for training.
- Define speedup gates against greedy, recurrent EAGLE, and any DFlash baseline available at the time.
- Identify which parts can be implemented in Python training first and which must run in Candle/Metal for measured speedup.
