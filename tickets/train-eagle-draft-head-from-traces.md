---
id: train-eagle-draft-head-from-traces
title: Train EAGLE draft head from hidden-state traces
status: done
priority: p1
dependencies: [record-hidden-state-traces, prototype-eagle-chain-drafter]
related: []
scopes: [inference/speculative, evals, runtime/candle]
shared_scopes: []
paths: [src/main.rs, evals/eagle/**, docs/research/eagle-draft-head-training.md]
tags: [speculative, eagle, training]
---
## Goal

Train a small direct-token EAGLE-style draft head from captured MiniCPM/Qwen hidden-state traces.

## Acceptance

- Build a trace dataset from prompts with captured low/mid/high hidden states and greedy next tokens.
- Train or fit a small draft head reproducibly with uv-managed tooling or Candle code.
- Export draft-head weights and metadata in a runner-loadable format.
- Report offline top-1/top-k accuracy and expected accepted length.
