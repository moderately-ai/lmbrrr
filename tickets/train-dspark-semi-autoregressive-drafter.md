---
id: train-dspark-semi-autoregressive-drafter
title: Train DSpark semi autoregressive drafter
status: todo
priority: p1
dependencies: [design-full-dspark-drafter]
related: []
scopes: [inference/speculative, evals, runtime/candle]
shared_scopes: [docs/research]
paths: [evals/dspark/**, evals/eagle/**, src/main.rs, docs/research/dspark-semi-autoregressive-training.md]
tags: [speculative, dspark, training]
---
## Goal

Train a DSpark-style drafter with a parallel backbone plus lightweight semi-autoregressive Markov head, not an EAGLE-only recurrent chain.

## Acceptance

- Build a trace dataset with target hidden features, target top-k distributions, target accepted-token labels, and draft-position labels.
- Train a parallel block drafter that emits base logits for multiple future positions in one forward path.
- Add a Markov sequential head that conditions each position on the previous sampled draft token.
- Train a confidence head using per-position prefix survival labels derived from draft/target distribution mismatch.
- Export a safetensors artifact and manifest with candidate vocabulary, confidence head, draft width, and calibration metadata.
