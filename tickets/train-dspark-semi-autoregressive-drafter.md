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

- Build a trace dataset in binary shards (safetensors/npz, not per-token JSON) capturing every block position: anchor hidden features from the capture layers, target tokens, and target top-k distributions with a tail-mass bucket (k >= 64) or raw hidden states for local frozen-LM-head projection. The current JSON exporter only records the last position per forward and is not viable at corpus scale.
- Train a parallel block drafter that emits base logits for multiple future positions in one forward path, with DFlash-style target-context injection into draft K/V.
- Add a Markov sequential head (low-rank transition bias B = W1*W2, r ~= 256) that conditions each position on the previous sampled draft token.
- Use the full vocabulary via the frozen shared target embedding and LM head; no observed-vocabulary output head.
- Train with the paper's three-term objective: cross-entropy + total-variation distribution matching + confidence BCE, position-weighted by exp(-(k-1)/gamma) (default weights 0.1 / 0.9 / 1.0).
- Train a confidence head using per-position prefix survival labels c* = 1 - 0.5 * total-variation(draft, target).
- Trainer runs on CUDA (Modal credits are available for corpus generation and training) as well as local MPS for smoke runs; evaluate reusing DeepSpec before writing training code from scratch.
- Export a safetensors artifact and manifest with backbone, Markov head, confidence head, draft width, capture layers, and calibration metadata.
