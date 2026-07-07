---
id: integrate-eagle-draft-runner
title: Integrate EAGLE draft head into runner
status: done
priority: p1
dependencies: [train-eagle-draft-head-from-traces, implement-greedy-spec-verifier]
related: []
scopes: [inference/speculative, runtime/candle]
shared_scopes: []
paths: [src/main.rs, src/minicpm.rs, src/qwen35.rs, docs/research/eagle-draft-runner.md, evals/eagle/train_eagle_draft_head.py, docs/research/eagle-draft-head-training.md]
tags: [speculative, eagle, performance]
---
## Goal

Run the trained EAGLE-style draft head inside the Candle generation loop and verify drafted chains against the target model.

## Acceptance

- Load a draft-head artifact and generate direct-token chains during decode.
- Verify chains with exact greedy reconstruction and report accepted length, waste, and speed.
- Compare against baseline generation on matched prompts.
- Document where overhead dominates if no speedup is achieved.
