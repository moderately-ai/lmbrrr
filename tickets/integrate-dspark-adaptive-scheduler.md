---
id: integrate-dspark-adaptive-scheduler
title: Integrate DSpark adaptive speculative scheduler
status: todo
priority: p2
dependencies: [prototype-dspark-confidence-scheduler, integrate-eagle-draft-runner]
related: []
scopes: [inference/speculative, runtime/candle]
shared_scopes: []
paths: [src/main.rs, docs/research/dspark-adaptive-runner.md]
tags: [speculative, dspark, scheduler]
---
## Goal

Move the DSpark-inspired confidence scheduler from offline verifier probes into an online speculative runner.

## Acceptance

- Let a drafter provide per-token confidence or calibrated proxy scores.
- Dynamically choose draft length during decode and verify accepted prefixes.
- Report accepted length, wasted draft tokens, target calls saved, and wall-clock speed.
- Compare fixed-width versus adaptive schedules on the same prompt set.
