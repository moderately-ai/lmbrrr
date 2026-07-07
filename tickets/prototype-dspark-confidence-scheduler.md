---
id: prototype-dspark-confidence-scheduler
title: Prototype DSpark confidence scheduler
status: done
priority: p2
dependencies: [design-dflash-block-drafter]
related: []
scopes: [inference/speculative]
shared_scopes: [docs/research]
paths: [src/main.rs, docs/research/dspark-confidence-scheduler.md]
tags: [speculative, dspark, scheduler]
---
After a block drafter exists, add DSpark-style confidence outputs and a single-request verification-length scheduler. Measure calibration, accepted length, verifier waste, and exact greedy output preservation before considering batch-aware scheduling.
