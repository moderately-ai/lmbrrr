---
id: relaxed-typical-acceptance-mode
title: Relaxed typical acceptance mode
status: todo
priority: p2
dependencies: [integrate-dspark-block-runner]
related: [tree-speculation-over-dspark]
scopes: [inference/speculative, runtime/candle]
shared_scopes: [docs/research]
paths: [src/main.rs, docs/research/relaxed-acceptance.md]
tags: [speculative, campaign-1000]
---
## Goal

Optional acceptance rule that admits a draft token when it falls inside the target's typical/top-k set instead of requiring exact argmax match (Medusa-style typical acceptance). Legal under the campaign's anything-goes quality bar; expected +20-50% tau.

## Acceptance

- `--acceptance {exact,typical}` flag with typical-set parameters (epsilon/top-k) on the DSpark runner; exact remains the default.
- Quality report comparing exact vs typical outputs on the calibration prompt set (prefix/jaccard metrics, sample texts) published alongside the tau gain.
- Interacts correctly with the confidence scheduler (acceptance probabilities recalibrated for the relaxed rule or documented as approximate).
