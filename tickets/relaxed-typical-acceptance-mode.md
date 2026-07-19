---
id: relaxed-typical-acceptance-mode
title: Relaxed typical acceptance mode
status: done
priority: p2
dependencies: [integrate-dspark-block-runner]
related: [tree-speculation-over-dspark, program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [inference/speculative, runtime/candle]
shared_scopes: [docs/research]
paths: [src/main.rs, docs/research/relaxed-acceptance.md]
tags: [speculative, campaign-1000]
---
## Outcome (2026-07-10)

Landed as `--accept-margin <logits>` (exact remains default), composing with confidence truncation; quality report in docs/research/relaxed-acceptance.md. Measured at round-1 quality (tau~2): +3-9% wall rate at m=2-4 (math 0.86x -> 0.95x at m=4) with visible token-level artifacts ("seconds/promote" for "seconds/prompt") — the +20-50% estimate assumed rejections concentrate at near-ties, which is false for a weak drafter. Keep exact as default; re-evaluate m=1-2 after each training round as rejections shift toward genuine ties. Confidence recalibration under the relaxed rule flagged in the report JSON.

## Goal

Optional acceptance rule that admits a draft token when it falls inside the target's typical/top-k set instead of requiring exact argmax match (Medusa-style typical acceptance). Legal under the campaign's anything-goes quality bar; expected +20-50% tau.

## Acceptance

- `--acceptance {exact,typical}` flag with typical-set parameters (epsilon/top-k) on the DSpark runner; exact remains the default.
- Quality report comparing exact vs typical outputs on the calibration prompt set (prefix/jaccard metrics, sample texts) published alongside the tau gain.
- Interacts correctly with the confidence scheduler (acceptance probabilities recalibrated for the relaxed rule or documented as approximate).
