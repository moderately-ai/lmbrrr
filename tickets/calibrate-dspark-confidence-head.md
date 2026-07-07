---
id: calibrate-dspark-confidence-head
title: Calibrate DSpark confidence head
status: todo
priority: p1
dependencies: [train-dspark-semi-autoregressive-drafter]
related: []
scopes: [inference/speculative, evals]
shared_scopes: [docs/research]
paths: [evals/dspark/**, docs/research/dspark-confidence-calibration.md]
tags: [speculative, dspark, calibration]
---
## Goal

Calibrate DSpark confidence outputs so scheduled prefix lengths are based on empirical prefix survival probabilities, not raw overconfident scores.

## Acceptance

- Build held-out validation traces for confidence calibration.
- Produce reliability diagrams or equivalent bins comparing predicted cumulative prefix survival to observed acceptance.
- Implement post-hoc calibration compatible with the paper's STS-style scaling requirement.
- Export calibrated confidence parameters with the drafter artifact.
- Demonstrate that higher scheduled confidence increases empirical acceptance rate and reduces verifier waste.
