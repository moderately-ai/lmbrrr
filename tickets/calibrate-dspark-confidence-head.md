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
- Implement post-hoc calibration compatible with the paper's STS-style scaling requirement. STS is absent from DeepSpec (it ships only ECE/AUROC/Brier diagnostics), so the left-to-right per-position temperature fit is ours; calibrate the cumulative product the scheduler consumes, not the raw per-position scores DeepSpec's static threshold path uses.
- Export calibrated confidence parameters with the drafter artifact.
- Demonstrate that higher scheduled confidence increases empirical acceptance rate and reduces verifier waste.
