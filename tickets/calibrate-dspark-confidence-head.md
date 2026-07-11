---
id: calibrate-dspark-confidence-head
title: Calibrate DSpark confidence head
status: done
priority: p1
dependencies: [train-dspark-semi-autoregressive-drafter]
related: []
scopes: [inference/speculative, evals]
shared_scopes: [docs/research]
paths: [evals/dspark/**, docs/research/dspark-confidence-calibration.md]
tags: [speculative, dspark, calibration]
---
## Outcome (2026-07-10 night, commit 5c0339b)

Cumulative fit landed: per-position Platt (STS left-to-right) on 587 rounds x 8 domains; cumulative survival reliability validated within ~3 points in every bin (0.9->0.956, 0.5->0.556, 0.3->0.333, 0.1->0.244, 0.0->0.077). sts.json v2 exports per-position parameters beside the checkpoint; runner records are round-grouped for future refits. Consumed by the scheduler.

## Progress (2026-07-10)

First pass landed (commit bba107f): runner emits per-position (logit, calibrated p, accepted) records; Platt fit over 455 samples from 4 domains gives scale 0.992 / shift 0.211 — the round-1 head is near-calibrated already (reliability table in the commit/doc: 0.9+ predicted -> 0.94-0.955 actual). sts.json exported beside the checkpoint and loaded by the runner; --confidence-threshold truncates proposals per the DeepSpec contract with 0-draft rounds supported. Measured: t=0.4 at gamma=8 matches best static gamma on math (0.86x) and beats it on tides (0.68x vs 0.63x). Remaining for the scheduler (per acceptance below): calibrate the CUMULATIVE prefix-survival product (per-position temperature, left-to-right) rather than the marginal per-position scores, on a larger held-out trace set, and re-fit per training round.

## Goal

Calibrate DSpark confidence outputs so scheduled prefix lengths are based on empirical prefix survival probabilities, not raw overconfident scores.

## Acceptance

- Build held-out validation traces for confidence calibration.
- Produce reliability diagrams or equivalent bins comparing predicted cumulative prefix survival to observed acceptance.
- Implement post-hoc calibration compatible with the paper's STS-style scaling requirement. STS is absent from DeepSpec (it ships only ECE/AUROC/Brier diagnostics), so the left-to-right per-position temperature fit is ours; calibrate the cumulative product the scheduler consumes, not the raw per-position scores DeepSpec's static threshold path uses.
- Export calibrated confidence parameters with the drafter artifact.
- Demonstrate that higher scheduled confidence increases empirical acceptance rate and reduces verifier waste.
