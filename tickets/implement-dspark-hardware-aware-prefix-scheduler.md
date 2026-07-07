---
id: implement-dspark-hardware-aware-prefix-scheduler
title: Implement DSpark hardware aware prefix scheduler
status: todo
priority: p1
dependencies: [calibrate-dspark-confidence-head, profile-dspark-verification-throughput-table]
related: []
scopes: [inference/speculative, runtime/candle, evals]
shared_scopes: [docs/research]
paths: [src/main.rs, evals/dspark/**, docs/research/dspark-hardware-aware-prefix-scheduler.md]
tags: [speculative, dspark, scheduler]
---
## Goal

Implement the DSpark scheduler as a throughput optimization problem using calibrated prefix survival probabilities and measured local verification throughput.

## Acceptance

- Consume calibrated per-position confidence scores and the local SPS/verification throughput table.
- Select per-request verification lengths by maximizing expected accepted tokens times measured verifier throughput.
- Preserve causal/early-stopping behavior so scheduling decisions do not introduce retrospective selection bias.
- Support single-request mode and a local multi-request simulation/batch mode.
- Report scheduled length, expected accepts, actual accepts, verifier waste, and throughput objective terms.
