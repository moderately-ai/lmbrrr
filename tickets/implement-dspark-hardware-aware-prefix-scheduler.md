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
- Preserve causal/early-stopping behavior so scheduling decisions do not introduce retrospective selection bias. Encode the paper's Appendix A invariant explicitly: the prefix scan must not evaluate c_{k+1} (a function of the realized token x_k) before position k is admitted, and must break at the first non-improving throughput. Unit-test the Appendix A scenario (a1 = 0.8, SPS = {1.0, 0.5, 0.45}): the scheduler returns length 0 without ever reading c_2.
- Schedule on STS-calibrated cumulative survival probabilities per the paper; DeepSpec ships only a raw per-position static threshold, which this ticket supersedes.
- Support single-request mode and a local multi-request simulation/batch mode.
- Report scheduled length, expected accepts, actual accepts, verifier waste, and throughput objective terms.
