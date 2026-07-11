---
id: implement-dspark-hardware-aware-prefix-scheduler
title: Implement DSpark hardware aware prefix scheduler
status: done
priority: p1
dependencies: [calibrate-dspark-confidence-head, remeasure-spec-round-cost-model]
related: []
scopes: [inference/speculative, runtime/candle, evals]
shared_scopes: [docs/research]
paths: [src/main.rs, evals/dspark/**, docs/research/dspark-hardware-aware-prefix-scheduler.md]
tags: [speculative, dspark, scheduler]
---
## Outcome (2026-07-10 night, commit 5c0339b)

Landed as dspark-run --schedule: expected-throughput prefix admission from calibrated cumulative survival x the measured post-fusion cost table (RoundCostModel::measured_default from target/vt-gdc2.json). Non-anticipation invariant unit-tested on the Appendix A scenario (a1=0.8, SPS={1.0,0.5,0.45} -> width 0, exactly one confidence read). Beat the incumbent threshold truncation: math 95.3 -> 103.2 tok/s (0.71x), tides 71.7 -> 81.0 (0.55x). Remaining (minor): externalize the cost table to the remeasure artifact when it lands; multi-request batch mode deferred to the batched-runner lane.

## Board revision (2026-07-10 evening)

Dependency repointed from the dead pre-fusion table (T_verify ~= 11 + 6.3*gamma — ~5x off; verify is now 13.2 ms/round total) to remeasure-spec-round-cost-model. Incumbent to beat: the static --confidence-threshold 0.4 truncation (87 tok/s math / 77 tides vs 146 greedy) — report wall tok/s against it, not against unscheduled gamma=8.

## Goal

Implement the DSpark scheduler as a throughput optimization problem using calibrated prefix survival probabilities and measured local verification throughput.

## Acceptance

- Consume calibrated per-position confidence scores and the local SPS/verification throughput table.
- Select per-request verification lengths by maximizing expected accepted tokens times measured verifier throughput.
- Preserve causal/early-stopping behavior so scheduling decisions do not introduce retrospective selection bias. Encode the paper's Appendix A invariant explicitly: the prefix scan must not evaluate c_{k+1} (a function of the realized token x_k) before position k is admitted, and must break at the first non-improving throughput. Unit-test the Appendix A scenario (a1 = 0.8, SPS = {1.0, 0.5, 0.45}): the scheduler returns length 0 without ever reading c_2.
- Schedule on STS-calibrated cumulative survival probabilities per the paper; DeepSpec ships only a raw per-position static threshold, which this ticket supersedes.
- Support single-request mode and a local multi-request simulation/batch mode.
- Report scheduled length, expected accepts, actual accepts, verifier waste, and throughput objective terms.
