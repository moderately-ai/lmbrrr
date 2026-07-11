---
id: remeasure-spec-round-cost-model
title: Rebuild the spec-round cost model from in-loop post-fusion measurements
status: todo
priority: p1
dependencies: []
related: [implement-dspark-hardware-aware-prefix-scheduler, cut-drafter-propose-cost]
scopes: [evals, inference/speculative]
shared_scopes: [docs/research]
paths: []
tags: [speculative, measurement, campaign-1000]
---
## Goal

The scheduler's declared input (docs/research/dspark-verification-throughput-table.md, T_verify ~= 11 + 6.3*gamma, 67 ms at gamma=8) is ~5x off post-fusion reality (verify 13.2 ms/round total) and would drive systematic under-verification. Rebuild the cost model from IN-LOOP measurements: T_verify(width) and T_propose(gamma) for width/gamma in 1..=12 across context lengths, plus fixed per-round overheads (syncs, capture concat, ctx append, from_slice uploads). Record where the 8.7 ms propose goes (backbone vs per-Markov-step lm_head/markov_w2 reads vs readback) — that breakdown decides cut-drafter-propose-cost's shape.

## Acceptance

- JSON artifact in the shape the scheduler consumes; scheduler ticket dependency repointed here.
- Propose-cost breakdown by phase.
- Mark docs/research/dspark-verification-throughput-table.md SUPERSEDED (pre-fusion) with a pointer.
- In-loop measurement (means over a real run), same-session protocol per lmbrrr-measurement-protocol memory.
