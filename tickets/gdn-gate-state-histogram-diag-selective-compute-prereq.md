---
id: gdn-gate-state-histogram-diag-selective-compute-prereq
title: GDN gate/β/state-Δ histogram diag (selective-compute prereq)
status: done
priority: p1
dependencies: []
related: [program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [evals, runtime/candle]
shared_scopes: []
paths: []
tags: [route-map, hybrid, program-2026-07-19]
---

## Program ID: `P1.5`

**Exactness:** —  
**Canonical plan:** `docs/research/full-acceleration-program-2026-07-19.md`  
**Hub:** `program-full-bonsai-acceleration-program-2026-07-19-canonical`

### Why
prereq P6.1

Instrumentation only.

### Spike
diag: histogram |β|, decay, state Δ per layer per token on Bonsai decode+verify

### Kill / done-when
if <10% near-zero mass, kill selective-gdn ticket

### Reporting
Ticket comment must include: exactness, blessed config, regime tags, measured-vs-inferred, kill result, what it does NOT prove.

