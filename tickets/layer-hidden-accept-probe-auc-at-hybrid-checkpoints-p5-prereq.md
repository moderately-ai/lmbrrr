---
id: layer-hidden-accept-probe-auc-at-hybrid-checkpoints-p5-prereq
title: Layer-hidden→accept probe AUC at hybrid checkpoints (P5 prereq)
status: todo
priority: p1
dependencies: []
related: [program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [evals]
shared_scopes: []
paths: []
tags: [route-map, approx-verify, program-2026-07-19]
---

## Program ID: `P1.6`

**Exactness:** —  
**Canonical plan:** `docs/research/full-acceleration-program-2026-07-19.md`  
**Hub:** `program-full-bonsai-acceleration-program-2026-07-19-canonical`

### Why
prereq for approximate verify

Offline probe only. No production path yet.

### Spike
capture verify hiddens at layers ~15/31/47/63; linear probe → accept/reject; report AUC

### Kill / done-when
if best AUC < 0.85 kill P5.1 early-exit family

### Reporting
Ticket comment must include: exactness, blessed config, regime tags, measured-vs-inferred, kill result, what it does NOT prove.

