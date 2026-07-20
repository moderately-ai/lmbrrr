---
id: on-device-overnight-trajectory-mining-accept-predictor-dataset
title: On-device overnight trajectory mining → accept-predictor dataset
status: in-progress
priority: p2
dependencies: []
related: [sprinter-approx-verify-audit, program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [evals, inference/speculative]
shared_scopes: []
paths: []
tags: [route-map, approx-verify, program-2026-07-19]
claimed_from: todo
assignee: agent-program
lease_expires_at: 1784558084
---

## Program ID: `P7.7 / P5 data`

**Exactness:** Q  
**Canonical plan:** `docs/research/full-acceleration-program-2026-07-19.md`  
**Hub:** `program-full-bonsai-acceleration-program-2026-07-19-canonical`

### Why
feeds P5.2

No Modal required for data collection.

### Spike
log (features→accept) overnight; build dataset for SPRINTER head

### Kill / done-when
kill if 1 week data cannot train probe AUC≥0.85

### Reporting
Ticket comment must include: exactness, blessed config, regime tags, measured-vs-inferred, kill result, what it does NOT prove.

