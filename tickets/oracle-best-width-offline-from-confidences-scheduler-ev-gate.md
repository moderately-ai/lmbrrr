---
id: oracle-best-width-offline-from-confidences-scheduler-ev-gate
title: Oracle best-width offline from confidences (scheduler EV gate)
status: done
priority: p1
dependencies: []
related: [program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [evals]
shared_scopes: []
paths: []
tags: [route-map, program-2026-07-19]
---

## Program ID: `P1.4`

**Exactness:** —  
**Canonical plan:** `docs/research/full-acceleration-program-2026-07-19.md`  
**Hub:** `program-full-bonsai-acceleration-program-2026-07-19-canonical`

### Why
gates P3.1 EV

Offline analysis of existing or one instrumented run.

### Spike
from logged confidences compute oracle width vs fixed; translate to tok/s upper bound

### Kill / done-when
if oracle lift < +3% tok/s-equivalent, deprioritize scheduler port

### Reporting
Ticket comment must include: exactness, blessed config, regime tags, measured-vs-inferred, kill result, what it does NOT prove.

