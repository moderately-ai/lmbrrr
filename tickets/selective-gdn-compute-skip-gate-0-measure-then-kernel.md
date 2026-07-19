---
id: selective-gdn-compute-skip-gate-0-measure-then-kernel
title: Selective GDN compute skip (gate≈0) — measure then kernel
status: todo
priority: p2
dependencies: []
related: [program-full-bonsai-acceleration-program-2026-07-19-canonical, fuse-deltanet-decode-step-kernel]
scopes: [runtime/metal, candle-fork]
shared_scopes: []
paths: []
tags: [route-map, hybrid, program-2026-07-19]
---

## Program ID: `P6.1`

**Exactness:** E→Q  
**Canonical plan:** `docs/research/full-acceleration-program-2026-07-19.md`  
**Hub:** `program-full-bonsai-acceleration-program-2026-07-19-canonical`

### Why
uses selective SSM structure

Measure first (gdn-gate histogram ticket). Only then implement skip.

### Spike
after P1.5 histogram; kernel flag skip recurrence when gate≈0

### Kill / done-when
kill if <10% skip mass or PPL moves

### Reporting
Ticket comment must include: exactness, blessed config, regime tags, measured-vs-inferred, kill result, what it does NOT prove.

