---
id: per-position-gdn-state-emission-for-free-partial-accept-rollback
title: Per-position GDN state emission for free partial-accept rollback
status: todo
priority: p2
dependencies: []
related: [emit-per-position-verify-states, program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [runtime/metal, candle-fork]
shared_scopes: []
paths: []
tags: [route-map, program-2026-07-19]
---

## Program ID: `P3.3`

**Exactness:** E  
**Canonical plan:** `docs/research/full-acceleration-program-2026-07-19.md`  
**Hub:** `program-full-bonsai-acceleration-program-2026-07-19-canonical`

### Why
graveyard item from speculative-state-rollback.md

Fork kernel change. Gate with byte-parity on partial accept.

### Spike
gated_delta_chunk writes per-position S; rollback selects without re-advance

### Kill / done-when
kill if realized rollback path saving < 1 ms/round

### Reporting
Ticket comment must include: exactness, blessed config, regime tags, measured-vs-inferred, kill result, what it does NOT prove.

