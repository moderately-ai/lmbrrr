---
id: int2b-signed-ternary-mm2d-path-delete-p-rowsum-7-bound
title: int2b signed ternary mm2d path (delete P-rowsum; ≤7% bound)
status: done
priority: p2
dependencies: []
related: [metal-ternary-matmul-kernel, program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [runtime/metal, candle-fork]
shared_scopes: []
paths: []
tags: [route-map, kernel, program-2026-07-19]
---

## Program ID: `P2.2`

**Exactness:** E  
**Canonical plan:** `docs/research/full-acceleration-program-2026-07-19.md`  
**Hub:** `program-full-bonsai-acceleration-program-2026-07-19-canonical`

### Why
exact micro; bounded

Metal 4.1 int2b. Do not expect large win; close cleanly either way.

### Spike
mm2d B=int2b_format {-1,0,+1}; delete P-rowsum; A/B vs uint2b+fold

### Kill / done-when
kill if spec < +1% (probe_nofold bound ≤7%)

### Reporting
Ticket comment must include: exactness, blessed config, regime tags, measured-vs-inferred, kill result, what it does NOT prove.

