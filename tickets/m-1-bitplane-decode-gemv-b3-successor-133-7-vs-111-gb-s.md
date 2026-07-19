---
id: m-1-bitplane-decode-gemv-b3-successor-133-7-vs-111-gb-s
title: m=1 bitplane decode GEMV (B3 successor; 133.7 vs 111 GB/s)
status: done
priority: p1
dependencies: []
related: [bitplane-popcount-twotier-verify, wider-unpack-weight-code, program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [runtime/metal, candle-fork]
shared_scopes: []
paths: []
tags: [route-map, kernel, program-2026-07-19]
---

## Program ID: `P2.1`

**Exactness:** E  
**Canonical plan:** `docs/research/full-acceleration-program-2026-07-19.md`  
**Hub:** `program-full-bonsai-acceleration-program-2026-07-19-canonical`

### Why
B3 measured 133.7 vs 111 GB/s at m=1; two-tier verify already lost to mm2d

Build m=1 bitplane popcount/sign-plane GEMV for Q2_0 decode path only. Do NOT revive two-tier verify.

### Spike
isolated profile-kernel then e2e plain decode A/B on M3

### Kill / done-when
kill if e2e plain median < +2% non-overlapping

### Reporting
Ticket comment must include: exactness, blessed config, regime tags, measured-vs-inferred, kill result, what it does NOT prove.

