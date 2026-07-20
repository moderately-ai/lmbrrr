---
id: mtlbinaryarchive-pipeline-cache-cold-ttft
title: MTLBinaryArchive pipeline cache (cold TTFT)
status: done
priority: p2
dependencies: []
related: [eval-latency-surface, program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [candle-fork]
shared_scopes: []
paths: []
tags: [route-map, ttft, program-2026-07-19]
---

## Program ID: `P9.3`

**Exactness:** P  
**Canonical plan:** `docs/research/full-acceleration-program-2026-07-19.md`  
**Hub:** `program-full-bonsai-acceleration-program-2026-07-19-canonical`

### Why
product TTFT not standings

candle-metal-kernels kernel.rs / loader path.

### Spike
MTLBinaryArchive load/store around metallib pipelines; cold TTFT A/B

### Kill / done-when
kill if cold TTFT improvement < 5%

### Reporting
Ticket comment must include: exactness, blessed config, regime tags, measured-vs-inferred, kill result, what it does NOT prove.

