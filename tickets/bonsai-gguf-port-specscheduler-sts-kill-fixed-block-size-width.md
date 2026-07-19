---
id: bonsai-gguf-port-specscheduler-sts-kill-fixed-block-size-width
title: "Bonsai gguf: port SpecScheduler + STS (kill fixed block_size width)"
status: todo
priority: p1
dependencies: []
related: [implement-dspark-hardware-aware-prefix-scheduler, program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [inference/speculative, runtime/candle]
shared_scopes: []
paths: []
tags: [route-map, program-2026-07-19]
---

## Program ID: `P3.1`

**Exactness:** E  
**Canonical plan:** `docs/research/full-acceleration-program-2026-07-19.md`  
**Hub:** `program-full-bonsai-acceleration-program-2026-07-19-canonical`

### Why
gguf today: width = drafter.config.block_size fixed

Port MiniCPM SpecScheduler into gguf Bonsai path. Requires confidence head outputs already present.

### Spike
wire schedule_prefix_width + STS into gguf spec after P1.4 oracle

### Kill / done-when
kill if 6-class suite ≤ fixed block_size width

### Reporting
Ticket comment must include: exactness, blessed config, regime tags, measured-vs-inferred, kill result, what it does NOT prove.

