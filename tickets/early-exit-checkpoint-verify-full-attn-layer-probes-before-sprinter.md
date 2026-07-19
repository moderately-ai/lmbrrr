---
id: early-exit-checkpoint-verify-full-attn-layer-probes-before-sprinter
title: Early-exit / checkpoint verify (full-attn layer probes) before SPRINTER
status: todo
priority: p1
dependencies: []
related: [sprinter-approx-verify-audit, program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [inference/speculative, runtime/candle]
shared_scopes: []
paths: []
tags: [route-map, approx-verify, program-2026-07-19]
---

## Program ID: `P5.1`

**Exactness:** Q  
**Canonical plan:** `docs/research/full-acceleration-program-2026-07-19.md`  
**Hub:** `program-full-bonsai-acceleration-program-2026-07-19-canonical`

### Why
only structural escape from verify 84% ceiling

Early-exit verify at hybrid full-attn layer checkpoints. Quality-gated. Never default without audit.

### Spike
after P1.6 AUC≥0.85; heads at full-attn checkpoints; audit rate 1/8

### Kill / done-when
kill if AUC gate fails OR Bonsai PPL battery bar fails

### Reporting
Ticket comment must include: exactness, blessed config, regime tags, measured-vs-inferred, kill result, what it does NOT prove.

