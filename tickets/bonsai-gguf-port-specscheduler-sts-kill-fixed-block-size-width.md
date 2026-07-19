---
id: bonsai-gguf-port-specscheduler-sts-kill-fixed-block-size-width
title: "Bonsai gguf: port SpecScheduler + STS (kill fixed block_size width)"
status: done
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

## RESCOPE (2026-07-19 oracle)

`oracle-best-width-offline-from-confidences-scheduler-ev-gate` measured:

- **Prefix-width admission = 0% lift** under flat mm2d verify (cost invariant in m≤8).
- **Skip-on-zero-accept = +12–22% oracle lift** (exact +21.6%, m1 +11.8%).
- Conf head separates accept/reject means but conf-threshold *width* cuts hurt.

**Implement instead:** predict total-reject rounds (from conf vector / STS) and
**skip draft → plain greedy step** (chain-handoff), not Appendix-A left-to-right
width scan. Kill criterion: suite ≥ +3% vs fixed width-4, no class −2%.

