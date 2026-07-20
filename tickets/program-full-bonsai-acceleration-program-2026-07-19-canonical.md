---
id: program-full-bonsai-acceleration-program-2026-07-19-canonical
title: "PROGRAM: Full Bonsai acceleration program 2026-07-19 (canonical)"
status: todo
priority: p1
dependencies: []
related: [verify-spec-acceleration-routemap]
scopes: [docs/research]
shared_scopes: []
paths: []
tags: [epic, route-map, program]
---

# PROGRAM EPIC: Full Bonsai acceleration (2026-07-19)

**Canonical plan + COLD-START HANDOFF:**
`docs/research/full-acceleration-program-2026-07-19.md` (read the handoff section at the bottom).

**As of git `802cd1e` (2026-07-19 evening).**

## Current OP (shipped)
- Default `gguf spec`: soft adapt `0,1.5,1,3` + planar mm2d + Q8_0 → **~20 tok/s** prose
- Escapes: `--no-adapt-margin` / `--exact` / `--fast` / `--grammar-json`
- M3: **one GPU workload at a time** (see AGENTS.md)

## Do not retry (this wave)
tree, PLD, fused argmax, skip-low-conf, skip-after-reject, recycle, whole-layer GDN skip,
int2b, LUT, chunk-assembly, per-pos GDN emission, MTLBinaryArchive, MLX qmv port,
block7 width-7 GGUF deploy, layer-RMS early-exit, hard adapt `1,2` as default.

## Next (ranked)
1. P0.5 full quality battery (local)
2. P4 class/entropy margin polish (local)
3. P7.4 Weaver train (Modal) — ~+11% ceiling if τ holds
4. P7.1 fidelity → P7.2 width-7 retrain (Modal; block7 artifact dead)
5. P7.3 EAGLE-3 (Modal)
6. P5 SPRINTER mid-forward head (not conf-skip)
7. P10.1 fullk-from-original (user $)
8. P8 tree deep only after rollback cheapening
9. P9 product: multiturn / 16GB / latency

## Physics (do not invert)
- `tok/s ≈ (accept+1)/round_wall`; verify ~84%, propose ~13% (backbone ~26 ms)
- Hot path instruction-issue-bound; mm2d m≤8 settled
- Conf AUC high ≠ skip-verify causal

## Ledger
Per-spike detail lives in **ticket comments** (append-only) and the program doc results log.
Older epic `verify-spec-acceleration-routemap` = refutation archive; ranked-actionable there is STALE.
