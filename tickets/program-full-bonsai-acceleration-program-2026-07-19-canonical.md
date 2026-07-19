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

**Canonical plan:** `docs/research/full-acceleration-program-2026-07-19.md`

This ticket is the dispatch hub for the full program (P0–P10). The older
`verify-spec-acceleration-routemap` keeps the measurement ledger / refutations;
**ranked actionable and phase order live here + the doc.**

## Physics (do not invert)

- Identity: tok/s ≈ (accept+1)/round_wall; verify ~84%, propose ~13%
- Hot kernels **instruction-issue-bound** (not DRAM-bound)
- mm2d verify @ m≤8 = M3-local ceiling (settled)
- Live classes: raise accept, skip verify work, exact scraps, product surface

## Phase order

| Phase | Name | First tickets |
|---|---|---|
| P0 | Truth | board-hygiene-*, blessed-v3-standings-*, eval-quality-reference-battery |
| P1 | Free flags | tree-pld-fused-argmax-*, gdn-gate-state-histogram-*, oracle-best-width-*, layer-hidden-accept-probe-* |
| P2 | Exact scraps | m-1-bitplane-*, int2b-signed-*, mlx-vs-lmbrrr-* |
| P3 | Control flow | bonsai-gguf-port-specscheduler-*, token-recycle-harvest-*, per-position-gdn-* |
| P4 | Accept policy | class-entropy-adaptive-*, grammar-schema-* |
| P5 | Approx verify | early-exit-checkpoint-*, sprinter-*, on-device-overnight-* |
| P6 | Hybrid | selective-gdn-*, mid-layer-target-* |
| P7 | Drafter | width-7-free-fidelity-*, eagle3-*, weaver-*, parked width-7 retrain |
| P8 | Deep tree | tree-speculation-*, gdn-rollback-free-*, after P1 tree + |
| P9 | Product | eval-multiturn-*, mtlbinaryarchive-*, eval-memory-envelope, eval-latency-surface |
| P10 | Gated $ | mm2d-fullk-*, width-7 retrain, m5-*, mlx-format migration |

## Exactness classes

E = exact/byte-match · Q = quality-gated · P = product-mode flag

## Kill rule

Every child spike names a kill criterion in its body. No kill criterion → not ready.

## Do-not-retry

See program doc §Do-not-retry. Do not re-open closed(wontdo) kernel routes without new regime evidence.
