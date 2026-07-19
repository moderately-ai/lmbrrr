---
id: tree-pld-fused-argmax-flag-battery-on-bonsai-code-exists-measure
title: Tree+PLD+fused-argmax flag battery on Bonsai (code exists, measure)
status: done
priority: p1
dependencies: []
related: [tree-speculation-over-dspark, ngram-draft-source-mux, program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [inference/speculative, evals]
shared_scopes: []
paths: []
tags: [route-map, program-2026-07-19]
---

## Program ID: `P1.1-1.3`

**Exactness:** E  
**Canonical plan:** `docs/research/full-acceleration-program-2026-07-19.md`  
**Hub:** `program-full-bonsai-acceleration-program-2026-07-19-canonical`

### Why
CODE EXISTS — measure only

gguf_run use_tree; ngram PLD; LMBRRR_FUSED_VERIFY_ARGMAX. Tree TW=3 keeps m=7 in flat tile.

### Spike
M3 A/B: tree on/off; PLD on/off; FUSED_VERIFY_ARGMAX on/off; exact+m1; hard+easy prompts

### Kill / done-when
per-arm kill: < +2% (tree/PLD) or < +1% (argmax) → record negative keep flag off

### Reporting
Ticket comment must include: exactness, blessed config, regime tags, measured-vs-inferred, kill result, what it does NOT prove.

