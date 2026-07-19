---
id: weaver-feature-reuse-adapter
title: Weaver feature-reuse drafter adapter (share embed/unembed + target hiddens)
status: todo
priority: p2
dependencies: []
related: [ingest-external-dspark-head, tree-speculation-over-dspark, program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [inference/speculative]
shared_scopes: []
paths: []
tags: [route-map, acceptance, research]
---
Bucket A / A3. Weaver adapter (arXiv 2607.06763): lightweight autoregressive head that reuses target hidden states and SHARES the target's embedding + output projection (near-zero added bandwidth), restoring conditional dependence between proposed tokens so wide trees stop being wasted. Lossless (changes only proposals). ~+10% alone (tau 4.44->4.90), a multiplier under Route C2 trees. Concrete port onto the existing DSpark-Bonsai drafter: feed target hiddens, share embed/unembed, emit a top-w frontier tree instead of a chain.
