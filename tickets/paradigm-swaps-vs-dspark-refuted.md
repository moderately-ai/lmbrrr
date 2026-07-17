---
id: paradigm-swaps-vs-dspark-refuted
title: "[REFUTED] Paradigm swaps vs DSpark (Medusa / lookahead / cascade / self-spec)"
status: closed
priority: p3
dependencies: []
related: [ngram-draft-source-mux, tree-speculation-over-dspark]
scopes: [inference/speculative]
shared_scopes: []
paths: []
tags: [route-map, verify-structure, refuted]
closed_reason: wontdo
closed_note: "REFUTED: every paradigm that replaces DSpark's block drafter regresses acceptance; PLD already measured -17%."
---
Bucket C / C5. Every paradigm that REPLACES DSpark's learned diffusion-block drafter regresses the numerator we are trying to grow: Medusa (arXiv 2401.10774) = weaker conditionally-independent heads + wide trees (width is what a modest exact-argmax drafter punishes); lookahead/Jacobi (2402.02057) = drafter-free lower acceptance AND floods the m dimension past the flat-to-8 tile into higher tiles; cascade/staged (SpecInfer 2305.09781) = adds a mid-tier stage but does not shrink the one big ternary verify; self-spec/LayerSkip (2404.16710) = orthogonal, does not change verify shape. PLD already measured-refuted (see ngram-draft-source-mux: -17% prose, 0 tokens accepted). The non-regressive C routes KEEP DSpark and add tree structure (see tree-speculation-over-dspark). Closing won't-do.
