---
id: component-aware-self-spec-refuted
title: "[REFUTED] Component-aware self-speculation (linear pathway as draft)"
status: closed
priority: p3
dependencies: []
related: [tree-speculation-over-dspark]
scopes: [inference/speculative]
shared_scopes: []
paths: []
tags: [route-map, verify-structure, refuted]
closed_reason: wontdo
closed_note: "REFUTED-measured: sequential hybrid -> PPL 82x, 0.026x wall-clock (38x slowdown). Parallel-hybrid-only (arXiv 2605.01106)."
---
Bucket C / C6. Isolate the SSM/linear pathway as a zero-cost draft by suppressing attention (arXiv 2605.01106). REFUTED-measured for SEQUENTIAL-interleaved hybrids (Qwen3.5/3.6 = ours): attention layers are serial pipeline stages, removing them blows perplexity 81.96x -> alpha=0.038 -> 0.026x wall-clock (a 38x SLOWDOWN, measured from the paper's full text). Works only for PARALLEL hybrids (Falcon-H1: alpha=0.68). Explicitly do-not-pursue. Closing won't-do.
