---
id: gdn-rollback-free-masked-solve
title: Rollback-free gated-DeltaNet masked-solve verify kernel (Trees-from-Marginals)
status: todo
priority: p1
dependencies: []
related: [tree-speculation-over-dspark, optimize-deltanet-chunked-prefill-and-verify-throughput, keep-deltanet-recurrent-state-f32, program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [runtime/metal]
shared_scopes: []
paths: []
tags: [route-map, kernel, verify-structure, research]
---
Bucket B / B7 (enabler for C2). Naive tree masking does not transfer to a running gated-DeltaNet recurrent state (a per-branch sequential scan would kill the M3). Trees-from-Marginals (arXiv 2607.06763, evaluated on Qwen3.6-27B = our family): verify a tree over a partial order via the dual-chunk gated-delta form as a masked triangular solve tiled in 32-node blocks, NEVER speculatively update state, commit + short-recurrence-replay only along the accepted branch. MEASURED (B200): fused solve 7.1x faster than per-branch recurrent at T=128; GDN verify only ~12% of step; lossless to 1e-4. This removes the unconditional +19 ms/round GDN rollback that killed the w=3 tree economics -> the prerequisite that makes lossless wide-tree verify (C2) affordable on our hybrid arch.
