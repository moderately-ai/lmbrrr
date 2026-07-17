---
id: async-cross-engine-draft-verify
title: Async cross-engine draft/verify overlap (drafter on CPU/AMX while GPU verifies)
status: todo
priority: p2
dependencies: []
related: [cut-drafter-propose-cost, dspark-ondevice-chunk-assembly]
scopes: [inference/speculative]
shared_scopes: []
paths: []
tags: [route-map, verify-structure, research]
---
Bucket C / C3. Decouple propose/verify: run round N+1 drafter on CPU/AMX (or fp16 ANE) concurrently with GPU verify of round N, rollback on disagreement. AMUSD (arXiv 2410.17375) 1.96x, PipeInfer (arXiv 2407.11798) 2.15x -- but on MULTI-device. VERDICT uncertain/MARGINAL on a single M3: the GPU verify IS the wall (no pipeline bubble to overlap into), so hiding the ~20% draft caps ~1.25x; real value only if the drafter host is genuinely idle and continuous speculation keeps the tile fed. Lossless. ANE is fp16-only + ~2.3 ms dispatch (Orion arXiv 2603.06728) so CPU/AMX is the better drafter host.
