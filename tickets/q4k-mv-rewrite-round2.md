---
id: q4k-mv-rewrite-round2
title: "q4_K mv rewrite round 2: the four untested dimensions (TG geometry, magic dequant, shape specialization, split-k)"
status: closed
priority: p1
dependencies: []
related: []
scopes: [candle-fork]
shared_scopes: []
paths: []
tags: [kernels, campaign-1000]
---
RE-OPENS the mv floor with receipts for why 'closed' was too broad. The falsification ledger (6 schedule variants, SoA layout, q8_0 fallback, N_DST tiling) closed per-thread SCHEDULE, LAYOUT, and FORMAT — it never tested: (V1) THREADGROUP GEOMETRY: every mv TG is ONE simdgroup (the multi-sg indexing is commented out at quantized.metal:4975; host dispatches nth 4x8=32); the lm_head = 62k TG launches against a measured 54.6% LAUNCH limiter; MLX qmv_fast uses up to 8 sg/TG. nsg sweep 2/4/8. (V2) CONVERT-PIPE BYPASS: inner loop is AND + int->float CVT + FMA (~3 ops per 0.5-byte element, issue-saturated at 13-20% occupancy); CUDA-style magic dequant as_type<half>(0x6400|nibble)-1024 is EXACT for 0-15 and the -1024 folds into the existing sumy/dmin correction -> OR+FMA (~2 ops), off the int-convert pipe. No prior variant touched the convert. (V3) FUNCTION-CONSTANT shape specialization: k=1024 -> nb=4 -> ONE block-iteration/thread, loop+index arithmetic dominates (the original pre-SoA diagnosis); compile k in {1024,2048,3584} constants, unroll fully. (V4) SPLIT-K for n<=2048 projections (down/out_proj launch only 8-16k threads — occupancy-impossible at m=1 without splitting the reduction). PRIZE: q4_K 132 GB/s eff vs q8_0 262 on the same shapes; halving the gap = mv 2.0->1.3ms = +30-35% bench greedy EXACT — larger than the rejected 32k head. GATES: micro-first (qmv bench per shape), kill any vector <1.15x on its target shape; V2 numerics = exact dequant values (0-15 exact in half), accumulation-grouping decides bit-preserving vs margin-gate class; e2e per protocol; full refit + re-price (chunk table, tree, recycling, 32k-head math) after. RISK, stated: micro wins here have repeatedly evaporated e2e, and the even 4-7%/line profile could be TOTAL-issue saturation; the kill gates are the discipline.
