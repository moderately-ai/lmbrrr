---
id: q4k-mv-round3-production-arbitrated
title: "q4_K mv round 3: production-arbitrated kernel variants (nr0=2, rt2, K-amortization) + M3/macOS-26 packed_numeric"
status: done
priority: p1
dependencies: []
related: []
scopes: [candle-fork]
shared_scopes: []
paths: []
tags: [kernels, campaign-1000]
---
The deep fanout (5 agents + 2 experiment rounds) OVERTURNED the 'floor closed' verdict in one specific way and confirmed it in another. OVERTURNED: llama.cpp/MLX achieve ~250-270 GB/s effective 4-bit matvec on M4-class hardware (llama.cpp #4167: Q4_0 7B M4 Max 69.95 tok/s = 267 GB/s) vs our production head at 196 GB/s (trace) — a ~1.3x real gap. CONFIRMED: the inner loops are BYTE-IDENTICAL (agent read ggml-metal.metal:8080-8198 — same mask trick, same deferred scales, same simd_sum); ~2-3 ops/element is universal; no ALU/format lever exists. THE GAP IS GEOMETRY + SHAPE: (a) llama.cpp runs N_R0=2 (we run N_DST=4) as a DELIBERATE register-spill fix (their PR #20399, 1.06-1.11x tg); (b) their models have K=4096-14336 which amortizes fixed per-thread cost 4-14x better than our K=1024 (our hidden size — we live permanently in the short-K regime); (c) our rt2 row-tile variant (+10% within-run on the head, bitwise-identical, fork 86e552cf) partially amortizes the same fixed cost. MEASUREMENT DISCIPLINE (hard lesson): the isolated serialized-dispatch micro-bench drifts +-35% cross-session on identical code (121 vs 77-79 GB/s) — it CANNOT arbitrate 1.3x questions. All further kernel variants are bit-identical per row -> arbitrate IN PRODUCTION: swap the dispatch, text-identical gate, bench-mode greedy A/B + trace per-encoder GB/s. WORK: (1) N_DST=2 variant (llama.cpp's spill fix — cheapest, best-evidenced); (2) rt2 row-tile; (3) N_DST=2 x rt2 combined; (4) verify short-K theory with a synthetic k=4096 bench row (explains, doesn't fix); (5) ON THE M3/macOS-26 BOX (user offered access): packed_numeric_type::unpack<half, uint4b_format, 8/16> micro-bench — the ONLY primitive found that structurally breaks 3-ops/element (MSL 4.1 §2.21; header absent on macOS 15, confirmed); if it lowers to a real instruction, it re-prices everything and justifies the OS upgrade path. Honest expected value: (1)-(3) sum to maybe +10-20% on the mv block = +7-15% e2e; (5) unknown, potentially large.
