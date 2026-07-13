---
id: eval-deltanet-sdpa-roofline
title: "EVAL: the other 25% — DeltaNet decode + sdpa_vector achieved bandwidth vs roofline"
status: todo
priority: p2
dependencies: []
related: []
scopes: [candle-fork]
shared_scopes: []
paths: []
tags: [eval-wave, kernels]
---
WHY: the entire kernel eval program targets q4_K mv (74.6% of decode GPU time). The remaining ~25% — gated_delta_v2 decode (18 layers), sdpa_vector (6 layers), residual elementwise — has NO roofline number since the regrid work. Amdahl bounds the prize (2x on 25% = +14% GPU-side) but that is NOT negligible next to the +2% ship bars elsewhere, and nobody has checked whether the regridded DeltaNet kernel actually sits at the memory roof.

PROCEDURE:
1. TRAFFIC MODEL FIRST (paper, not code): from src/qwen35.rs + the fork's gated_delta_v2.metal, write down the decode-step bytes for one DeltaNet layer: f32 recurrent state read+write (heads x head_dim x head_dim... read the actual dims from the code — hidden 1024, conv_dim/value_dim/num_heads per config), conv state, qkvz/ba inputs, output. Same for sdpa_vector at KV length {1k, 4k, 16k}: K+V bf16 read per head. Post the byte counts here BEFORE benching — the roofline claim is only as good as this arithmetic.
2. MICRO: check candle-metal-kernels/examples/metal_benchmarks.rs for existing deltanet/sdpa tasks (earlier campaign work benched deltanet); add tasks if missing, cloning the nsg-sweep harness (warmup, MIN_DUR 1.5s, deployed shapes, bitwise gate vs the production kernel N/A — same kernel, this is measurement not variants). Achieved GB/s = modelled bytes / measured time, within-session per eval-protocol-ambient-control.
3. TRACE CROSS-CHECK: the decode-step-5 gputrace (repo root) already has per-kernel timings — compute achieved GB/s from the trace numbers as a second estimate; they should agree within ~20% or the traffic model is wrong (fix the model, not the measurement).

DECISION: achieved >= 75% of the 245 GB/s roof on both kernel families -> close as at-roof, record the numbers in the dossier (this PERMANENTLY prices all future 'optimize DeltaNet' suggestions — the point of the eval). 50-75% -> file one targeted round with a specific hypothesis (occupancy? f32 state traffic halvable via bf16 state with f32 accumulate? NOTE: bf16-recurrent-state-storage was CLOSED for accuracy compounding — read that ticket before proposing it again; the accuracy objection stands unless someone designs error-compensated storage). < 50% -> escalate to p1 with the traffic model + trace numbers as receipts.
