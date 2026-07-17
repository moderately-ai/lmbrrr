---
id: wider-unpack-weight-code
title: Wider byte-aligned ternary weight code (spend spare DRAM BW to cut unpack)
status: todo
priority: p2
dependencies: []
related: [metal-ternary-matmul-kernel, spike-ternary-type42-block-format]
scopes: [quantization, runtime/metal]
shared_scopes: []
paths: []
tags: [route-map, kernel, research]
---
Bucket B / B4. The wall is the per-weight 2-bit unpack+scale-fold stealing FMA issue slots; DRAM is only ~29% used (~3.5x spare). Trade that spare, coalesced-load-friendly bandwidth (the primitive Apple GPUs are good at) for a byte-aligned / partially-unpacked ternary weight code so one 32-bit lane load feeds several MACs with minimal bit-extraction. Most diagnosis-aligned UNTRIED kernel idea; measure issued ALU/LSU instruction count delta, not GB/s.

PREMISE NOW MEASURED, TARGET RESCOPED (2026-07-17, B3 spike fallout): the bitplane kernel sustains **133.7 GB/s at m=1 on identical 2.125-bpw bytes where the exhausted mv reads 106** (Q4K mv = 142 = the platform roof for this access class) — direct proof the Q2_0 m=1 mv is ~25% INSTRUCTION-limited by its unpack, not bandwidth-limited. So B4's live target is the **m=1 decode mv** (plain-decode floor 14.42 -> ~17-18 tok/s if closed), NOT the verify (mm2d rules m=5-8 and the matmul2d op ceiling is architectural). Failed sketches recorded in the B3 spike comment (exact-from-sign-planes with float activations = mc-class cost; simd-ballot structure = marginal). The open design question: a weight code + inner loop with <=3 per-weight lane-ops that stays EXACT with bf16 activations.
