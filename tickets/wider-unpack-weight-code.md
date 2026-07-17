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

PREMISE NOW MEASURED, TARGET RESCOPED (2026-07-17, B3 spike fallout): the bitplane kernel sustains **133.7 GB/s at m=1 on identical 2.125-bpw bytes where the exhausted mv reads 106** (Q4K mv = 142 = the platform roof for this access class) — direct proof the Q2_0 m=1 mv is ~25% INSTRUCTION-limited by its unpack, not bandwidth-limited. So B4's live target is the **m=1 decode mv** (plain-decode floor 14.42 -> ~17 tok/s IF the mv reaches the ~142 roof AND that flows through the ~65% of the step that is quantized mv; INFERRED upper bound, not measured), NOT the verify (mm2d rules m=5-8 and the matmul2d op ceiling is architectural).

STALE PREMISE CORRECTED: this ticket's original "DRAM ~29% used, ~3.5x spare" is a DEAD number — the mv now measures 106/142 = ~75% of the achievable roof, so DRAM headroom is only ~1.35x. A byte-aligned code (1 byte/weight = 8 bpw, 3.8x more bytes) would be DRAM-BOUND and ~2.8x SLOWER, not faster. So the "spend spare bandwidth" framing is refuted; the ONLY live route is FEWER per-weight instructions at the SAME 2.125 bpw. Analysis (B3 comment): exact-from-sign-planes with bf16 activations is mc-class (per-weight select+add, no popcount collapse without bitsliced a); the one untested hope is a cheaper 2-bit->ternary MAP inside the existing mv. Decision gate before building: a clean gpudebug counter capture of the isolated mv (which=mv) to localize whether the extract or the map is the issue-limited hotspot — do NOT build blind; the exact reformulations analyzed so far are all mc-class. LOW priority vs the acceptance axis.
