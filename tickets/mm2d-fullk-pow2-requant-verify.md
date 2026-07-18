---
id: mm2d-fullk-pow2-requant-verify
title: "mm2d verify: fold-free fullk GEMM on a power-of-two-scale (ue8m0) requant — the only realized lever past the instruction-issue wall"
status: todo
priority: p1
dependencies: []
related: [verify-spec-acceleration-routemap, eval-matmul2d-uint4b-tensor-op, dequant-bf16-dense-gemm-verify, bitplane-popcount-twotier-verify]
scopes: [quantization, runtime/metal]
shared_scopes: []
paths: [candle-metal-kernels/src/metal_src/mm2d_q2_0.metal]
tags: [route-map, kernel, verify, mm2d, requant]
---
Child of the verify-wall (routemap bucket B). The mm2d verify kernel is the DOMINANT hot path (~84% of the spec round). Its bottleneck class is NAMED + PROVEN (2026-07-17, gpudebug, m=5, M3 Pro): **instruction-issue-bound (instruction_throughput_limiter 70.95%) at 54.32 GB/s = only ~36% of the 150 GB/s M3 Pro peak** — NOT bandwidth-bound, large BW headroom. Instruction split: f32 FMA is the 50.21% limiter but only 40% utilized (STALLED, not saturated); address_generation 36.93% limiter / 12.59% util; integer unpack 35.55% + 30.13%. So removing per-K-tile instruction overhead lets the FMA climb toward the ~150 GB/s roof.

WHAT'S ALREADY DECIDED (measured, do not re-litigate): (1) strip-probe probe_fullk (NO K-loop at all) = -15-18% isolated (~50-53 GB/s), so the K-loop carries real removable cost. (2) The removable cost is the FOLD (per-K-tile scale multiply `d*p - d*rowsum`) + scale-loads + per-tile cooperative-tensor realloc — NOT the index arithmetic: the cheap byte-exact index-hoist was BUILT and REFUTED (2026-07-17, 5-round rotated spec A/B = +0.08% tok/s, byte-parity identical; reverted candle cd2499cc). So the ONLY path past 54 GB/s here is making the FOLD free, which requires power-of-two (ue8m0, 8-bit-exponent/0-mantissa) per-32-K scales so the fold is a bit-shift/exponent-add instead of a multiply.

CONCRETE LEVER, ORDERED (each step gates the next; STOP if a gate fails):
- STEP 0 (DECISIVE GATE, cheap, DO FIRST): quantify the PPL/quality cost of a power-of-two-scale requant BEFORE building any kernel or paying the whole-model requant. Round the Q2_0 per-128 block scales `d` to nearest 2^k at load (pessimistic proxy: the real requant is per-32, FINER -> less error, so if per-128-pow2 PPL is acceptable, per-32-pow2 is safe), run `gguf ppl` (non-planar mv path, no plane rebuild needed) pow2-on vs off, compare mean_kld / top1_agreement vs the existing q4k-head baseline gate. Implementation: an env-gated transform that rewrites each Q2_0 block's half `d` to its nearest power of two in the loaded QTensor bytes (propagates to mv; for the planar path the planes must be rebuilt from the rounded blocks). If PPL blows past the campaign's byte/PPL envelope -> fullk REFUTED for lossless verify (pow2 scales cost too much), fall back to a bounded-drift mode or close the lever. If PPL holds -> proceed.
- STEP 1: build the per-32-K ue8m0 requant of the model's verify weights (new plane artifact: codes unchanged, scales -> ue8m0 exponents per-32). Re-gate PPL at per-32 granularity (should beat the per-128 proxy).
- STEP 2: write the fold-free fullk mm2d kernel — process all K without the per-tile fold multiply (the exponent-add folds into the accumulate or a final scale), no per-tile cooperative-tensor realloc. Byte-exactness is BROKEN vs the arbitrary-scale path by construction; gate on PPL + a both-path (mv-decode / fullk-verify) consistency check, NOT bitwise.
- STEP 3: intervention-validate on the REAL spec (rotated A/B, planar) — the index-hoist refutation PROVES isolated-counter/probe wins do NOT always translate to spec, so probe_fullk's -15-18% is a CANDIDATE until a correct fold-free kernel moves the spec verify_seconds. Target ~+10-12% tok/s (routemap projection); adopt only if it clears.

WHY p1 but not started: STEP 0 is bounded but STEPS 1-3 are a multi-hour quality-gated build (whole-model requant + new kernel + PPL/consistency gates). This is the biggest single non-acceptance lever (84% of the round) but it is PRICED and its spec-translation is UNVALIDATED (per the hoist refutation). STEP 0 is the cheap decisive gate to run first. Full mechanism + the ue8m0 spec check are in docs/research/dspark-verify-weightbound-gemm.md.
