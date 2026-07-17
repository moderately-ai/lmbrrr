---
id: deltanet-raise-occupancy-latency-bound
title: "Raise gated-DeltaNet kernel occupancy (8% -> higher): the recurrence is latency-bound, hits BOTH verify and prefill"
status: todo
priority: p1
dependencies: []
related: [deltanet-fused-chunk-long-prefill, verify-spec-acceleration-routemap, optimize-deltanet-metal-decode, fuse-deltanet-decode-step-kernel]
scopes: [runtime/metal]
shared_scopes: []
paths: []
tags: [route-map, kernel, deltanet, occupancy, verify]
---
WHITE-BOX FINDING (2026-07-17, gpudebug counters on isolated gated_delta kernels; details in deltanet-fused-chunk-long-prefill comments): the fused gated-DeltaNet chunk kernel (gated_delta_chunk_bf16 — the SAME kernel the spec VERIFY runs every round, l=5-8) is **latency-bound at ~8% kernel_occupancy**. xctrace: 100% GPU-busy at Max clocks (busy_frac 1.0, gaps ~1ms/13s) yet EVERY unit is idle-ish — alu_util 6%, gpu_bandwidth 2-13%, instruction_throughput ~16% — and occupancy_manager_target is ~100% (the GPU WANTS full occupancy but gets 8%). Signature = the serial per-position recurrence (threadgroup-barrier-chained across positions) can't be hidden at 8% occupancy, so the core spins/stalls. Low occ + manager 100% + low l1_evict => REGISTER / THREADGROUP-MEMORY capped, NOT a manager decision.

WHY IT MATTERS: verify is ~84% of the spec round; the DeltaNet chunk is ~27% of the verify step (mlp ~50%, the rest). So raising DeltaNet occupancy speeds the HOT spec path, not just prefill. This is a whole-pipeline lever, higher-value than the prefill-only streaming work that surfaced it.

THE CAP — CONFIRMED via pipeline-info (2026-07-17, `gguf profile-kernel --which pipeline-info`, no capture): gated_delta_chunk_bf16 AND gated_delta_prefill_bf16 both report **maxTPT = 1024 (the FULL max -> NOT register-limited) and staticThreadgroupMemoryLength = 25856 B (25.25KB)**. On the M3's 32KB/core that fits only ONE threadgroup/core (2x25.25=50.5>32) -> the 8% occupancy is PURELY threadgroup-memory-capped, not registers. The 4x [GDC_MAX_L*128] f32 stages (k_sh/q_sh/v_sh/delta_sh) = 24576 B dominate; kk_sh/qk_sh [L*L]*2 = 1152 B; rest small. To fit 2 tg/core need < 16KB (cut ~10KB); 3 tg/core need < ~10.6KB.

SIZING MATH (the exact lever): tgMem(L) ~= 4*L*128*4 + 2*L*L*4 + small = 2048*L + 8*L^2. L=12 -> 25.9KB (1 tg/core, 8%). L=8 -> 16.9KB (still 1/core — width-7 verify, l=8). **L=5 -> 10.5KB (3 tg/core, ~3x occupancy) — the CURRENT width-4 drafter verifies at l=5**, EXACT, no bf16. So sizing GDC_MAX_L to the ACTUAL verify width is a big, exact verify win for width-4; width-7 (l=8) needs an extra cut to break 16KB. FIX DESIGN: template the kernel on GDC_MAX_L, instantiate a few sizes (5/8/12), host picks the smallest >= verify chunk width per call. For PREFILL (quality-preserving, not byte-exact) additionally bf16 the [L*128] stages (-> ~13KB at L=12, 2 tg/core) — but GATE the WY-solve/delta precision. ALU is only 6% used, so trading threadgroup memory for recompute (drop kk_sh/qk_sh, recompute the k.k/q.k dots) is likely a pure occupancy win worth testing first (frees only ~1.1KB though — the [L*128] stages are the real target).

LEVERS TO TEST (measure occupancy + wall each, one variable, gpudebug counters as the gate — wall-clock alone mis-diagnosed this 3x): (a) bf16 the [L*128] stages (k_sh/q_sh/v_sh/delta_sh) -> ~12KB -> ~2 tg/core; GATE the WY-solve precision (the delta forward-substitution accumulates — may need f32 for delta_sh; test parity). (b) drop kk_sh/qk_sh staging (recompute the k.k / q.k dots) trading ALU (only 6% used!) for threadgroup memory — ALU is FREE here, so recompute is likely a pure occupancy win. (c) reduce the register arrays / split work so maxTPT rises. (d) smaller GDC_MAX_L for verify (l<=8 never needs 12) -> less tg-mem for the hot verify instantiation specifically. Expected: latency-bound kernels gain ~linearly with occupancy up to the latency-hiding point; 8%->16% could be a large DeltaNet speedup. Applies to gated_delta_chunk (verify), gated_delta_v2, AND gated_delta_prefill.
