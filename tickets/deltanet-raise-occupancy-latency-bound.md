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

THE CAP: occupancy is limited by (1) threadgroup memory — 4x [GDC_MAX_L*128] f32 stages (k_sh/q_sh/v_sh/delta_sh) ~= 24KB + kk_sh/qk_sh [L*L] -> ~26KB total, so only ~1 threadgroup/core fits on the M3's 32KB; and (2) per-thread register pressure (ks0/qs0/decay_j[L] + the unrolled loops). Raising occupancy = fitting MORE threadgroups/core to hide the barrier latency.

LEVERS TO TEST (measure occupancy + wall each, one variable, gpudebug counters as the gate — wall-clock alone mis-diagnosed this 3x): (a) bf16 the [L*128] stages (k_sh/q_sh/v_sh/delta_sh) -> ~12KB -> ~2 tg/core; GATE the WY-solve precision (the delta forward-substitution accumulates — may need f32 for delta_sh; test parity). (b) drop kk_sh/qk_sh staging (recompute the k.k / q.k dots) trading ALU (only 6% used!) for threadgroup memory — ALU is FREE here, so recompute is likely a pure occupancy win. (c) reduce the register arrays / split work so maxTPT rises. (d) smaller GDC_MAX_L for verify (l<=8 never needs 12) -> less tg-mem for the hot verify instantiation specifically. Expected: latency-bound kernels gain ~linearly with occupancy up to the latency-hiding point; 8%->16% could be a large DeltaNet speedup. Applies to gated_delta_chunk (verify), gated_delta_v2, AND gated_delta_prefill.
