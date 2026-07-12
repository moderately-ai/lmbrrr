---
id: q4k-soa-plane-repack
title: SoA q4_K plane-split repack (device layout) — the floor-mover
status: todo
priority: p1
dependencies: []
related: [gemv-width-splitk-concurrency, certified-subvocab-head]
scopes: [candle-fork, runtime/metal]
shared_scopes: []
paths: []
tags: [kernels, jump2]
---
## Why (evidence, 2026-07-12)

The greedy floor (~215 tok/s bench / ~181 realized in-loop) is set by the m=1 q4_K mv kernel, and the per-element micro-opt class on that kernel is CLOSED: six falsifications this session (u32 unpack in mv, dot()-based accumulation, float4 y-loads, lane-wise float4 variants, NC_MV=8) all lost or tied against the shipped u16 four-mask form. The MSL spec review (recorded on `gemv-width-splitk-concurrency`) settled the diagnosis: the kernel is latency/occupancy-bound at ~119 GB/s effective vs 546 peak (22%), while q8_0 mv reaches 262 and dense bf16 320–358 on the same shapes. The structural cause is the q4_K wire layout: 144-byte AoS superblocks (2 half scales + 12 scale bytes + 128 nibble bytes, QK_K=256) straddle cache lines and force strided per-lane access that defeats coalescing. No instruction-schedule change fixes a layout problem — this is the only remaining lever that moves the floor, and the campaign's 1000 tok/s composition (Jump 2: floor 215 → 280–300) depends on it.

## Design

Repack at LOAD into SoA planes — device-side layout only, wire format unchanged (GGUF/quantize_onto artifacts stay bitwise-identical):

- Split each superblock into separate contiguous planes: scales plane (d/dmin halves + 12 scale bytes) and quants plane (128 nibble bytes), laid out so consecutive SIMD lanes read consecutive bytes within each plane across blocks of the same row segment.
- Kernel variant (`kernel_mul_mv_q4_K_soa` or a layout flag) reads the planes; host repack happens once in QMetalStorage / load path, cost amortized over the session.
- Start with mv (m=1) only — that is the floor; mc/mm variants follow only if the mv gate passes.

## Plan and gates

1. **Prerequisite — confirm the diagnosis with a GPU capture before writing any kernel**: wire `--capture` (MTLCaptureManager, pattern on fork branch `AddMetalGpuTraceToQuantized`) into `metal_benchmarks qmv`; inspect occupancy/limiter counters in Xcode. Deeper per-encoder counter infra exists on `feat/metal-profile-comprehensive` (per-dispatch timestamps unavailable on Apple Silicon — AtDispatchBoundary unsupported).
2. **Micro gate**: dispatch-level qmv bench (lm_head 248094 + n-sweep shapes) must show ≥1.5–2× effective GB/s on the SoA variant, else STOP and record the falsification.
3. **E2e per protocol**: verify_table → in-loop refit → rotated suite; oracles green.

## Fallback if falsified

q8_0 weights for the lm_head only (262 GB/s today, 2× bytes — net wash on paper, but it changes `certified-subvocab-head` gate math). Campaign consequence of falsification: ceiling ~750–850 tok/s structured-domain (Jump 1 only) — record explicitly, don't bury.
