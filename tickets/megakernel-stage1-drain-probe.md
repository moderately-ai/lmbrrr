---
id: megakernel-stage1-drain-probe
title: "EVAL: Stage-1 dispatch-boundary drain probe (single-threadgroup chained loop vs N real dispatches)"
status: todo
priority: p2
dependencies: []
related: []
scopes: [candle-fork]
shared_scopes: []
paths: []
tags: [eval-wave, kernels]
---
WHY: the megakernel agent's verdict — spin-barrier persistent kernels are unsafe on Apple (OBE violation proven by Apple's own CAS test, Sorensen OOPSLA 2021), but the GPU-side inter-dispatch drain in our 340-dispatch dependency chain is real, ecosystem-unaddressed, and partially recoverable (~0.15-0.3ms/token) via starvation-free patterns. BEFORE building any of that, this zero-risk probe sizes the true per-boundary cost. It needs NO cross-threadgroup sync (fully spec-safe).

DESIGN: compare (A) N chained DEPENDENT small dispatches (each reads the previous output — the auto-barrier fires between each, exactly our decode pattern) vs (B) ONE dispatch, ONE threadgroup (1024 threads), computing the same N steps in an internal loop with threadgroup_barrier between steps (intra-TG barrier only — legal). Use a small dense f32 GEMV as the step op: y_{t+1} = W_t * y_t with W = 1024x1024 f32 (fits: each of 1024 threads owns one output row... one TG of 1024 threads, each thread dot-products one row = 1024 muls; loop N times over N distinct weight buffers so memory traffic matches (A)). N sweep: {8, 32, 128, 340}.

PROCEDURE:
1. Add `drain-probe` task to candle-metal-kernels/examples/metal_benchmarks.rs (copy the nsg-sweep harness structure: warmup 3, MIN_DUR 1.5s, many outer reps).
2. Variant A: N sequential call-style dispatches of a 1-TG GEMV kernel in one command buffer (hazard on the ping-pong y buffers serializes them — same as production).
3. Variant B: one kernel `chained_gemv_loop` taking an array of N weight buffer offsets (single concatenated weight buffer + stride), looping internally with threadgroup_barrier(mem_flags::mem_device) between steps.
4. GATE: outputs of A and B bitwise-equal (same arithmetic order per row).
5. METRIC: (time_A - time_B) / N = per-boundary drain+ramp cost in us, per the ambient-control protocol (within-session only). Also record time_B/N vs the theoretical single-step time to see the irreducible memory-ramp component.
6. DECISION: per-boundary cost >= 3us -> the chain's ~340 boundaries cost >= 1ms/token and starvation-free partial fusion (metal-icb-decode-replay ticket, re-scoped) gets promoted to p1 with the FidelityFX last-finisher / decoupled-fallback design. < 1us -> the drain prize is < 0.34ms; record and leave fusion-by-hand (elementwise ticket) as the only comb attack.
CAVEAT for the executor: variant B's single TG uses 1 of 40 GPU cores — its ABSOLUTE time will be slow; only the PER-STEP DELTA between A and B measures boundary cost. Do not read B's throughput as a megakernel projection.
