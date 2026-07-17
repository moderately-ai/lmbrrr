---
id: deltanet-fused-chunk-long-prefill
title: "Fused DeltaNet chunk for long prefill (TTFT is linear at ~20 ms/token — the fused kernel is gated l<=12)"
status: todo
priority: p1
dependencies: []
related: [verify-spec-acceleration-routemap, optimize-deltanet-chunked-prefill-and-verify-throughput, gdn-rollback-free-masked-solve]
scopes: [runtime/candle, runtime/metal]
shared_scopes: []
paths: []
tags: [route-map, prefill, ttft, kernel, deltanet]
---
NEW LEVER (2026-07-17), the prefill/TTFT axis the campaign names but had not examined. FULLY TRIANGULATED:

- **Black-box (M3, measured):** prefill is LINEAR at ~19.6 ms/token with ZERO amortization — 89 tok 1.83s, 160 tok 3.07s, 232 tok 4.77s, 304 tok 6.02s, 448 tok 8.64s, 592 tok 11.6s (per-token flat 19.2-20.6 ms across the whole range). A weight-bound batched prefill would amortize the weight read and be near-flat in N; instead each token pays a fixed cost.
- **Source/algebra:** the fused gated-DeltaNet chunk kernels (v1 `forward_fused_chunk`, v2 `forward_fused_v2`) are BOTH gated to `(2..=12).contains(&l)` (qwen35.rs:1414/1539). For prefill l>=13 neither is eligible -> falls to the UNFUSED `recurrent_delta_rule_chunked` (CHUNK=32): materialized tensor ops (transposes, WY/UT Neumann-doubling solve, state matmuls) x ceil(l/32) chunks x ~60 DeltaNet layers. That is the linear per-token cost.
- **Intervention (moves the number):** forcing `LMBRRR_DELTANET_SEQUENTIAL=1` swings prefill 4.80 -> 31.68 s at 232 tok (6.6x) — changing ONLY the recurrence path, proving DeltaNet is the DOMINANT prefill term (MLP/attn are batched-cheap). Separately, `--warmup` (cold vs warm prefill identical, 1.07 vs 1.05 s) REFUTES the cold-shader-compile hypothesis: the cost is real repeatable compute, not JIT.

**The fix:** reuse the fast fused chunk kernel for prefill by looping it over <=12-token (or a new larger-cap) sub-chunks carrying conv_state+recurrent_state, instead of the tensor-path materialization. The chunk-to-chunk state chaining already exists (that IS how the spec verify chains rounds), so sub-chunked prefill is the same chaining within one forward. GATE: byte-parity vs the tensor-path chunked scan (the trusted reference) on a multi-length prefill, before trusting any TTFT number. RISK: conv depthwise-causal window carry across sub-chunks + capture assembly must match; the l<=12 cap likely reflects the kernel's threadgroup staging (can't hold l=232 in one dispatch) so host-side sub-chunk looping is the tractable route, not a single-dispatch long kernel.

EXPECTED: fused chunk ~1.8-2x+ the tensor path (uncertain until measured) -> 232-tok TTFT 4.8 -> ~2.5 s or better; scales to every prompt length. Compounds with nothing on the decode axis (pure TTFT win). Instrumentation shipped: `gguf decode --warmup` + `cold_prefill_seconds`.

RESULT 1 (2026-07-17, MEASURED, M3, LMBRRR_DELTANET_PREFILL_FUSED, opt-in): proj hoisted out of the loop (once, weight-amortized), fused conv+recurrence kernel looped over CAP=12 sub-chunks. **232-tok prefill 4.77 -> 3.76 s = 1.27x.** QUALITY-PRESERVING (not byte-exact): teacher-forced PPL of the tensor-path greedy ids is IDENTICAL under both prefills (tensor 3.9108 / mean_lp -1.36374 vs fused 3.9045 / -1.36213, 0.16% = rounding; min_lp matches to 0.03). The token divergence at gen-pos ~35 is an argmax-flip on a near-tie from CAP=12 vs the tensor path's CHUNK=32 rounding, NOT a logic bug. Component ceiling (profiled): recurrence 22% + conv 9% + gate 5% = ~36% of prefill; MLP 40% is already optimal (mm2d intervention: 4.77/4.80/6.28s for tile/mm2d/planar -> tile mm wins at m=128, ruled out). So 1.27x of a ~1.5x ceiling captured; remainder = dispatch overhead (20 sub-chunks) + small-chunk solve.
NEXT: (a) raise GDC_MAX_L past 12 (threadgroup budget allows ~15 at 32KB with the current f32 staging; matching CHUNK=32 needs a persistent kernel that walks the sequence without staging the whole chunk); (b) the persistent single-dispatch chunked-scan kernel is the true optimum (no host round-trips, state on-chip). Default stays opt-in until a multi-prompt quality gate + spec-loop check, then flip on (pure TTFT win, quality-preserving).
