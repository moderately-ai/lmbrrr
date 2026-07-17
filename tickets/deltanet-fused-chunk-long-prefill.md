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
