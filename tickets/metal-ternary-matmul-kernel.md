---
id: metal-ternary-matmul-kernel
title: "FEATURE: Metal ternary matmul kernel (ternary weight x activation, BitNet-style)"
status: done
priority: p2
dependencies: [spike-ternary-type42-block-format, ternary-type42-dequant]
related: [ternary-bonsai-27b-support, eval-matmul2d-uint4b-tensor-op, bf16-activation-quantized-matmul-metal]
scopes: [candle-fork]
shared_scopes: []
paths: []
tags: [ternary-bonsai, model-compat, fork]
---
## WHY

The point of a ternary target is the byte diet: ~2 bpw weights vs q4k's ~4.5 → roughly half the decode bandwidth of our current MiniCPM stack, on a 27B model where bandwidth is the wall. Dequant-to-bf16-then-GEMV throws that away (materializes bf16, moves 8x the bytes). The win needs a kernel that consumes the packed ternary weights directly — a ternary GEMV/GEMM (weight ∈ {-1,0,+1} × bf16 activation, or int8-activation BitNet-style) — analogous to our existing `mm2d` uint4b tensor-op path, but for type 42.

## REFERENCE (located 2026-07-15; see `docs/research/ternary-type42-format.md`)

Port target: `~/workspace/github.com/PrismML-Eng/llama.cpp` (branch `prism`), `ggml/src/ggml-metal/ggml-metal.metal` — a Metal kernel for our exact GGUF format. Take `kernel_mul_mv_q2_0_f32_impl<nr0,nr1,tpb>` (l.3858): decode = nr1=1 (bit-decomposition, deterministic order via `q2_0_dot_y`), verify = nr1=2..4 (weights read ONCE, reused across draft columns — the bandwidth win). Arithmetic is just `d·(Σq·y − Σy)`, LSB-first unpack. `mul_mv_ext_q2_0` (l.4418) = the mlx qmv_wide wide-m fallback; `mul_mm_q2_0` (l.10758) = prefill GEMM (only for ne11≳32). This maps 1:1 onto our `mm2d` multi-column structure. (MLX fork = affine 2-bit, different scheme — reference only for the qmv_wide small-batch geometry.)

## WORK ITEMS

1. Design the ternary GEMV: unpack trits in-kernel, accumulate `sum(sign * act)` per output with the per-block scale, threadgroup-staged like the mm2d/`mul_mv` path. Decide activation precision (bf16 direct vs int8-quantized activations for a true BitNet int matmul) — measure both. Start from the prism `kernel_mul_mv_q2_0_f32_impl` template above.
2. Route it at the decode shapes of the 27B (hidden 5120, ffn 17408, qkv/gate) — the body/head split like the existing routing table.
3. Correctness gate: kernel output == the [[ternary-type42-dequant]] reference within the stub-oracle noise bound (reuse the mm2d oracle harness).
4. Bench vs dequant-to-bf16-GEMV and vs q4k on the same shapes; confirm the bandwidth win is realized in-loop (not just isolated latency — the split-K/t32 lesson: in-loop-arbitrated).

## DONE-WHEN

Ternary decode runs on Metal consuming packed type-42 weights, bit-close to the dequant reference, and beats dequant-to-bf16 on measured decode bandwidth for the 27B shapes. Candidate for upstreaming alongside the other fork kernels ([[upstream-fork-kernels]]).

## GEMV LANDED (2026-07-15) — candle fork `lmbrrr` branch (uncommitted)

Decode GEMV done + correctness-gated (work items 1&3 for nr1=1). NOT the bench (item 4) or the verify-width nr1>1 / prefill mm (deferred — decode is m=1).

- `candle-metal-kernels/src/metal_src/quantized.metal`: `block_q2_0` (matches candle_core `BlockQ2_0`, 34 B) + `q2_0_dot_y<SW>` (bit-decomposition `d·(acc_lo + 2·acc_hi − sumy)`, select-form) + `kernel_mul_mv_q2_0_impl_t<YT,DT>` + entrypoints `_f32`/`_bf16`/`_bf16_bf16`. Geometry: reuses Q8_0 dispatch (nth0=8,nth1=8,align=8 → 2 SG × N_DST=4 rows); each 128-code block split across tpb=8 threads (SW=16), `ix=tiisg/8`, `il=(tiisg%8)*16`, `ib += 4`.
- `candle-metal-kernels/src/kernels/quantized.rs`: `GgmlDType::Q2_0` + `bf16_src1/dst_supported` + Q8_0 (nth0,nth1,align) arm + 3 mv name arms; `Err` arms in `mm_t` + `get_rows` (no tile-mm / packed-embedding kernel yet).
- `candle-core/src/quantized/metal.rs`: `From` maps Q2_0 → metal Q2_0 (Q1_0 still panics).
- lmbrrr `src/quantized_linear.rs`: Q2_0 → `bf16_direct`.
- **Verify**: `test_matmul_q2_0_accuracy` (candle-core, metal, m=1) PASS — f32 path ~1e-6 rel, bf16_bf16 ~2e-3 (bf16 floor); n=515 exercises the row tail guard. `cargo check` clean both crates.

REMAINING: (a) the **prefill routing landmine** — `fwd()` sends Q2_0 m>1 to `call_quantized_matmul_mm_t` → the `Err` arm; the mv kernel already handles m>1 in one dispatch, so E2E must route Q2_0 through `fwd_mv` for all m (or port `mul_mm_q2_0`). (b) packed-embedding `get_rows_q2_0` (or dequant embed at load) for the loader. (c) the in-loop bandwidth bench vs dequant-bf16 / q4k on M3 (item 4). (d) push fork branch + bump the 4 candle rev pins.
