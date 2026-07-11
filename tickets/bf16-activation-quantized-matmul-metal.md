---
id: bf16-activation-quantized-matmul-metal
title: BF16 activation quantized matmul on Metal
status: done
priority: p1
dependencies: []
related: [quantize-full-text-decoder-q4-incl-lm-head, cut-drafter-propose-cost, batched-multi-stream-decode-runner]
scopes: [runtime/metal, runtime/candle, candle-fork]
shared_scopes: [docs/research]
paths: [Cargo.toml, Cargo.lock, src/quantized_linear.rs, docs/research/bf16-qmatmul-metal.md]
tags: [kernels, quantization, campaign-1000, fork]
---
## Micro-bench observation (2026-07-11, TREAT WITH CAUTION)

quant-matmul-bench lm_head rows scale LINEARLY in m across dense AND all quant tiers (dense: 2.87/5.59/11.19/23.12 ms at m=1/2/4/8) — consistent with per-row weight re-reads — BUT the absolute numbers contradict end-to-end facts (lm_head alone at m=8 "costs" more than the whole 15.7ms verify forward), so the micro-bench conditions (per-iteration sync, cold pool) inflate and the linear pattern may be sync-amplified. Per the measurement protocol, only same-session end-to-end A/Bs decide. Next investigation step when this ticket is picked up: Instruments GPU capture of one verify chunk (the only tool that attributes GPU time truthfully here), THEN kernel work.

## Progress (2026-07-10 night, fork ec0f74e5)

CORRECTED ATTRIBUTION (post-hoc routing read): fwd_mv is only reached at m==1 (metal.rs:397 routes src dim Minus2==1), so the ne11=m single-dispatch change is currently DEAD CODE via QMatMul - and the mv grid (height=m) re-reads the weight per row regardless, so it saves dispatch overhead only, never bandwidth. The 115.9 tok/s gain re-attributes to the scheduler x gamma4 x q8 operating point (lm_head m<=4 through the mm path; markov steps m=1 mv). REAL remaining scope, one package: (1) BF16-src1 variants for mv AND mm; (2) a genuinely weight-shared small-m quantized matmul (same problem class as the dense skinny-gemm: the tile mm at m=2-8 and the mv grid both waste bandwidth); (3) lift the m routing once (2) exists. gamma8+q8 draft 15.7ms is the mm path measured at m=8.

## Board revision (2026-07-10 evening, agent-verified)

Quantified: under q4k-full-text the F32 round-trip is ~374 extra dispatches/token across 187 quantized projections ~= 0.8 ms (~11% of the 7ms token) — worth about as much as quantizing the whole attention stack. The per-row mv dispatch loop (metal.rs:324-338; every mv kernel is a *_f32 variant, kernels/quantized.rs:127-141) makes this ticket a HARD PREREQUISITE for the aggregate lane: 8-stream quantized decode would pay ~1500 gemv dispatches/step without the batched (ne11=m) single-dispatch fix. New consumer: cut-drafter-propose-cost (quantized drafter heads eat the same cast tax). Stage gate re-derived: >= 250 forwards/s is reachable but edge — depends on the Q4K mv kernel sustaining ~250 GB/s at these shapes; re-bench quant-matmul-bench (post-gemv-routing) before promising, and custom mv tiling is the contingency fork work.

## Goal

Remove the F32 activation round-trip around every quantized matmul on Metal. candle's Metal mm path hard-asserts F32 activations (candle-core/src/quantized/metal.rs:390) and the mv path dispatches once per batch row; lmbrrr's `MixedLinear` casts BF16->F32->BF16 on every call. This work lands in the candle fork (~/workspace/github.com/huggingface/candle) and lmbrrr pins to the fork rev.

## Acceptance

- Fork: BF16-activation variants of the quantized mv and mm kernels (or an in-kernel cast) so `QMatMul::forward` accepts BF16 directly; a batched GEMV so M==1 with batch > 1 is one dispatch, not one per row.
- lmbrrr: drop `force_f32_input` when the fork path is active; pin candle via `git = ... rev = ...` (no vendored submodule), declare the `candle-fork` external scope on this ticket.
- Bench with `quant-matmul-bench` at the model's real shapes; report per-shape deltas and the end-to-end decode gain under the q4k-full-text policy.
- Stage gate (with quantize-full-text-decoder-q4-incl-lm-head + fusion tickets): >= 250 quantized forwards/s single-stream.
