---
id: bf16-activation-quantized-matmul-metal
title: BF16 activation quantized matmul on Metal
status: todo
priority: p1
dependencies: []
related: [quantize-full-text-decoder-q4-incl-lm-head, cut-drafter-propose-cost, batched-multi-stream-decode-runner]
scopes: [runtime/metal, runtime/candle, candle-fork]
shared_scopes: [docs/research]
paths: [Cargo.toml, Cargo.lock, src/quantized_linear.rs, docs/research/bf16-qmatmul-metal.md]
tags: [kernels, quantization, campaign-1000, fork]
---
## Progress (2026-07-10 night, fork ec0f74e5)

Batched quantized mv LANDED: single dispatch with ne11=m for the quantized-block kernels (they address src1 by r1*ne10; the old per-row loop re-read the whole weight m times). Fork quantized suite green vs CPU. Measured: q8 drafter heads flip POSITIVE - gamma4+q8+scheduler = 115.9 tok/s math (0.78x), draft 4.9ms. Remaining scope: BF16-activation variants (the F32 cast tax stands), and the m>=8 route hits the mm path (F32 assert + slow at these shapes: gamma8+q8 draft still 15.7ms) - lift both together.

## Board revision (2026-07-10 evening, agent-verified)

Quantified: under q4k-full-text the F32 round-trip is ~374 extra dispatches/token across 187 quantized projections ~= 0.8 ms (~11% of the 7ms token) — worth about as much as quantizing the whole attention stack. The per-row mv dispatch loop (metal.rs:324-338; every mv kernel is a *_f32 variant, kernels/quantized.rs:127-141) makes this ticket a HARD PREREQUISITE for the aggregate lane: 8-stream quantized decode would pay ~1500 gemv dispatches/step without the batched (ne11=m) single-dispatch fix. New consumer: cut-drafter-propose-cost (quantized drafter heads eat the same cast tax). Stage gate re-derived: >= 250 forwards/s is reachable but edge — depends on the Q4K mv kernel sustaining ~250 GB/s at these shapes; re-bench quant-matmul-bench (post-gemv-routing) before promising, and custom mv tiling is the contingency fork work.

## Goal

Remove the F32 activation round-trip around every quantized matmul on Metal. candle's Metal mm path hard-asserts F32 activations (candle-core/src/quantized/metal.rs:390) and the mv path dispatches once per batch row; lmbrrr's `MixedLinear` casts BF16->F32->BF16 on every call. This work lands in the candle fork (~/workspace/github.com/huggingface/candle) and lmbrrr pins to the fork rev.

## Acceptance

- Fork: BF16-activation variants of the quantized mv and mm kernels (or an in-kernel cast) so `QMatMul::forward` accepts BF16 directly; a batched GEMV so M==1 with batch > 1 is one dispatch, not one per row.
- lmbrrr: drop `force_f32_input` when the fork path is active; pin candle via `git = ... rev = ...` (no vendored submodule), declare the `candle-fork` external scope on this ticket.
- Bench with `quant-matmul-bench` at the model's real shapes; report per-shape deltas and the end-to-end decode gain under the q4k-full-text policy.
- Stage gate (with quantize-full-text-decoder-q4-incl-lm-head + fusion tickets): >= 250 quantized forwards/s single-stream.
