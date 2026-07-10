---
id: bf16-activation-quantized-matmul-metal
title: BF16 activation quantized matmul on Metal
status: todo
priority: p1
dependencies: []
related: [quantize-full-text-decoder-q4-incl-lm-head, fix-runner-hot-path-naive-ops]
scopes: [runtime/metal, runtime/candle, candle-fork]
shared_scopes: [docs/research]
paths: [Cargo.toml, Cargo.lock, src/quantized_linear.rs, docs/research/bf16-qmatmul-metal.md]
tags: [kernels, quantization, campaign-1000, fork]
---
## Goal

Remove the F32 activation round-trip around every quantized matmul on Metal. candle's Metal mm path hard-asserts F32 activations (candle-core/src/quantized/metal.rs:390) and the mv path dispatches once per batch row; lmbrrr's `MixedLinear` casts BF16->F32->BF16 on every call. This work lands in the candle fork (~/workspace/github.com/huggingface/candle) and lmbrrr pins to the fork rev.

## Acceptance

- Fork: BF16-activation variants of the quantized mv and mm kernels (or an in-kernel cast) so `QMatMul::forward` accepts BF16 directly; a batched GEMV so M==1 with batch > 1 is one dispatch, not one per row.
- lmbrrr: drop `force_f32_input` when the fork path is active; pin candle via `git = ... rev = ...` (no vendored submodule), declare the `candle-fork` external scope on this ticket.
- Bench with `quant-matmul-bench` at the model's real shapes; report per-shape deltas and the end-to-end decode gain under the q4k-full-text policy.
- Stage gate (with quantize-full-text-decoder-q4-incl-lm-head + fusion tickets): >= 250 quantized forwards/s single-stream.
