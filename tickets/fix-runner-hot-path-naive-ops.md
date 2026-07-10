---
id: fix-runner-hot-path-naive-ops
title: Fix runner hot path naive ops
status: todo
priority: p2
dependencies: []
related: []
scopes: [runtime/candle]
shared_scopes: [docs/research]
paths: []
tags: [performance]
---
## Goal

Clean up the per-token/per-layer naive ops found in the 2026-07-10 full-source audit that burn Metal dispatches on every decode step, each gated by strict logits parity and a no-regression bench.

## Acceptance

- `Qwen35RmsNorm`: cache the F32 weight (with the zero-centered +1.0 pre-applied) at load instead of converting per call, and evaluate `candle_nn::ops::rms_norm` for the fused path — same class of fix as the cached `dt_bias_f32`/`a_log_exp_f32` constants.
- Cache/reuse the causal mask instead of rebuilding an O(seq * total) CPU Vec and re-uploading it on every prefill/verify chunk.
- `FullAttention`: eliminate the per-step full-cache `k_t` transpose+contiguous copy and the `repeat_kv` materialization where possible (broadcast GQA or Candle SDPA), keeping the softmax numerics parity-gated.
- `MixedLinear`: quantify and, if material, remove the double F32 activation cast around `QMatMul` on Metal (fork lane if the kernel only accepts F32).
- Fix `quant-matmul-bench` prefill throughput accounting: the `tokens_per_second` field divides iterations (not iterations * chunk_tokens) by elapsed, understating prefill_mm throughput by the chunk size.
- Re-run the standard bench matrix and logits parity after each change; document results in docs/research.
