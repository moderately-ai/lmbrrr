---
id: fix-runner-hot-path-naive-ops
title: Fix runner hot path naive ops
status: in-progress
priority: p1
dependencies: []
related: []
scopes: [runtime/candle]
shared_scopes: [docs/research]
paths: []
tags: [performance]
claimed_from: todo
assignee: claude
lease_expires_at: 1783704428
---
## Goal

Clean up the per-token/per-layer naive ops found in the 2026-07-10 full-source audit that burn Metal dispatches on every decode step, each gated by strict logits parity and a no-regression bench.

## Update (decode audit, 2026-07-10, pass 3 scope)

Fresh-agent audit quantified the remaining items. RMSNorm: the hand-rolled `Qwen35RmsNorm::forward` (qwen35.rs:168-175) is 9 dispatches and there are 79 norm applications per token (48 layer + 18 DeltaNet group + 12 q/k + final) -> ~710 dispatches/token, the single largest bucket; switching to `candle_nn::ops::rms_norm` (1 dispatch) saves ~640 — weight must be stored BF16 and q/k-norm inputs (narrows of q_gate) need one `.contiguous()` each. SDPA-vector confirmed drop-in for decode (ops.rs:1078-1093): head_dim 256, q_seq==1, native GQA (no repeat_kv), strided k/v (feed the cache narrow directly — no k_t transpose/contiguous), auto 2-pass at k_seq>=1024; fold 1/sqrt(d) into the sdpa scale arg. Current cost eliminated: repeat_kv = 4 whole-cache copy2d per K and V + transpose-contiguous ~= 12 MB/layer at 1k ctx (~72 MB/token, linear in context) + F32 softmax cast pair. Residual after SDPA: with partial_rotary_factor<1 the rotary cat branch (qwen35.rs:216-223) still does 2 cats + narrows per q/k per layer (~25 dispatches/token) — rotate in place or passthrough kernel. New sibling tickets from the same audit: fix-metal-buffer-pool-purge-and-residency-commits, remove-per-token-decode-synchronize, keep-deltanet-recurrent-state-f32, precompute-deltanet-conv-taps, narrow-prefill-hidden-before-lm-head, two-stage-argmax-device-sampling, reduce-metal-dispatch-layer-overheads.

## Progress (2026-07-10, pass 2)

- GEMV routing landed in the fork (rev ff3666ec): candle's Metal backend sent every matmul to the mlx GEMM tile kernel; call_mlx_gemv existed fully implemented but was never dispatched and not even exported. Now m==1/n==1 matmuls route to gemv with gemm fallback on incompatible strides. Validated in-repo (45/45 Metal matmul+cast tests) and in lmbrrr: thermally-controlled interleaved A/B shows +5.7% steady decode (60.0 -> 63.4 tok/s), strict parity clean.
- Measurement lesson recorded: the earlier per-shape roofline anomaly (MLP GEMV at 83 GB/s) was substantially queue/thermal variance — cross-session micro-bench comparisons on this machine mislead; use same-session interleaved end-to-end A/B for all kernel work.
- Remaining pass-2 items: GQA regrouping (reshape Q to [b, kv_heads, groups*l, d] instead of repeat_kv + full-cache k_t contiguous per token — note the win grows with context length, so bench at long contexts), SDPA switch, MixedLinear F32 round-trip.

## Progress (2026-07-10)

Pass 1 landed (commit 8f7d14b): RMSNorm F32 weight cache with pre-applied zero-centered +1, on-device causal mask via log(tril2) replacing the O(seq*total) CPU build, quant-bench prefill tokens/s accounting fixed. Strict logits parity passes; decode 65.5 tok/s (unchanged, as expected — these were small dispatches), prefill ~805 tok/s. Remaining below: SDPA switch, k_t cache-copy + repeat_kv elimination, MixedLinear F32 round-trip, and the MLP GEMV tiling investigation — these carry the real tok/s and likely end in fork kernels.

## Acceptance

- `Qwen35RmsNorm`: cache the F32 weight (with the zero-centered +1.0 pre-applied) at load instead of converting per call, and evaluate `candle_nn::ops::rms_norm` for the fused path — same class of fix as the cached `dt_bias_f32`/`a_log_exp_f32` constants.
- Cache/reuse the causal mask instead of rebuilding an O(seq * total) CPU Vec and re-uploading it on every prefill/verify chunk.
- `FullAttention`: eliminate the per-step full-cache `k_t` transpose+contiguous copy and the `repeat_kv` materialization where possible (broadcast GQA or Candle SDPA), keeping the softmax numerics parity-gated.
- `MixedLinear`: quantify and, if material, remove the double F32 activation cast around `QMatMul` on Metal (fork lane if the kernel only accepts F32).
- Fix `quant-matmul-bench` prefill throughput accounting: the `tokens_per_second` field divides iterations (not iterations * chunk_tokens) by elapsed, understating prefill_mm throughput by the chunk size.
- Investigate the measured MLP GEMV underutilization (3584x1024 runs at ~83 GB/s vs ~350 achievable, ~4.7 ms/token on the table per docs/research/metal-roofline-and-dispatch-overhead.md): profile candle's Metal gemv/gemm dispatch for these shapes and either route to a better kernel (mlx gemv variants) or fix tiling in the fork.
- Re-run the standard bench matrix and logits parity after each change; document results in docs/research.
