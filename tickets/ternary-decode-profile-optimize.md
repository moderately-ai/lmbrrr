---
id: ternary-decode-profile-optimize
title: "PERF: profile + optimize the ternary Bonsai-27B decode toward > 13.7 tok/s"
status: in-progress
priority: p1
dependencies: [gguf-loader-qwen35-hybrid, causal-text-model-generic-decode]
related: [ternary-bonsai-27b-support, metal-ternary-matmul-kernel, optimize-deltanet-metal-decode, design-ternary-bonsai-e2e-bringup]
scopes: [runtime/candle, candle-fork]
shared_scopes: []
paths: [src/commands/gguf_run.rs, src/qwen35.rs]
tags: [ternary-bonsai, perf]
---
## GOAL

Ternary target decode faster than the llama.cpp reference (13.7 tok/s, M3 Pro, same GGUF). Iterate data-driven; every number from the M3 referee.

## BASELINE (2026-07-15, M3, release, `gguf-run` naive greedy decode)

- **4.98 tok/s** (200 ms/token) vs llama.cpp **13.7** → at ~36% of the bar.
- prefill 9–15 tok/s; load 7 s.

## EVIDENCE (measured, not assumed)

1. **Host path is negligible.** fwd/head split: argmax + full-logits readback = **0.35 ms/token**; the entire 200 ms is the model forward. (So fused-argmax / async-readback / device-chain — the MiniCPM fast paths — buy ~nothing here. Do NOT start there.)
2. **The Q2_0 GEMV kernel is fine.** Isolated micro-bench (`--bench-gemv`): Q2_0 17408×5120 = **93.3 GB/s (~62% of the M3's ~150 peak)**, Q4K = 137.6 GB/s (~90%). Q2_0 reads fewer bytes so it's faster per call (0.254 vs 0.364 ms). The kernel is not the bottleneck (some headroom vs Q4K's tuning, but not the 2.5× gap).
3. **Decode achieves only ~24% of memory bandwidth** (7.17 GB / 200 ms = 36 GB/s) — vs MiniCPM decode at ~60%.
4. **GPU-trace (Metal System Trace, xctrace 16.0): decode is 99% GPU-BUSY** — GPU-bound, NOT host/dispatch-bound. The 2.5× is inside the forward, on-GPU.
5. **~73% of each layer's GPU time is non-matmul.** Per-layer command buffer ≈ 4 ms; the layer's big matmuls are ~1 ms (bandwidth-bound, efficient); the other ~3 ms is the **DeltaNet recurrent step (48/64 layers) + attention + elementwise (sigmoid gate / rmsnorm / swiglu)** — many small low-occupancy dispatches that keep the GPU "busy" at low utilization. ~160 command buffers/token; `sigmoid` also appears as 132 separate tiny dispatches/decode.

## HYPOTHESIS / LEVERS (ranked by the evidence)

1. **DeltaNet decode recurrent step is the prime lever** (48/64 layers, ~3 ms/layer of low-occupancy compute). See [[optimize-deltanet-metal-decode]] / the fused-deltanet-decode kernel — is `forward_fused_decode` actually the path here, and is it occupancy-bound at 27B head counts (48 v-heads × 128×128 state)?
2. **Elementwise dispatch fusion** — sigmoid/rmsnorm/swiglu as separate small kernels; the campaign's metal-elementwise-fusion applies.
3. **Q2_0 GEMV → close the gap to Q4K's 90%** (secondary; ~1 ms/layer only).

## PER-OP PROFILE (2026-07-15, M3, `--profile`, 24 decode steps, per-op-sync inflates absolute to 309 ms but the split is real)

| component | ms/token | pct |
|---|---|---|
| **deltanet_recurrent_rule** | **142.5** | **46.1** |
| mlp | 61.2 | 19.8 |
| deltanet_qkv_projection | 20.1 | 6.5 |
| deltanet_output_gate_norm_projection | 17.0 | 5.5 |
| mlp_residual_next_norm / post_attn_norm / gates / conv | ~50 | ~16 |
| full_attention_* (16 layers) | ~18 | ~6 |

## ROOT CAUSE (found 2026-07-15) — GQA DeltaNet falls off the fused decode path

`deltanet_recurrent_rule` (46%) is the UNFUSED candle-op path (qwen35.rs ~1890). Decode is supposed to hit `forward_fused_decode` (the `fused_deltanet.rs` kernel), but `fused_decode_eligible` (and `fused_chunk`/`fused_v2`) require **`num_k_heads == num_v_heads`** (line 1402). Bonsai's DeltaNet is **GQA-style: `num_k_heads=16` (ssm.group_count), `num_v_heads=48` (ssm.time_step_rank)** — so it fails the check and falls through to the slow recurrent_delta_rule. The unfused path handles the mismatch via `maybe_repeat_heads` (k/q repeated 16→48); the fused kernel assumes equal heads (built for MiniCPM, which has them equal).

**FIX (decided 2026-07-15): GQA-aware fused v2 DECODE kernel, decode-only, no bloat.** Prefill (l=33 > chunk limit 12) is unfused regardless, so ONLY `gated_delta_v2_decode_bf16` needs changing. In it, grid is one threadgroup per VALUE-head `h`; q/k are read at `h*dk` / `key_dim+h*dk` (lines 309-310) and their conv-state written at 385. Add a `num_k_heads` param; map `kh = h*num_k_heads/heads` for the q/k channel reads + conv-state writes ONLY (v, gates b/a, decay, and the recurrent state stay per-v-head `h`). Guard the q/k conv-out write to the first v-head of each k-group (`h % (heads/num_k_heads) == 0`) to keep single-writer. conv_state stays raw (10240), state stays [48,dk,dv] (unfused-prefill→v2-decode handoff via `take_state_for_v2`, the proven MiniCPM path). Relax `fused_decode_eligible` `num_k_heads == num_v_heads` → `num_v_heads % num_k_heads == 0`. Files: candle fork `gated_delta_v2.metal` (decode kernel) + `kernels/gated_delta.rs` (pass num_k_heads) + `fused_deltanet.rs`; lmbrrr `qwen35.rs` (eligibility + thread num_k_heads). Verify: output stays coherent + decode tok/s.

## XCTRACE METHODOLOGY (macOS 27 / Xcode 26.6 CLI — reusable)

Programmatic per-dispatch GPU timing, replay-free (replay inflates tiny dispatches 5–10×):
- Capture: `xcrun xctrace record --template "Metal System Trace" --output X.trace --launch -- <binary+args>` (runs to process exit).
- TOC: `xctrace export --input X.trace --toc` → schemas. Key one: **`metal-gpu-intervals`** (start, duration, gpu-channel-name, event-depth, formatted-label, process).
- Export: `xctrace export --input X.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]'` → XML (id/ref dedup; resolve `ref` → the earlier `id`'s `fmt`).
- Parse (python ElementTree): filter `process contains lmbrrr` + `gpu-channel-name == Compute`; first `<duration>` per row is the GPU interval; rows whose label starts `Command Buffer` are the depth-0 buffer spans (serial on candle's one queue → their union = GPU-busy time). Bucket by start-time to separate load / prefill / decode. candle labels command buffers by their elementwise ops (`rmsnorm & swiglu`), so per-kernel matmul-vs-deltanet split needs the software profiler, not the trace.
