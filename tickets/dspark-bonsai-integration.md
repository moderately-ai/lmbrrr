---
id: dspark-bonsai-integration
title: DSpark speculative decoding — Ternary-Bonsai-27B integration
status: in-progress
priority: p1
dependencies: []
related: [ternary-bonsai-27b-support, ternary-decode-profile-optimize, gguf-loader-qwen35-hybrid, verify-spec-acceleration-routemap]
scopes: [inference/speculative, runtime/candle]
shared_scopes: [docs/research]
paths: [src/dspark.rs, src/commands/gguf_run.rs, docs/research/dspark-verify-weightbound-gemm.md]
tags: [dspark, bonsai, spec-decode, metal]
---

# DSpark speculative decoding — Ternary-Bonsai-27B integration

Wire the shipped Bonsai DSpark drafter into lmbrrr's (already complete, Qwen35-native) spec-decode loop so `gguf spec` runs speculative decode on the ternary Bonsai target. NOT blocked — the drafter is shipped.

## Artifacts (on M3 `~/models/Ternary-Bonsai-27B/`)
- Target: `Ternary-Bonsai-27B-Q2_0.gguf` (7.17 GB, ternary Q2_0_g128, 64-layer Qwen3.6 hybrid).
- Drafter: `Ternary-Bonsai-27B-dspark-Q4_1.gguf` (1.95 GB, default) + `-dspark-bf16.gguf` (7.29 GB, ref). Both downloaded.
- Reference forward: prism `PrismML-Eng/llama.cpp` branch **`prism`**, `src/models/dspark.cpp` (+ `src/llama-arch.cpp` for KV/tensor names). Consult for exact semantics.

## Drafter config (from `dspark-Q4_1.gguf` metadata, arch=`dspark`)
- 6 backbone layers, hidden 5120, ffn 5120, heads 40 / kv 4, head_dim 128, rope_base 1e7, rms_eps 1e-6, vocab 248320.
- `block_size = 4`, `mask_token_id = 248319`, `target_layers = [1,16,31,46,61]` (5 taps), `markov_rank = 256`.
- `confidence_head = confidence_head_with_markov = true`. **`log_snr_conditioning = true`, min/max = -9/+9.**

## Drafter tensors (79 total)
- Backbone `blk.{0..5}`: `attn_{q,k,v,output}.weight` (Q4_1; q [5120,5120], k/v [5120,512]), `attn_{q,k}_norm` [128] F32, `attn_norm`/`ffn_norm` [5120] F32, `ffn_{gate,up,down}` [5120,5120] Q4_1.
- `token_embd.weight` [5120,248320] **Q2_0**, `output.weight` [5120,248320] Q4_1, `output_norm.weight` F32.
- `dspark.fc.weight` [25600,5120] Q4_1 (fuses 5×5120 target taps → 5120), `dspark.hidden_norm.weight` [5120] F32.
- `dspark.markov_head_a.weight` [256,248320] BF16, `dspark.markov_head_b.weight` [256,248320] Q4_1 — NOTE `[rank,vocab]` (transposed vs the safetensors port's `[vocab,rank]` markov_w1/w2).
- `dspark.confidence_head.weight` [5376,1] Q4_1 (+bias [1]) — features = [hidden 5120 ; markov_rank 256].
- `dspark.log_snr_fc1.{weight [128,5120],bias [5120]}` BF16, `dspark.log_snr_fc2.{weight [5120,5120],bias [5120]}` BF16.

## Log-SNR conditioning (the port's ONE modeling gap — src/dspark.cpp:245-295)
Added to the draft-block embedding BEFORE the layer loop. Per draft position `pos` (n_draft = block_size-aligned):
- `log_snr = (pos % block_size == 0) ? max_snr : min_snr` (anchor row = max, masked = min).
- `t = (log_snr - min)/(max - min) * 1000`.
- 128-dim sinusoidal: `half=64; freq_i = exp(-ln(10000)*i/half); feat[pos, i]=sin(t*freq_i); feat[pos, half+i]=cos(t*freq_i)`.
- `snr = fc2( silu( fc1(feat) + b1 ) ) + b2`  → `inpL += snr`.

## Work items
1. **DONE — GGUF drafter loader** (`DsparkDrafter::load_gguf`, src/dspark.rs): config from metadata, name-map, markov_head_a/b `[rank,vocab]` handled, fused-quantized backbone (byte-cat qkv/gate_up), packed Q2_0 embed (`EmbedTable::Packed`/`PackedEmbed`), lm_head kept Q4_1.
2. **DONE — Log-SNR conditioning** in `propose_backbone` (+ DsparkConfig fields; `compute_log_snr_bias`), the formula above.
3. **DONE — block_size = 4** flows from config (gamma ≤ block_size).
4. **DONE — Target-load seam**: `Qwen35CausalLM` got the spec-target methods (set_device_capture/take_device_captures/set_verify_state_capture/snapshot/restore/rollback_to_prefix/forward_all_logits) + `TokenEmbedding::Packed` (−1.84 GB, the fix for the memory-pressure residual-collapse bug); a focused `spec_decode` loop (`gguf spec`) uses readvance rollback. Output byte-correct vs plain decode, mean_accepted 3.4–4.0/4.
5. **IN PROGRESS — Q2_0 weight-bound verify GEMM (toolchain gate CLEARED 2026-07-16).** This is the whole speedup story. Verify = 81% of round time ⇒ spec sits at parity (~11–12.5 tok/s vs 14.68 plain), not 2–3×. Five hand-rolled kernel variants exhausted (mc compute-bound, tile-mm padding-waste, from-scratch + planar read/staging-bound at ≤38 GB/s); the ceiling was the software 2-bit unpack. The hardware fix is `tensor_ops::matmul2d` with a 2-bit B operand (`uint2b_format`) — which is Metal 4.1. **This was gated on Xcode 26.6's toolchain (`32023.883`, no 4.1) and is now UNBLOCKED on Xcode 27 beta 3 (`metalfe-32023.918.1`):** `-std=metal4.1` + `uint2b_format` + a `matmul2d` 2-bit-B kernel all compile+link on the M3 (verified — 11 KB test metallib). Building the real `mm2d_q2_0` now: mirror the q4_K mm2d stack (`q2_0_mm2d_planes` planes already written; `mm2d_q2_0.metal` with the ternary fold `d·(P − rowsum)`; `mm2d_q2_0.metallib` via `-std=metal4.1`; `Source::Mm2dQ2_0` + `call_quantized_matmul_mm2d_q2_0`; route verify m∈2..8). Full analysis + mirror steps + the toolchain timeline/meta-lesson: **`docs/research/dspark-verify-weightbound-gemm.md`**.
6. **DONE — e2e** (correct + at-parity) + throughput/breakdown measured. The 2–3× is item 5, now unblocked and in progress (native 2-bit `matmul2d` on the M3).

See `docs/research/dspark-verify-weightbound-gemm.md` and memory [[dspark-bonsai-drafter-shipped]], [[dspark-bonsai-e2e-working]], [[ternary-q2_0-gemv-exhausted]]. Progress log: `.comments/`.
