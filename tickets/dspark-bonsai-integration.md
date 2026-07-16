# DSpark speculative decoding — Ternary-Bonsai-27B integration

Wire the shipped Bonsai DSpark drafter into lmbrrr's (already complete, Qwen35-native) spec-decode loop so `gguf-run`/`dspark` runs speculative decode on the ternary Bonsai target. NOT blocked — the drafter is shipped.

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
4. **DONE — Target-load seam**: `Qwen35CausalLM` got the spec-target methods (set_device_capture/take_device_captures/set_verify_state_capture/snapshot/restore/rollback_to_prefix/forward_all_logits) + `TokenEmbedding::Packed` (−1.84 GB, the fix for the memory-pressure residual-collapse bug); a focused `spec_decode` loop (`gguf-run --spec-drafter`) uses readvance rollback. Output byte-correct vs plain decode, mean_accepted 3.4–4.0/4.
5. **PARTLY DONE / GATED — Q2_0 weight-bound verify GEMM.** This is the whole speedup story. Verify = 81% of round time ⇒ spec sits at parity (~11–12.5 tok/s vs 14.68 plain), not 2–3×. Five kernel variants measured (mc is compute-bound, tile-mm wastes padding, the from-scratch + planar kernels are read/staging-bound). `q2_0_mm2d_planes` (planar [k][row] repack) + `kernel_mul_mm2d_q2_0_smallm` are committed and correct (rel_err 0.0000, flat/weight-bound-structured) but the hand-rolled software unpack caps at 38 GB/s. **Root cause: the weight-bound path needs `tensor_ops::matmul2d` with a 2-bit B operand (`int2b_format`), which is Metal 4.1 — NOT in the M3's shipped toolchain (only 4-bit); 4-bit up-convert = 2× memory → OOM.** So the 2–3× is a Metal-4.1-toolchain gate. Full analysis + the exact q4_K-mirror path + version facts: **`docs/research/dspark-verify-weightbound-gemm.md`**.
6. **DONE — e2e** (correct + at-parity) + throughput/breakdown measured. The 2–3× awaits item 5's toolchain gate (or CUDA / a larger block_size drafter — see the research doc's "Paths to the multiplier").

See `docs/research/dspark-verify-weightbound-gemm.md` and memory [[dspark-bonsai-drafter-shipped]], [[dspark-bonsai-e2e-working]], [[ternary-q2_0-gemv-exhausted]]. Progress log: `.comments/`.
