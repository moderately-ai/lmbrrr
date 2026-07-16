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
1. **GGUF drafter loader** (new; mirror `DsparkDrafter::load` in src/dspark.rs but source from gguf `Content`): config from metadata, tensor name-map, markov_head_a/b `[rank,vocab]` orientation (transpose or adapt the port), dequant-to-bf16 backbone / keep-quantized heads (MixedLinear).
2. **Log-SNR conditioning** in `propose_backbone` (+ DsparkConfig fields `log_snr_conditioning`/`min`/`max`) — the formula above. Guard: fail loudly if metadata on but tensors absent.
3. **block_size = 4** flows from config (gamma ≤ block_size).
4. **Target-load seam**: drive the Bonsai GGUF `Qwen35CausalLM` through `dspark_drafter_run` (src/commands/dspark.rs, hard-typed to MiniCpm). Add spec-target methods to `Qwen35CausalLM` (delegate to inner `Qwen35TextModel` + own lm_head: forward_all_logits/_and_hidden, forward_verify_ids_and_hidden, forward_tree_all_logits, set_device_capture, set_verify_state_capture, snapshot/restore/rollback_to_prefix/rollback_tree, take_device_captures) and make the loop generic over a `SpecTarget` trait (impl for both MiniCpm + Qwen35CausalLM).
5. **Q2_0 mm2d verify GEMM** (perf, deferrable): the `kernel_mul_mv_q2_0_mc_t` verify kernel is correct but compute-bound (m=8 ≈ 5× m=1); a simdgroup-matrix ternary GEMM (like q4k mm2d + q4k_mm2d_planes) gives weight-bound verify for the DSpark economics.
6. **e2e**: greedy spec decode on Bonsai, verify token-parity vs plain decode, measure tok/s + acceptance vs the 14.63 baseline (llama.cpp claims 1.34× on CUDA).

See memory [[dspark-bonsai-drafter-shipped]], [[ternary-q2_0-gemv-exhausted]].
