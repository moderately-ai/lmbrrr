# Modal pipeline rightsizing — measured (2026-07-11)

Prep phase for round-2 training per the user directive: measure and rightsize the pipeline with small-scale dry runs before the long runs. All numbers from live probes with the new `GpuMonitor` (util/mem/headroom sampled every 10 s into every GPU stage's logs).

## Data generation (`regenerate`, 1×H100, 96-sample slices, max_new_tokens 1024)

| generate batch | admission | padded tok/s | mean GPU util | peak mem |
| --- | --- | --- | --- | --- |
| 16 (round-1 config) | arrival order | 392 | 39% | 40.0 GiB |
| 16 | length-sorted | 598 | 38% | 24.7 GiB |
| 64 | arrival order | 846 | 44% | 25.4 GiB |
| 64 | length-sorted | 1037 | 41% | 21.8 GiB |
| **128** | **length-sorted** | **1432** | 43% | 44.8 GiB |
| 192 | — | OOM | — | >134 GiB alloc |

**Operating point: batch 128 + `--sort-by-length` = 3.65× the round-1 config.** The 192 failure is the prefill logits materialization (batch × padded-len × 248k vocab in HF generate) — the knee is real, not tunable-away without patching generate. Length-sorted admission alone is worth ~1.5× at any batch (generate runs every batch to its slowest member). Utilization stays ~43% because decode is bandwidth-bound at these batch sizes; the remaining headroom would need continuous batching (vLLM/SGLang), noted as a future option, not needed for round-2 economics.

## Training (`train`, 1×H100 probe, global batch 64, 10 steps, round-1 cache = 125 GiB / 20k samples)

| config | step time | GPU util | notes |
| --- | --- | --- | --- |
| volume-fed, lbs=1 (round-1 shape) | **180 s/step** | **0%** | volume random reads ≈ 2 MB/s effective; GPU fully starved |
| NVMe-staged, lbs=4 | **5.3 s/step steady** | bursts to 100% | staging: 125 GiB in 184 s (694 MB/s) |

**~35× training-loop speedup.** Round-1's 11 s/step (4×H100, gb=512) was IO-bound the whole time. Micro-batching works (loss trajectory matches lbs=1); lbs=4 peaked at 76.7/79.7 GiB, so **lbs=2 is the run default** (4k-token tails need margin; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set as belt-and-braces). New defaults in `train`: stage to NVMe (1 TiB ephemeral disk covers the ~625 GiB 100k cache), lbs=2.

## Deployment-config trace checkpoint

`lmbrrr fakequant-export` (new subcommand) rewrites the HF checkpoint with every q4k-full-text-policy tensor passed through **candle's own Q4_K quantize→dequantize** — the deployment's exact rounding (gguf-py only implements k-quant dequantize; llama.cpp conversion doesn't support this arch). 150 tensors, max |Δw| 0.026, output verified coherent through the lmbrrr runner and uploaded to `/vol/models/minicpm-v46-fakequant-q4kft`. Residual mismatch: generator lm_head stays dense (tied embedding). A 500-sample validation regen with this checkpoint + the new settings is the final prep gate.

## Revised round-2 costs (vs the corpus-scaling plan's estimates)

~110M generated tokens for 100k conversations at ≥1.4k tok/s/GPU → **~22 GPU-h ≈ $90** data-gen (plan said $160). Training 15–20 epochs at gb=512: with staged NVMe + lbs=2, projected **~10–14 s/step single-GPU → $130–180**, or ~4×H100 for ¼ the wall-clock at the same cost. Cache prep unchanged (unprofiled — it ran at acceptable cost in round 1; profile only if it surprises). Round-2 total tracks **~$250–300**, under the $310 plan.

## Validation regen (fakequant checkpoint, budgeted batching) — PASSED

500/500 conversations, 0 errors, 1249 padded-tok/s through the full length distribution (`/vol/data/regen-fakequant-500.jsonl`; samples verified coherent). Two hardening fixes came out of this gate: the fakequant export now backfills `chat_template.jinja`/`tokenizer_config.json` (lmbrrr's hub cache never fetches them), and batch admission is token-budgeted (32k positions) because sorted admission back-loads long conversations and HF generate materializes batch × padded-len × 248k-vocab prefill logits — the fixed-batch-128 config OOM'd at sample 414 on exactly that tail. Peak memory under budget: 22.5 GiB (budget has headroom to raise if the HF path stays the engine).

## Open: continuous-batching engine probe

HF generate is framework-bound (~43–46% util, ~1% of the 2.6 GB model's decode-bandwidth ceiling; composite lacks `logits_to_keep`). SGLang probe in flight (CUDA-devel base; first attempt failed on deep_gemm needing CUDA_HOME, not on the architecture). If SGLang serves the hybrid composite: expect ~10×+ data-gen throughput; else vLLM transformers-backend fallback; else the HF numbers above are the engine of record and round-2 economics stand as revised.
