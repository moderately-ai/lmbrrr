---
id: ternary-bonsai-27b-support
title: "EPIC: Run prism-ml Ternary-Bonsai-27B (ternary qwen35-hybrid VLM + external DSpark head) on lmbrrr"
status: todo
priority: p2
dependencies: []
related: [spike-ternary-type42-block-format, ternary-type42-dequant, metal-ternary-matmul-kernel, gguf-loader-qwen35-hybrid, qwen35-27b-config-scaleup, ingest-external-dspark-head, design-ternary-bonsai-e2e-bringup]
scopes: [candle-fork, runtime/candle, evals]
shared_scopes: []
paths: []
tags: [ternary-bonsai, model-compat, dspark]
---
## WHY

Compatibility probe for the "does our stack generalize beyond MiniCPM-V-4.6" question. `prism-ml/Ternary-Bonsai-27B-gguf` is a **Qwen3.6-27B hybrid-attention VLM** shipped as GGUF with a **ternary-quantized target** + a **pre-trained DSpark draft head**. Running it validates (a) external GGUF weight ingestion, (b) BitNet-style ternary quantization support in our Metal stack, and (c) our DsparkDrafter against an independently-trained full-DSpark head (a reference for the faithful-DSpark directive).

## FINDINGS (2026-07-15 GGUF inspection; receipts below)

Target `Ternary-Bonsai-27B-Q2_0.gguf` (7.17 GB, header parsed via range request):
- `general.architecture: qwen35` — **the same Qwen3.5-hybrid family our `qwen35.rs` already implements**, just 27B: block_count 64, hidden 5120, ffn 17408, GQA head_count 24 / kv 4, head_dim (key/value_length) 256, `full_attention_interval 4`, GatedDeltaNet SSM (`ssm_conv1d k=4`, `ssm_state_size 128`, `ssm_alpha/beta/a/dt/norm/out`), mrope `rope.dimension_sections [11,11,10,0]`, ctx 262144, vocab 248320, gpt2 BPE tokenizer.
- **Quantization is the departure**: 498 weight tensors are a **custom ggml type `42`** (every 2-D matrix: `token_embd`, `output`, `attn_qkv`, `attn_gate`, `ffn_up/gate/down`, `ssm_alpha/beta/out`); 353 F32 tensors are the norms/biases/conv/small SSM params. `file_type 41`. Type 42 is NOT in mainline llama.cpp / candle / gguf-python (their types top out ~39) → bespoke prism-ml ternary format (~2 bpw per the repo's `ternary`/`2-bit` tags).

DSpark head `Ternary-Bonsai-27B-dspark-bf16.gguf` (7.29 GB, **bf16 — ternary is target-only**, `arch: dspark`, 3.646B params): a **full DSpark draft model**, not a 1-layer MTP head — 6 transformer blocks + its own `token_embd`/`output` (1.27B each, 248320×5120) + DSpark machinery: `fc` [25600→5120] fusing **5 target layers** (`target_layers [1,16,31,46,61]`), rank-256 **Markov head** (`markov_head_a/b [256,248320]`), **confidence head** [5376→1] (= hidden⊕markov), **log-SNR conditioning** (`log_snr_fc1/fc2`, min/max ±9), **masked-block prediction** (`block_size 4`, `mask_token_id 248319`).

## GAP (what running this requires)

1. NOT a gap — the qwen35-hybrid architecture (config scale-up + GGUF name mapping): see [[gguf-loader-qwen35-hybrid]], [[qwen35-27b-config-scaleup]].
2. THE gap — custom ternary type 42: format spike [[spike-ternary-type42-block-format]] → dequant [[ternary-type42-dequant]] → Metal kernel [[metal-ternary-matmul-kernel]].
3. External full-DSpark head reconciliation vs our DsparkDrafter: [[ingest-external-dspark-head]].
4. End-to-end bring-up + eval: [[design-ternary-bonsai-e2e-bringup]].

## DONE-WHEN

Ternary-Bonsai-27B loads from GGUF and generates coherent text on the M-series, with its DSpark head drafting in our spec loop; a short eval confirms target parity vs the F16 reference and a measured draft acceptance. Ternary matmul runs on Metal (not just CPU dequant-to-bf16). Reusable artifacts (GGUF loader, ternary type, name map) are documented for the next external model.

## NON-GOALS / notes

Not committing to shipping Bonsai in the campaign — this is the compatibility experiment. The full F16 target (53.8 GB) is a fallback reference for validation; the ternary Q2_0/PQ2_0 (7.17 GB) is the on-device target. VLM `mmproj` is out of scope (text-only decode).
