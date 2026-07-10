---
id: train-dspark-semi-autoregressive-drafter
title: Train DSpark semi autoregressive drafter
status: in-progress
priority: p1
dependencies: [design-full-dspark-drafter]
related: []
scopes: [inference/speculative, evals, runtime/candle]
shared_scopes: [docs/research]
paths: [evals/dspark/**, evals/eagle/**, src/main.rs, docs/research/dspark-semi-autoregressive-training.md]
tags: [speculative, dspark, training]
claimed_from: todo
assignee: claude
lease_expires_at: 1783704607
---
## Goal

Train a DSpark-style drafter with a parallel backbone plus lightweight semi-autoregressive Markov head, not an EAGLE-only recurrent chain.

## Progress (2026-07-10)

- Every key DeepSpec file read in full (~5,500 lines: modeling/loss/markov/common, trainers, ckpt_manager, full data pipeline, eval loop, utils); no assumed semantics remain. Load-bearing facts: training REQUIRES flex_attention (block mask is a flex BlockMask), block context window is strictly kv < anchor_pos, checkpoints are standard HF safetensors via save_pretrained (Candle-loadable), verify is true rejection sampling.
- MiniCPM adaptation committed on DeepSpec branch `minicpm-v46` (~/workspace/github.com/deepseek-ai/DeepSpec, commit 0869223): minicpm config builder (clean Qwen3Config from text_config, rope_parameters set both flat and dict forms), MiniCPMDSparkTrainer (AutoModelForImageTextToText for the frozen embed/lm_head copy — verified resolving on the wrapper), prepare_target_cache backbone shim (model_type minicpmv4_6 -> .language_model), `minicpm` chat template (loss mask verified against the real template: assistant content + <|im_end|> supervised, think scaffold excluded), config file (block 8, 2 layers, capture [1,6,11,16,21], mask_token 248077 = <unk>, num_anchors 256 for the vocab-248094 OOM guard), synthetic smoke script.
- Local CPU smoke validated config -> 672.9M-param model (~165M trainable) -> anchor sampling -> block mask -> forward entry; FlexAttention has no CPU backward (torch limitation), so the forward+backward gate runs on a Modal GPU in the pinned env.
- Modal: 1.5.1 installed via uv, profile `moderately-ai` authenticated; no `huggingface` secret yet (create at deploy). Image needs DeepSpec requirements + flash-linear-attention + causal-conv1d (transformers falls back to slow torch DeltaNet without them).
- Modal app landed (evals/dspark/modal_app.py + regenerate_answers.py): staged functions smoke/download_prompts/regenerate/inspect_data/prepare_cache/train over the `lmbrrr-dspark` volume, pinned image (torch 2.9.1 / transformers 5.10.2 / flash-linear-attention), `huggingface` secret created via --from-dotenv (no token exposure).
- SMOKE PASSED on Modal H100 in the pinned env: forward through the flex block mask, finite three-term loss (2.639), gradients on backbone/fc/markov/confidence, frozen heads clean. Hardware constraint discovered: head_dim-256 flex kernels need >100KB SM shared memory — L4/Ada cannot train this drafter; A100/H100 only.
- In flight: download_prompts(2000) -> regenerate(500, H100) -> inspect_data gate, then prepare_cache + smoke train.

## Acceptance

- Build a trace dataset in binary shards (safetensors/npz, not per-token JSON) capturing every block position: anchor hidden features from the capture layers, target tokens, and target top-k distributions with a tail-mass bucket (k >= 64) or raw hidden states for local frozen-LM-head projection. The current JSON exporter only records the last position per forward and is not viable at corpus scale.
- Train a parallel block drafter that emits base logits for multiple future positions in one forward path, with DFlash-style target-context injection into draft K/V.
- Add a Markov sequential head (low-rank transition bias B = W1*W2, r ~= 256) that conditions each position on the previous sampled draft token.
- Use the full vocabulary via the frozen shared target embedding and LM head; no observed-vocabulary output head.
- Train with the paper's three-term objective: cross-entropy + total-variation distribution matching + confidence BCE, position-weighted by exp(-(k-1)/gamma) (default weights 0.1 / 0.9 / 1.0).
- Train a confidence head using per-position prefix survival labels c* = 1 - 0.5 * total-variation(draft, target).
- Trainer runs on CUDA (Modal credits are available for corpus generation and training). FALSIFIED: local MPS smoke is not viable — training requires flex-attention (BlockMask), which has no MPS support and no CPU backward; the pinned-env Modal GPU smoke replaces it (passed on H100).
- Export a safetensors artifact and manifest with backbone, Markov head, confidence head, draft width, capture layers, and calibration metadata.
- MiniCPM adaptation per the 2026-07-10 deep review: parser chat-template registration, the VLM backbone shim in prepare_target_cache (`_get_target_backbone`/`_get_target_hidden_size` — highest-risk touch point), minicpm modeling/config/trainer/evaluator registrations, and a `config/dspark/dspark_minicpm.py`.
- `mask_token_id` is an existing reserved MiniCPM special id (embeddings are frozen copies; never add a vocab row); capture layers strictly exclude the final layer (start [1, 6, 11, 16, 21]).
- Guard the known OOM at vocab 248094: fp32 draft-logits are [1, num_anchors, gamma, V] in the loss (~3.5 GB at defaults); reduce num_anchors or chunk the loss.
