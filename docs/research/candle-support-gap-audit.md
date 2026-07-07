# Candle Support Gap Audit

Date: 2026-07-07

Ticket: `audit-candle-support`

This audit compares the MiniCPM-V-4.6/Qwen3.5 implementation surface against the Candle source snapshots vendored under `docs/research/candle`.

## Snapshot Used

Captured from `huggingface/candle` `main`:

- `docs/research/candle/models-github-contents.json`
- `docs/research/candle/qwen3.rs`
- `docs/research/candle/quantized_qwen3.rs`
- `docs/research/candle/qwen3_vl/config.rs`
- `docs/research/candle/qwen3_vl/mod.rs`
- `docs/research/candle/qwen3_vl/text.rs`
- `docs/research/candle/qwen3_vl/vision.rs`
- `docs/research/candle/siglip.rs`
- `docs/research/candle/metal-kernels-github-contents.json`

## Existing Candle Pieces We Can Reuse

### Qwen3 Full-Attention Text Path

`qwen3.rs` already has the building blocks for ordinary Qwen-style full attention:

- Config parsing for dense Qwen3.
- RoPE tables.
- Q/K RMSNorm.
- GQA repeat.
- KV cache through `ConcatKvCache`.
- MLP with SiLU gated projection.
- Tied output head when `tie_word_embeddings` is true.

This is useful for MiniCPM-V-4.6's full-attention layers, but it is not enough for Qwen3.5. MiniCPM uses full attention only every fourth layer and uses a different q-projection shape because `q_proj` emits both query and a sigmoid gate.

### Qwen3-VL Multimodal Patterns

`qwen3_vl` is useful as a Candle-native reference for:

- Separating text and vision models.
- Running vision features, converting them to text dtype/device, and inserting them into text embeddings.
- Building multimodal position ids.
- Handling image/video placeholder regions.
- Using scatter-style updates into an embedding tensor.

This does not directly port MiniCPM-V-4.6 because MiniCPM uses a different processor, visual token layout, window merger, and Qwen3.5 text backbone.

### SigLIP Vision Blocks

`siglip.rs` provides reusable vision transformer ideas:

- Conv2D patch embedding.
- Vision attention.
- MLP blocks.
- Layer norms.

MiniCPM's vision tower is modified for NaViT variable-resolution packing and adds a `MiniCPMV4_6ViTWindowAttentionMerger` after vision layer 6. The existing SigLIP implementation is a starting point, not a drop-in module.

### Quantized Qwen3 Path

`quantized_qwen3.rs` is useful later for:

- GGUF-oriented quantized model loading.
- QMatMul patterns.
- Existing cache optimizations.
- Quantized Qwen-style forward structure.

It does not help the first MiniCPM milestone directly because the MiniCPM checkpoint we inspected is a BF16 safetensors checkpoint with native HF tensor names.

## Missing Pieces

### MiniCPM-V-4.6 Model Wrapper

No `minicpmv4_6` implementation appears in the captured Candle model list.

Needed:

- `MiniCPMV4_6Config` equivalent.
- `MiniCPMV4_6VisionConfig` equivalent.
- Full model wrapper containing `vision_tower`, `language_model`, `merger`, and tied `lm_head`.
- Weight-name mapping for the native safetensors names.

### Qwen3.5 Text Model

No `qwen3_5` implementation appears in the captured Candle model list.

Needed:

- `Qwen3_5TextConfig` equivalent.
- Hybrid `layer_types` schedule.
- Full-attention block variant with gated q-projection.
- Gated DeltaNet block for linear-attention layers.
- Position id handling compatible with Qwen3.5's 4-row text/multimodal position id convention.

### Gated DeltaNet

No `GatedDeltaNet`, `linear_attn`, `conv_state`, or `recurrent_state` code was found in the captured Candle sources.

Needed for correctness:

- `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`.
- Depthwise causal conv1d with kernel size 4.
- L2 normalization of q/k inside the gated-delta rule.
- `A_log` and `dt_bias` handling in float precision.
- Chunked gated-delta rule for prefill.
- Recurrent gated-delta rule for single-token decode.
- Gated RMSNorm equivalent.
- Output projection.

Needed for speed later:

- Metal kernels for the single-token decode path.
- Probably a fused depthwise conv + SiLU + state-shift path.
- A faster chunked prefill path if the reference implementation is too slow.

### Hybrid Cache State

Existing Qwen3 code assumes attention-style KV cache. Qwen3.5 needs two cache families:

- Full-attention layers: KV cache.
- Linear-attention layers: `conv_state` and `recurrent_state`.

The cache API should make layer type explicit rather than forcing linear-attention state into KV-cache abstractions.

### MiniCPM Processor

Candle model code alone is not enough. MiniCPM requires processor parity:

- Chat template or equivalent prompt assembly.
- Image resize and slicing.
- NaViT patch packing.
- Placeholder token expansion based on `target_sizes` and `downsample_mode`.
- `16x` vs `4x` visual token counts.
- Optional image IDs.

The image processor is likely a separate Rust module or a small support crate, not part of the pure model layer.

### MiniCPM Vision Merger

Needed:

- Variable-resolution patch embedding.
- Nearest-position id selection.
- Vision attention over packed sequences using cumulative sequence lengths.
- Window-attention merger after layer 6 for default `16x`.
- Target-size downsampling after the window merger.
- Final 2x2 merger MLP into the 1024-dim text space.

### Validation Infrastructure

Candle lacks repo-local validation scripts for this target because this repo is still a skeleton.

Needed:

- A Transformers oracle path that can run MiniCPM-V-4.6 on CPU/MPS/CUDA outside Candle.
- Fixture generation for token ids, image processor outputs, hidden states, and logits.
- A Rust benchmark CLI that can compare shapes and dump timing metrics.

## Upstream Watch Item

Candle issue #3514 describes a proposed Qwen3.5/3.6 Gated DeltaNet port and fused kernels. It is not merged in the captured source snapshot, but it is highly relevant. Before implementing the full Gated DeltaNet path from scratch, check whether a PR or branch became available.

Source:

- https://github.com/huggingface/candle/issues/3514

## First Implementation Shape

The lowest-risk project shape is:

1. Create a local `lmbrrr` Rust crate that depends on Candle.
2. Implement config parsing and safetensors metadata checks first.
3. Add a text-only `qwen3_5_text` model path.
4. Use a correctness-first Gated DeltaNet implementation, even if slow.
5. Validate against Transformers fixtures.
6. Only then add MiniCPM vision and processor work.
7. Only after correctness and measurement exist, start quantization or speculative decoding.

This keeps the fork/upstream decision open. If the Qwen3.5 path becomes generally useful, it can be upstreamed into Candle later without requiring the whole MiniCPM experiment to live upstream from day one.

## Follow-Up Tickets

Existing tickets unlocked by this audit:

- `scaffold-baseline-harness`
- `design-dynamic-quantization-lab`
- `design-speculative-decoding-lab`

Implementation tickets to create after the harness exists:

- Add Qwen3.5 config parsing and tensor-name validation.
- Implement Qwen3.5 full-attention block.
- Implement Gated DeltaNet reference path.
- Implement linear-attention state cache.
- Add Transformers parity fixture generator.
