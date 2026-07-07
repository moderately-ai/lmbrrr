# MiniCPM-V-4.6 Implementation Surface

Date: 2026-07-07

Ticket: `research-minicpm-surface`

This note maps MiniCPM-V-4.6 onto the pieces we would need to implement in Candle/Rust. It is based on the repo-local snapshots under `docs/research/models/minicpm-v-4.6`, `docs/research/models/qwen3.5`, and `docs/research/candle`.

## Artifact Snapshot

Vendored artifacts used for this pass:

- `docs/research/models/minicpm-v-4.6/hf-model/config.json`
- `docs/research/models/minicpm-v-4.6/hf-model/preprocessor_config.json`
- `docs/research/models/minicpm-v-4.6/hf-model/tokenizer.json`
- `docs/research/models/minicpm-v-4.6/hf-model/tokenizer_config.json`
- `docs/research/models/minicpm-v-4.6/hf-model/model-safetensors-header.json`
- `docs/research/models/minicpm-v-4.6/transformers/modeling_minicpmv4_6.py`
- `docs/research/models/minicpm-v-4.6/transformers/processing_minicpmv4_6.py`
- `docs/research/models/minicpm-v-4.6/transformers/image_processing_minicpmv4_6.py`
- `docs/research/models/qwen3.5/transformers/modeling_qwen3_5.py`
- `docs/research/candle/qwen3.rs`
- `docs/research/candle/qwen3_vl/*.rs`
- `docs/research/candle/siglip.rs`

Also vendored:

- `docs/research/papers/dspark-2607.05147v1.pdf`
- `docs/research/papers/dflash-2602.06036.pdf`
- `docs/research/papers/eagle3-2503.01840.pdf`
- `docs/research/papers/qwen3-2505.09388.pdf`

## Checkpoint Facts

MiniCPM-V-4.6 is native in Transformers 5.7+; the Hugging Face checkpoint does not ship custom Python model files. The relevant implementation is in `transformers.models.minicpmv4_6` and the text backbone uses `transformers.models.qwen3_5`.

The checkpoint header has 779 BF16 tensors. No full weight download was needed; `model-safetensors-header.json` was created from the safetensors header only.

Important weight-surface findings:

- All tensors are BF16.
- Language embedding is `model.language_model.embed_tokens.weight` with shape `248094 x 1024`.
- There is no separate `lm_head.weight`; the HF model ties `lm_head` to the embedding matrix.
- The config contains `mtp_num_hidden_layers = 1`, but the safetensors header contains no `mtp` tensors. Treat built-in MTP as unavailable for this checkpoint until proven otherwise.
- Text layers are under `model.language_model.layers.*`.
- Vision layers are under `model.vision_tower.*`.
- The multimodal projection is under `model.merger.mlp.0.*`.

## Processor And Token Surface

The chat template converts image content to `<|image_pad|>` and video content to `<|video_pad|>`. The processor later expands each visual placeholder into the correct count of repeated image/video placeholder tokens.

Relevant special token ids from `tokenizer.json`:

- `<|im_start|>`: 248045
- `<|im_end|>`: 248046
- `<|vision_start|>`: 248053
- `<|vision_end|>`: 248054
- `<|image_pad|>`: 248056
- `<|video_pad|>`: 248057
- `<image>`: 248078
- `</image>`: 248079
- `<slice>`: 248088
- `</slice>`: 248089
- `<image_id>`: 248090
- `</image_id>`: 248091

The model config repeats the important runtime ids:

- `image_token_id`: 248056
- `video_token_id`: 248057
- `eos_token_id`: 248044, with generation config also ending on 248046.

Processor behavior to port:

- Default image slicing is enabled.
- Default `max_slice_nums` is 9 in `preprocessor_config.json`, but the README examples pass `max_slice_nums=36`.
- Default `scale_resolution` is 448.
- Patch size is 14.
- `downsample_mode = "16x"` is the default; `"4x"` keeps 4x more visual tokens.
- `downsample_mode` must be passed consistently to prompt construction and generation, because it changes placeholder count and vision merging.
- `use_image_id` defaults true, so generated prompts include `<image_id>{local_index}</image_id>`.

The image processor emits:

- `pixel_values`: NaViT-packed tensor shaped like `[1, channels, patch_size, total_patch_sequence]`.
- `target_sizes`: per-source/slice patch-grid sizes.
- `grids`: image slice grid layout.
- `num_patches_per_image`: how many source/slice visual units each input image produced.

## Vision Path

The vision embedding path is a modified SigLIP/NaViT style pipeline:

1. A Conv2D patch embedding maps images to hidden size 1152.
2. Position ids are selected with nearest-position logic from the target patch grid.
3. The vision encoder has 27 transformer layers.
4. In default `"16x"` mode, a `MiniCPMV4_6ViTWindowAttentionMerger` runs after vision layer 6.
5. After the window merger, target sizes are halved and later attention metadata is updated.
6. A final layer norm produces vision hidden states.
7. `MiniCPMV4_6Merger` does a 2x2 spatial merge and projects from 1152-side vision hidden state into the 1024-dim text embedding space.

The model also supports `"4x"` mode by skipping the intermediate ViT merger. This keeps more visual tokens and changes placeholder count through the processor's divisor logic.

Port implication: we should implement image-only first. Video repacks frames through the same image feature path, but it adds frame/grid bookkeeping that can wait.

## Text Decoder Path

The text config identifies as `qwen3_5_text`, but the shape is smaller than public Qwen3.5 dense defaults:

- Hidden size: 1024
- Layers: 24
- MLP intermediate size: 3584
- Full-attention heads: 8
- Full-attention KV heads: 2
- Full-attention head dim: 256
- Linear-attention key heads: 16
- Linear-attention value heads: 16
- Linear key/value head dims: 128
- Linear conv kernel dim: 4
- Context: 262144
- Partial rotary factor: 0.25
- RoPE theta: 10000000

Layer pattern:

- Linear attention: layers 0, 1, 2
- Full attention: layer 3
- Repeat every 4 layers through layer 23

The full-attention block is close to Qwen3, but not identical:

- `q_proj` emits both query and a gate: shape `4096 x 1024` for MiniCPM-V-4.6.
- `k_proj` and `v_proj` are `512 x 1024`.
- `q_norm` and `k_norm` operate on head dim 256.
- Attention output is multiplied by `sigmoid(gate)` before `o_proj`.

The linear-attention block is the new hard part. Transformers names it `Qwen3_5GatedDeltaNet`; it contains:

- `in_proj_qkv`: `6144 x 1024`
- `in_proj_z`: `2048 x 1024`
- `in_proj_b`: `16 x 1024`
- `in_proj_a`: `16 x 1024`
- depthwise causal conv weight: `6144 x 1 x 4`
- `dt_bias`: 16
- `A_log`: 16
- gated RMS norm weight: 128
- `out_proj`: `1024 x 2048`

Runtime state is not a normal KV cache for linear layers. Each linear layer needs:

- `conv_state`: rolling depthwise convolution context.
- `recurrent_state`: Gated DeltaNet recurrent state shaped around value heads, key dim, and value dim.

Prefill uses the chunked gated-delta rule. Single-token decode uses the recurrent gated-delta rule.

## Candle Reuse And Gaps

Repo-local Candle snapshots show useful starting points:

- `docs/research/candle/qwen3.rs`: full-attention Qwen3 text decoder with RoPE, q/k norm, GQA, KV cache, tied lm head support.
- `docs/research/candle/quantized_qwen3.rs`: quantized Qwen3 GGUF path and cache optimizations.
- `docs/research/candle/qwen3_vl/*.rs`: multimodal embedding insertion, 3D position id handling, Qwen3-VL vision/text structure.
- `docs/research/candle/siglip.rs`: SigLIP-style vision transformer building blocks.

Observed gaps:

- No `minicpmv4_6` model implementation.
- No `qwen3_5` model implementation in the captured Candle model list.
- No `GatedDeltaNet`, `linear_attn`, `conv_state`, or `recurrent_state` matches in the captured Candle sources.
- Existing `qwen3.rs` assumes every layer is full attention; it cannot directly represent the hybrid Qwen3.5 layer schedule.
- Existing `qwen3_vl` gives useful multimodal patterns, but MiniCPM-V's image processor, visual token layout, window merger, and text backbone differ.

There is an open upstream Candle issue, #3514, describing a Qwen3.5/3.6 Gated DeltaNet port and fused kernels. That is worth tracking before duplicating large amounts of work.

Source:

- https://github.com/huggingface/candle/issues/3514

## Recommended Port Order

1. Config and artifact loader
   - Parse MiniCPM and Qwen3.5 configs into Rust structs.
   - Load safetensors metadata and validate expected tensor names/shapes without running inference.
   - Treat BF16 as the baseline dtype to verify on Metal.

2. Text-only Qwen3.5 decoder
   - Implement full-attention layers by adapting `qwen3.rs`.
   - Add the q-projection gate and partial-RoPE behavior.
   - Implement the hybrid layer schedule.
   - Implement Gated DeltaNet in a correctness-first path before optimizing.

3. Gated DeltaNet cache/state
   - Define explicit per-layer cache structs for `conv_state` and `recurrent_state`.
   - Implement prefill/chunk mode and single-token recurrent decode mode.
   - Match Transformers layer outputs on small fixtures before touching Metal kernels.

4. Baseline harness
   - Run text prompts and log TTFT, tokens/sec, generated length, dtype, device, and sample output.
   - Add a Transformers comparison script for logits on short prompts.

5. MiniCPM image processor and vision path
   - Port image resize/slice/patch packing.
   - Port modified SigLIP/NaViT vision tower.
   - Port window attention merger and final merger MLP.
   - Add image placeholder expansion and masked embedding insertion.

6. Optimization tracks
   - Quantization should start after correctness is established.
   - Speculative decoding should not start from built-in MTP for this checkpoint, because MTP tensors are absent.
   - First speculative experiment should likely be an external EAGLE-style head or a separate small drafter, after hidden-state capture and verifier metrics exist.

## Validation Checklist

Minimum parity fixtures before optimization:

- Tokenizer/chat-template output matches Transformers for text-only and single-image prompts.
- Image processor emits matching `target_sizes`, `grids`, and placeholder token counts for a small set of images.
- One Gated DeltaNet layer matches Transformers on a short synthetic prefill.
- One Gated DeltaNet layer matches Transformers on single-token cached decode.
- Full text decoder logits match Transformers on a short prompt.
- Vision feature count matches placeholder count in `"16x"` and `"4x"` modes.
- Full MiniCPM-V image prompt produces plausible output and stable token counts.

## Follow-Up Tickets

Existing tickets that should use this note:

- `audit-candle-support`
- `scaffold-baseline-harness`
- `port-qwen35-text-decoder`
- `port-minicpm-v46-full-path`
- `design-dynamic-quantization-lab`
- `design-speculative-decoding-lab`

New tickets likely needed after the Candle audit:

- Implement Qwen3.5 Gated DeltaNet reference path.
- Add Qwen3.5 linear-attention cache/state types.
- Add MiniCPM image processor fixtures.
- Add MiniCPM vision window merger.
- Add Transformers parity scripts.
