# MiniCPM-V-4.6 Candle Runner Implementation Notes

Date: 2026-07-07

## Implemented

- Added a Rust `lmbrrr` crate with a `lmbrrr run` CLI.
- Uses published Candle crates at `0.11.0`; no local Candle path dependency.
- Supports default CPU builds and `--features metal` builds.
- Resolves model artifacts from Hugging Face with local overrides for config, tokenizer, generation config, preprocessor, and safetensors.
- Validates required MiniCPM/Qwen3.5 safetensor names and shapes before model construction.
- Implements Qwen3.5 text decoding in Candle:
  - zero-centered Qwen3.5 RMSNorm semantics;
  - partial RoPE;
  - full-attention layers with q-projection gate, q/k norm, KV cache, and causal mask;
  - Gated DeltaNet layers with causal depthwise conv state and recurrent state;
  - tied LM head fallback from token embeddings.
- Implements MiniCPM image preprocessing:
  - resize/slice grid selection;
  - normalization;
  - NaViT patch packing;
  - target size and placeholder expansion.
- Implements the MiniCPM vision path:
  - patch embedding plus nearest position ids;
  - variable-length vision attention;
  - layer-6 window attention merger;
  - final 2x2 visual downsample MLP merger;
  - replacement of `<|image_pad|>` token embeddings with visual features.
- Generation supports greedy or sampled decoding, EOS stop ids, repeat penalty, streaming decoded text, and timing metrics.

## Verification

- `cargo check`
- `cargo test`
- `cargo check --features metal`

## Current Gaps

- The full `openbmb/MiniCPM-V-4.6` checkpoint was not executed in this pass; the code compiles but still needs an end-to-end model run on the real safetensors.
- No Transformers parity oracle has been generated yet for token ids, image processor outputs, hidden states, or logits.
- Gated DeltaNet uses a recurrent correctness path for both prefill and decode. It should be mathematically aligned with the recurrent rule, but it is not the chunked FLA prefill kernel and will be slower on long prompts.
- Video-shaped inputs are still reserved at the CLI/model boundary but not implemented.
- Quantization, speculative decoding, and custom Metal kernels are intentionally outside this first runner.
