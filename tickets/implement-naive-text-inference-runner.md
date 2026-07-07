---
id: implement-naive-text-inference-runner
title: Implement initial MiniCPM Qwen3.5 inference runner
status: done
priority: p1
dependencies: [scaffold-baseline-harness, port-qwen35-text-decoder]
related: []
scopes: [runtime/candle, model/qwen, model/minicpm]
shared_scopes: []
paths: [src/main.rs, src/minicpm.rs, src/qwen35.rs, src/weights.rs, src/prompt.rs, tests/minicpm_v46_text_parity.rs, docs/research/minicpm-v46-transformers-parity-oracle.md, docs/research/minicpm-v46-text-logits-parity.md]
tags: [implementation, inference, minicpm]
---
## Goal

Build the first correctness-first inference runner for MiniCPM-V-4.6 using Candle libraries.

This is the initial implementation, not a deliberately slow one: use efficient Candle primitives where they are available, keep model-specific reference paths only where Candle does not yet provide the required Qwen3.5/MiniCPM kernels, and defer quantization/speculative decoding/custom kernels.

## Work

- Create a runner that can load MiniCPM-V-4.6 config/tokenizer metadata and safetensors weights.
- Implement the Qwen3.5 text decoder path needed by the checkpoint, including the hybrid layer schedule.
- Implement a reference Gated DeltaNet path for linear-attention layers.
- Implement full-attention layers by adapting Candle Qwen3-style code, including q-projection gating.
- Implement basic generation: prompt tokenization, prefill, single-token decode loop, EOS handling, and deterministic sampling/greedy mode.
- Emit simple timing metrics: load time, prefill time, time to first token, decode tokens/sec, and output token count.
- Include the MiniCPM image processor and image embedding insertion path where practical, with text-only still working as the simplest smoke path.

## Acceptance

- A local command can run a text-only prompt against MiniCPM-V-4.6 or an extracted compatible fixture using Candle.
- Image-conditioned prompts have a compiled processor/model path and documented validation status.
- The runner validates tensor names/shapes against the safetensors header before generation.
- Short-prompt logits or next-token output are compared against a Transformers oracle for at least one fixture.
- Known missing pieces are documented, especially parity validation, video support, and performance limitations.
