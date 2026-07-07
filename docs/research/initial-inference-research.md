# Initial Inference Research

Date: 2026-07-07

This note captures the initial research direction for `lmbrrr`: pushing small language and vision-language models on Apple Metal through Candle, with MiniCPM-V-4.6 as the first concrete target.

The DSpark v1 paper is vendored at `docs/research/papers/dspark-2607.05147v1.pdf`. It matches arXiv `2607.05147v1`, "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation."

## Repository Context

The repo is currently a README-only skeleton. The stated direction is:

- Use Candle in Rust.
- Target Metal first.
- Start from MiniCPM-V-4.6.
- Upstream generally useful Candle improvements when possible.

MiniCPM-V-4.6 is not a plain text decoder. Its Hugging Face config says it is `MiniCPMV4_6ForConditionalGeneration`, with:

- Vision stack: `minicpmv4_6_vision`, hidden size 1152, 27 layers, 16 heads, patch size 14.
- Text stack: `qwen3_5_text`, hidden size 1024, 24 layers, 8 attention heads, 2 KV heads for full attention, vocab size 248094.
- Hybrid token mixing: three `linear_attention` layers followed by one `full_attention` layer, repeated through the 24-layer text stack.
- Long context: `max_position_embeddings` 262144, partial RoPE factor 0.25, RoPE theta 10000000.
- Linear-attention specifics: 16 key heads, 16 value heads, 128-dim linear key/value heads, 4-wide convolution kernel.
- MTP: `mtp_num_hidden_layers` is 1.
- Multimodal glue: image token id 248056, video token id 248057, visual features inserted at layer 6.
- Processor defaults: max image slices 9, scale resolution 448, slice mode enabled.

Implication: the first implementation is a model-port project before it is an inference-optimization project. A useful first milestone is a text-only Qwen3.5/MiniCPM decoder path, then the full MiniCPM-V processor and vision insertion path.

Sources:

- MiniCPM-V-4.6 model: https://huggingface.co/openbmb/MiniCPM-V-4.6
- Config: https://huggingface.co/openbmb/MiniCPM-V-4.6/resolve/main/config.json
- Preprocessor config: https://huggingface.co/openbmb/MiniCPM-V-4.6/resolve/main/preprocessor_config.json

## Speculative Decoding Threads

### EAGLE-3

EAGLE-3 is the strongest "simple enough to implement first" speculative-decoding family to study. It moves away from EAGLE's feature-regression constraint and uses direct token prediction, multi-layer feature fusion, and a training-time test procedure so the draft model learns to operate on its own prior outputs. It keeps EAGLE-2's context-aware dynamic draft tree.

The paper reports up to 6.5x speedup and roughly 20%-40% improvement over EAGLE-2, depending on target model, task, and sampling setup.

Implementation implications:

- Needs target-model hidden states from multiple layers.
- Needs a trained draft head/checkpoint per target model.
- Needs tree verification or a simpler chain verifier for a first local prototype.
- More approachable than DFlash/DSpark because it is still autoregressive/tree-style drafting rather than block diffusion plus scheduling.

Sources:

- Paper: https://arxiv.org/abs/2503.01840
- Code: https://github.com/SafeAILab/EAGLE

### DFlash

DFlash uses a lightweight block-diffusion drafter to generate a whole draft block in parallel, conditioned on target features. A key engineering idea is KV injection: target hidden features are inserted into every draft layer as key/value context, rather than fused once at the input.

The paper reports up to 6.1x speedup on Qwen3-8B, up to 2.5x higher speedup than EAGLE-3 in its experiments, and roughly 4x class speedups on Qwen3 reasoning-mode evaluations. It also shows DFlash is useful in SGLang serving measurements on Qwen3-4B, Qwen3-8B, and Qwen3-Coder-30B-A3B.

Implementation implications:

- Strong research target after EAGLE-style instrumentation exists.
- Requires a trained block-diffusion drafter for the target.
- Needs target hidden feature extraction and KV-injected draft layers.
- For local Apple Silicon, the "single forward pass for many draft tokens" advantage is plausible, but must be measured because Metal kernel shapes and memory traffic may differ sharply from B200/FA4 results.

Sources:

- Paper: https://arxiv.org/abs/2602.06036
- Code link from paper: https://github.com/Chen-Yesheng/DFlash

### DSpark

DSpark builds on the parallel-drafter line and specifically addresses two DFlash-style problems:

- Parallel draft positions lack intra-block dependency, causing later draft positions to decay in acceptance.
- Verifying a long block regardless of confidence wastes target-model capacity under serving load.

Its two main components are:

- Semi-autoregressive generation: a parallel backbone plus a lightweight sequential Markov or RNN head that injects local dependency between sampled draft tokens.
- Confidence-scheduled verification: a confidence head predicts per-position prefix survival probabilities, then a hardware-aware prefix scheduler chooses how many draft tokens to verify for each request.

The paper reports offline accepted-length improvements over EAGLE-3 of 30.9%, 26.7%, and 30.0% on Qwen3-4B, 8B, and 14B targets, and improvements over DFlash of 16.3%, 18.4%, and 18.3%. In DeepSeek-V4 production serving, it reports 60%-85% faster per-user generation for V4-Flash and 57%-78% for V4-Pro at matched throughput levels.

Implementation implications:

- The semi-autoregressive head is potentially useful for single-user local generation.
- The hardware-aware scheduler is primarily a high-concurrency serving optimization. For this repo, the first version should be a simpler confidence-threshold or expected-value scheduler, then we can add load-aware behavior once we have batching.
- DSpark is not a first milestone. It needs baseline generation, acceptance-length measurement, a draft-model training/eval path, and verifier instrumentation first.

Sources:

- Paper: https://arxiv.org/abs/2607.05147
- DeepSpec training repo: https://github.com/deepseek-ai/DeepSpec
- DSpark checkpoints: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark

## Quantization Threads

### MLX Learned Quantization

MLX LM documents four quality-oriented quantization paths:

- Distilled Weight Quantization (DWQ)
- Activation-aware Weight Quantization (AWQ)
- Dynamic quantization
- GPTQ

The dynamic quantization path estimates sensitivity for each quantizable layer, then uses higher precision for sensitive layers and lower precision for less sensitive layers. MLX notes that dynamic quantization is fastest to run, while DWQ is slower but can produce better quality, and methods can be cascaded.

Implementation implications for Candle:

- Start with per-module sensitivity scoring rather than a single uniform bit width.
- Keep a JSON sensitivity artifact so quantization experiments are reproducible.
- Compare at least: fp16/bf16, uniform 8-bit, uniform 4-bit, dynamic 4/5-bit, and selective "do not quantize" modules.
- For MiniCPM-V, separately score text linear attention, full attention, MLP, embeddings/LM head, MTP head, vision encoder, and multimodal projection/insertion modules.

Sources:

- MLX LM learned quantization: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md

### Unsloth Dynamic 4-bit

Unsloth's dynamic 4-bit writeup is especially relevant because it emphasizes multimodal failure modes. Their key practical point is that full 4-bit quantization can damage sensitive modules, especially vision-side modules in some VLMs. Leaving selected modules at higher precision costs more memory but can recover much of the lost quality.

Implementation implications:

- Do not assume text-only quantization heuristics transfer cleanly to MiniCPM-V.
- Treat the vision encoder and multimodal bridge as likely high-sensitivity until measured.
- Keep visual/OCR/doc-understanding tasks in the eval harness, not just text perplexity or chat prompts.

Source:

- Unsloth dynamic 4-bit: https://unsloth.ai/blog/dynamic-4bit

## Model Architecture Threads

### Qwen3 and Qwen3.5

Qwen3 introduced unified thinking and non-thinking behavior plus a thinking-budget mechanism. Qwen3.5 is directly relevant to MiniCPM-V-4.6 because the MiniCPM text config identifies as `qwen3_5_text` and uses the 3:1 hybrid linear/full-attention pattern.

Implementation implications:

- Hybrid attention is the core decoder work. Full-attention-only assumptions will not hold.
- Linear attention state/cache design matters as much as KV cache design.
- Benchmark with thinking enabled and disabled where the model supports it, because reasoning traces change output length and speculative acceptance behavior.

Sources:

- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- Qwen3.5 model docs: https://huggingface.co/docs/transformers/model_doc/qwen3_5
- Qwen3.5 release blog: https://qwen.ai/blog?id=qwen3.5

### Gemma 4

Gemma 4 is relevant as a comparative model family rather than the first port. The model card describes multimodal open-weight models with up to 256K context, dense and MoE variants, and sizes including E2B, E4B, 12B, 26B A4B, and 31B. It also exposes thinking mode, variable visual token budgets, and MTP documentation.

Implementation implications:

- Gemma 4 E2B/E4B are plausible second targets after the Candle infrastructure exists.
- Gemma's MTP path is worth studying alongside MiniCPM's `mtp_num_hidden_layers = 1`.
- Variable visual token budgets provide a clean benchmark axis for speed/quality tradeoffs.

Source:

- Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4

## Recommended Project Sequence

1. Baseline harness
   - Create a Rust/Candle CLI that can load config metadata, run deterministic prompts, and record tokens/sec, time to first token, memory, generated length, and sample outputs.
   - Add a small prompt suite: short chat, math, code, long-context, OCR/image later.

2. Text-only decoder path
   - Start from a Qwen3.5 text-only model or extract the MiniCPM text path if practical.
   - Implement the hybrid `linear_attention` + `full_attention` stack, with correct caches/state.
   - Validate against Transformers logits on a few short prompts before optimizing.

3. Full MiniCPM-V path
   - Implement processor behavior: slicing, scale resolution, patching, image ids.
   - Implement vision encoder and layer-6 visual insertion.
   - Add image/OCR eval prompts because quantization and visual token budgets will need them.

4. Quantization lab
   - Implement or import uniform quantized linear kernels first.
   - Add per-module sensitivity scoring inspired by MLX/Unsloth.
   - Measure quality and speed for mixed precision on text and vision tasks.

5. Speculative decoding lab
   - Start with the model's built-in MTP-1 path if weights and config support it cleanly.
   - Add an EAGLE-style chain/head prototype, because it exercises the verifier loop and hidden-state capture.
   - Move to DFlash/DSpark only once accepted length, verifier waste, and draft latency are measurable.

6. Metal kernel work
   - Prioritize kernels that appear in the baseline profile: quantized matmul/dequant fusion, full attention, and Qwen3.5 linear attention state updates.
   - Leave DSpark-style variable-length batch verification for later; it is more relevant after batching/concurrency exists.

## Immediate Open Questions

- Is the first implementation target full MiniCPM-V-4.6, or a text-only Qwen3.5/MiniCPM extraction to de-risk the decoder?
- Do we want this repo to vendor/fork Candle immediately, or start with a local example crate and upstream once the model path is proven?
- What Apple hardware is the primary benchmark target: M-series laptop, Mac Studio, or both?
- Which quality benchmarks are acceptable for early iteration: deterministic prompt snapshots, lm-eval text tasks, image/OCR tasks, or all of the above?
