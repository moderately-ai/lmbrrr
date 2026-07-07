# Dynamic Quantization Lab

Date: 2026-07-07

Ticket: `design-dynamic-quantization-lab`

## Goal

Design a measurable quantization lab for MiniCPM-V-4.6 on Candle/Metal. The lab
should adapt MLX learned quantization and Unsloth dynamic quantization ideas
without assuming that uniform 4-bit quantization is safe for a small VLM.

## Source Notes

MLX LM learned quantization documents four calibration-based paths: DWQ, AWQ,
dynamic quantization, and GPTQ. The dynamic path estimates per-layer output
sensitivity, uses higher precision for more sensitive layers, lower precision
for the rest, and saves a reusable sensitivity JSON artifact.

Unsloth dynamic quantization emphasizes that naive 4-bit can break VLMs. Their
dynamic 4-bit approach selectively avoids quantizing sensitive parameters while
using only modest extra memory. Their newer Dynamic 2.0 GGUF notes extend this
to per-model, per-layer quant choices, use larger curated calibration data, and
prefer KL divergence / flip-style behavior over plain perplexity.

Relevant sources:

- `docs/research/quantization/mlx-learned-quants.md`
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md
- `docs/research/quantization/unsloth-dynamic-4bit.html`
- https://unsloth.ai/blog/dynamic-4bit
- https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs

## Candle Constraints

Candle already has useful quantized infrastructure:

- `candle::quantized::QTensor`
- `candle::quantized::QMatMul`
- GGUF-oriented `quantized_var_builder`
- quantized Qwen examples such as `quantized_qwen3.rs`
- Metal quantized storage and matmul paths

Important local caveats:

- Existing Candle quantized model paths are GGUF-oriented. MiniCPM-V-4.6 is
  currently loaded from native HF safetensors.
- Quantized embeddings and norms often dequantize to normal tensors in
  `quantized_nn.rs`; the first useful target is weight-only linear layers.
- Candle's Metal quantized MM path currently asserts F32 activations. The
  single-token MV path is separate. Measure prefill/chunk and decode separately
  before assuming one kernel helps both.
- The current MiniCPM runner uses BF16 on Metal by default. Any F32 activation
  cast required by quantized kernels must be counted as part of the latency.

## Calibration Data

Use a small but mixed calibration suite first:

Text-only:

- short factual answer;
- arithmetic;
- long reasoning;
- code completion;
- tool-call style prompt;
- thinking enabled and disabled.

Vision-language:

- single small image;
- high-resolution sliced image;
- tall image;
- OCR/document image;
- multi-image comparison;
- video once video parity exists.

Keep the first calibration artifact deterministic and repo-local:

```text
evals/calibration/minicpm_v46_quant_calibration.jsonl
```

Each row should store prompt text, optional media paths, downsample mode, token
ids, and expected baseline metadata. Do not put large image bytes in JSON.

## Sensitivity Metrics

Score each candidate module with multiple signals:

1. Weight reconstruction error:
   - relative MSE;
   - max absolute error;
   - per-output-channel error.
2. Activation reconstruction error:
   - run baseline activations through original and quantized module weights;
   - compare MSE, cosine similarity, max absolute error;
   - record separately for prefill and one-token decode.
3. Logit drift:
   - KL divergence between baseline and quantized next-token distributions;
   - top-1 flip rate;
   - top-k overlap;
   - selected-token logit delta.
4. Task-level regression:
   - exact token match for deterministic prompts;
   - reasoning answer correctness;
   - OCR and visual detail preservation.
5. Performance:
   - load memory;
   - prefill tok/s;
   - output tok/s;
   - synchronized model-forward seconds;
   - Metal command/kernel count if available.

The primary sensitivity artifact should be JSON:

```json
{
  "model_id": "openbmb/MiniCPM-V-4.6",
  "revision": "...",
  "calibration_set": "...",
  "modules": [
    {
      "name": "model.language_model.layers.0.linear_attn.in_proj_qkv",
      "family": "text.deltanet",
      "candidate_quant": "q4k",
      "weight_error": {},
      "activation_error": {},
      "logit_kl": 0.0,
      "top1_flip_rate": 0.0,
      "latency_delta": 0.0,
      "recommended_policy": "q5k"
    }
  ]
}
```

## Precision Policies

Start conservative. Policy names should be explicit and reproducible.

### `bf16-baseline`

Current runner behavior. This is the correctness and performance reference.

### `q8-text-linears`

Quantize only text linear weights to Q8. Keep embeddings, LM head, RMSNorm,
DeltaNet state tensors, vision, and merger in BF16/F32. This validates loader and
matmul plumbing before quality risk.

### `q4k-mlp-only`

Quantize text MLP `gate_proj`, `up_proj`, and `down_proj` to Q4K. Keep attention
and DeltaNet projections in BF16. This tests a high-parameter area without
touching recurrent state behavior.

### `q4k-text-safe`

Quantize text MLP and full-attention `q/k/v/o` projections. Keep DeltaNet
`in_proj_a`, `in_proj_b`, `dt_bias`, `A_log`, conv weights, norms, embeddings,
LM head, vision, and merger in BF16/F32.

### `dynamic-4-5-text`

Use sensitivity scores to choose Q4K for low-sensitivity text linears and Q5K or
Q6K for high-sensitivity text linears. Keep VLM modules protected.

### `dynamic-vlm`

Only after text policies pass: allow selected vision-tower and merger linears to
move to Q8 or Q5/Q6. Use OCR and image-description regressions as gates.

## Protected Modules Initially

Do not quantize these in the first variants:

- token embeddings and tied LM head;
- RMSNorm / LayerNorm weights;
- RoPE tables and scalar constants;
- DeltaNet `A_log` and `dt_bias`;
- DeltaNet recurrent and conv state;
- image processor outputs;
- vision tower;
- MiniCPM merger / multimodal projection;
- MTP modules, because this checkpoint has no MTP weights.

The protected list can shrink only after sensitivity and task data justify it.

## Benchmark Matrix

Every policy must be compared against `bf16-baseline` with:

- `cargo run --release --features metal -- logits --fail-on-mismatch`
- `cargo run --release --features metal -- bench --profile short --profile medium --profile long`
- one `profile --profile long` synchronized component report;
- image/OCR evals once multimodal next-token parity exists;
- memory report: total weight bytes by precision family.

Pass gates:

- top-1 logits parity on text fixtures, or documented top-k/logit drift if exact
  parity is impossible;
- no deterministic prompt regression;
- no OCR/image regression for policies touching vision/merger modules;
- median output tok/s improves by at least 10% or memory drops by a meaningful
  target without speed regression;
- sensitivity artifact and policy manifest are committed.

## Required Implementation Tickets

1. `build-minicpm-quantization-calibration-set`
   - Create deterministic text and image calibration fixtures.
   - Store prompt/media metadata under `evals/calibration`.
2. `score-minicpm-quantization-sensitivity`
   - Add a command that runs module-level quant simulation and emits the
     sensitivity JSON artifact.
3. `convert-minicpm-mixed-precision-weights`
   - Convert native safetensors to a mixed-precision artifact with a policy
     manifest.
   - Start with Q8 and Q4K text linears only.
4. `load-minicpm-quantized-linear-weights`
   - Teach the runner to load quantized linear weights beside BF16 tensors.
   - Keep non-linear, norm, embedding, vision, and merger tensors unchanged.
5. `benchmark-metal-quantized-matmul-kernels`
   - Measure Candle's existing Metal quantized MV/MM paths for MiniCPM shapes.
   - Decide whether BF16/F16 activation kernels are needed before custom work.

## Kernel Priorities

Do not write custom kernels first. Use the existing Candle QMatMul path to answer
whether quantized weights help on this model.

Write or port kernels only if measurement shows:

- F32 activation casts erase the gain from quantized weights;
- decode MV is faster but prefill MM regresses;
- Q4K/Q5K output dtype conversion dominates;
- DeltaNet projection matmuls become a top bottleneck after recurrent-state work.

If custom kernels become necessary, first target:

1. BF16/F16 activation x Q4K/Q5K weight matvec for one-token decode.
2. BF16/F16 activation x Q4K/Q5K matrix multiply for prompt prefill.
3. Optional fused dequant + projection patterns for DeltaNet grouped projections.

## Pivot Rule

Quantization is worth implementing when either:

- memory pressure blocks larger batch, multimodal, or draft experiments; or
- profiles show projection/MLP matmuls or bandwidth dominate after the current
  DeltaNet recurrent path is optimized.

Given the current Metal profile, quantization is not the first speed lane. It is
still worth designing now because the sensitivity artifacts and mixed-precision
manifest will be reusable once the main recurrent bottleneck is lower.
