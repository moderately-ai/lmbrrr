# Experiment Pivot Plan

Date: 2026-07-07

This note defines when MiniCPM-V-4.6 Candle runner work should move from
baseline hardening into experimental inference changes such as DSpark-style
speculative decoding, EAGLE-style draft heads, dynamic quantization, or custom
Metal kernels.

The short version: we can start text-only performance hardening now, but we
should not claim experimental speedups until text correctness, measurement, and
profiling gates are closed. Multimodal experiments have their own gate and
should not be used as evidence until image parity is stronger.

## Current Status

| Gate | Status | Evidence | Missing |
| --- | --- | --- | --- |
| Text prompt/token parity | Passed | `evals/fixtures/minicpm_v46_text_prompts.json` and `tests/minicpm_v46_text_parity.rs` match the MiniCPM chat template and tokenizer. | Nothing for prompt/token formatting. |
| Text logits parity | Passed | `evals/fixtures/minicpm_v46_transformers_text_logits.json` records live Transformers top-10 next-token logits for three text prompts; `lmbrrr logits --fail-on-mismatch` passed on Metal with top-1 agreement, top-10 overlaps of 9/10, 9/10, and 10/10, and max shared logit delta 0.25. | Nothing for initial text logits coverage. |
| Multimodal prompt expansion | Partial | `evals/fixtures/minicpm_v46_transformers_image_expansion.json` records the model-card refract image expanding from 19 template tokens to 211 processor tokens at `16x`; Rust expansion shape is covered synthetically. | Pixel/target-size parity, representative aspect ratios, visual feature insertion, and multimodal next-token or hidden-state comparison. |
| Measurement harness | Partial | `lmbrrr bench` records JSONL rows with prefill rate, TTFT, decode/output rate, token counts, device, dtype, revision, and generation settings. | Release-mode baseline matrix, variance policy, and generation-loop overhead isolation. |
| Profiling | Passed for initial text decode | `docs/research/metal-decode-hot-path-profile.md` ranks the Metal decode path: linear-attention/DeltaNet layers are ~81% of synchronized component time, DeltaNet recurrent rule is ~35%, full-attention matmul/softmax is ~1.7%, and argmax/scalar transfer is below 1%. | Exact Metal kernel launch counts still require Xcode/Metal capture or lower-level Candle tracing. |

## Pivot Rule

Text-only experimental optimization can begin after all of these are true:

1. Text correctness gate is closed.
2. Measurement gate is closed for the target hardware and dtype.
3. Profiling gate identifies the top bottleneck with enough evidence to choose
   between generation-loop cleanup, DeltaNet work, attention work,
   quantization, and speculative decoding.

Multimodal optimization can begin only after the multimodal correctness gate is
closed for the image/video path being optimized. Text-only gates are not enough
to justify VLM throughput or quality claims.

## Correctness Gates

### Text Gate

The text gate is closed when:

- Prompt strings and token ids match upstream Transformers for the committed
  short, open-thinking, long-reasoning, and image-marker prompt cases.
- Candle can dump the next-token logits for the same three text-only prompts as
  `evals/fixtures/minicpm_v46_transformers_text_logits.json`.
- For each prompt, Candle and Transformers agree on the top-1 token.
- For each prompt, at least 8 of the top-10 token ids overlap. If overlap is
  lower, the note must identify whether the mismatch is numerical tolerance,
  dtype/device behavior, an implementation bug, or tokenizer/prompt drift.
- For shared top-k tokens, logit differences are recorded. Use the observed
  distribution to set the final tolerance; do not hide a rank-changing mismatch
  behind a broad absolute threshold.

This initial text gate is now closed for the covered prompts. Keep using the
logits command after changes to decode math, caches, dtype handling,
quantization, speculative verification, or kernel code.

### Multimodal Gate

The multimodal gate is closed when:

- Processor outputs match Transformers for representative images:
  - model-card refract image;
  - square image without slicing;
  - portrait image with slicing;
  - landscape image with slicing;
  - high-resolution image near `max_slice_nums`.
- Target sizes, grids, number of patches, and expanded image-token counts match
  exactly or have documented, justified differences.
- Image placeholder replacement positions match the tokenized text.
- Pixel tensors are compared against Transformers after resize, normalization,
  and NaViT packing. Any tolerance must be based on actual resize/backend
  differences, not guessed.
- At least one multimodal prompt has a next-token top-k or hidden-state
  checkpoint comparison against Transformers.

Until this gate closes, multimodal output can be useful for debugging but should
not drive optimization decisions.

## Measurement Gate

The measurement gate is closed when:

- Benchmarks are run with `cargo run --release --features metal -- bench`.
- Interactive TUI output is not part of benchmark timing.
- Each comparison records model id, revision or snapshot, dtype, device,
  feature flags, prompt profile, prompt tokens, generated tokens,
  `max_new_tokens`, prefill seconds, prefill tokens/sec, TTFT, decode seconds,
  output tokens/sec, steady-state tokens/sec, EOS status, and generation
  settings.
- Each claimed comparison uses at least one warmup and at least five measured
  iterations per prompt profile.
- Results report median and spread, not just a single run. If decode
  tokens/sec has coefficient of variation above 5 percent, rerun or explain the
  source of instability.
- Before/after comparisons preserve prompt, generation settings, dtype, device,
  and output-token cap. If output length changes, compare both token-normalized
  speed and final generated text.
- Optimization claims require a material delta. Treat changes below 5 percent
  as noise unless profiling explains them; prefer changes above 10 percent for
  local runner work and above 20 percent before changing architecture direction.

## Profiling Gate

The profiling gate is closed when a decode profile ranks the top costs with
evidence. The minimum breakdown is:

- CLI/generation-loop overhead: token streaming, TUI, JSON writing, sampling,
  repeat penalty, tokenizer decode, and host/device synchronization.
- Model forward time per decoded token.
- Qwen3.5 Gated DeltaNet recurrent path.
- Full-attention layers and KV cache work.
- MLP/matmul-heavy projections.
- Metal kernel launch count and avoidable intermediate tensors.
- Memory movement or dtype conversion costs.

The profile must name one recommended first optimization and one tempting path
that is not justified yet. This prevents us from jumping to DSpark, DFlash,
quantization, or custom kernels before knowing which cost actually dominates on
the local hardware.

## Roadmap

### Before Text-Only Experimental Claims

1. Remove or isolate generation-loop overhead from `bench`, especially greedy
   argmax and avoidable host/device transfers.
2. Record a release-mode Metal baseline matrix for short, medium, and long
   prompts.
3. Choose the first optimization based on the profile:
   - if generation-loop overhead dominates, optimize the runner first;
   - if DeltaNet dominates, optimize recurrent decode;
   - if full attention dominates, consider attention/kernel work;
   - if matmuls or bandwidth dominate, design dynamic quantization;
   - if target decode is stable and correct, design speculative decoding.

### Before Multimodal Experimental Claims

1. Validate processor parity for representative image shapes.
2. Compare visual feature counts and insertion positions.
3. Add one multimodal next-token or hidden-state oracle comparison.
4. Add image/video benchmark profiles only after the above pass.

### After Gates Close

The first experimental lane should be selected by measured bottleneck, not by
paper novelty. The current text decode profile points to DeltaNet first:

- DeltaNet/custom Metal work is first if recurrent decode dominates.
- Dynamic quantization is first if matmul or memory bandwidth dominates.
- Speculative decoding is first if target-model decode is stable, correctness
  is proven, and draft/verify acceptance can be measured cleanly.
- DFlash-style attention work waits unless full-attention layers are a top
  contributor on this hybrid Qwen3.5/MiniCPM decoder.

## Immediate Next Tickets

1. Continue `optimize-generation-loop-overhead` now that text logits are
   comparable.
2. Run `profile-metal-decode-hot-path` once loop overhead is isolated enough
   that the profile reflects model execution.
3. Run `validate-minicpm-image-parity` before using multimodal behavior as
   evidence for optimization work.
