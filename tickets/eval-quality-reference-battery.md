---
id: eval-quality-reference-battery
title: "EVAL: quality reference battery — perplexity + text-agreement vs bf16, the standing gate for any non-bit-preserving change"
status: todo
priority: p1
dependencies: []
related: []
scopes: [evals, quantization]
shared_scopes: []
paths: []
tags: [eval-wave, quality]
---
WHY: our quality instruments are logit-local (transformers parity oracle, stub-oracle 0.75 bound at 128 tokens, per-linear quant sensitivity) or anecdotal (argmax coverage on two classes for the 32k head). NOBODY has measured what the deployed q4k-full-text stack costs end-to-end vs the bf16 reference in perplexity or long-form text agreement. Three queued lines of work are non-bit-preserving and currently have NO quantitative ship gate: imatrix-calibration-pipeline (q3 mixes), any MLX-format migration out of eval-apples-to-apples-qmv, and any future head/vocab trick. This battery is their gate, and its first run establishes the baseline we should have had all along.

PART 1 — BUILD `lmbrrr ppl` (teacher-forced perplexity):
- New subcommand in src/commands/ (clone quant.rs arg-handling style): `lmbrrr ppl [--quantized-manifest ... --quantize-lm-head q4k] --corpus evals/calibration/minicpm_v46_quant_calibration.jsonl --max-tokens 20000 --chunk 256`.
- Method: tokenize each corpus text row; run the existing chunked prefill path; per chunk compute log_softmax over the vocab and gather the target-token logprob for every position (NOTE: the prefill path narrows hidden to the LAST position before lm_head as an optimization — the ppl path must bypass that narrowing and apply lm_head to all positions in the chunk; 256 x 248094 bf16 logits = 127 MB per chunk, fine). Accumulate mean NLL in f64 on host; report ppl = exp(mean NLL), token count, per-row breakdown.
- Determinism: same chunking for every arm (chunk boundaries change bf16 rounding; both arms MUST use identical --chunk).
- Reference arm: same command WITHOUT --quantized-manifest/--quantize-lm-head (the runner's bf16 safetensors path; see ModelArgs in src/main.rs for the model-path flag/env if it is not defaulted). If the bf16 path can't fit or has bit-rotted, fixing that is IN SCOPE for this ticket — the reference arm is the point.

PART 2 — TEXT-AGREEMENT BATTERY: greedy 256-token generations on the 18 validation-suite prompts (evals/prompts/), bf16 arm vs deployed q4k arm. Report: exact-match rate, mean first-divergence token index, and eyeball-diff of the 3 worst prompts pasted here.

BASELINE RUN (the deliverable): bf16 vs deployed q4k-full-text — record ppl delta (expect small, ~1-3% relative, based on the parity-oracle history; if >5% STOP and investigate before trusting any downstream gate) and the agreement stats, as a comment here + campaign log.

STANDING GATE (copy into any non-bit-preserving change's procedure): candidate ships without escalation iff Delta-ppl(candidate vs deployed) <= +1% relative AND first-divergence index distribution not visibly worse; anything beyond that needs an explicit user quality/speed decision with these numbers attached. Cross-reference: imatrix-calibration-pipeline MUST run this battery per candidate mix (comment added there).
