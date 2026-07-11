---
id: quantize-full-text-decoder-q4-incl-lm-head
title: Quantize full text decoder to Q4 including lm_head
status: done
priority: p1
dependencies: []
related: [bf16-activation-quantized-matmul-metal]
scopes: [quantization, runtime/candle, evals]
shared_scopes: [docs/research]
paths: [src/quant_convert.rs, src/quant_sensitivity.rs, src/quantized_linear.rs, src/main.rs, evals/**, docs/research/q4-full-decoder-policy.md]
tags: [quantization, performance, campaign-1000]
---
## Outcome (2026-07-11, commit 660d0c1 — core landed)

lm_head wiring shipped as runtime --quantize-lm-head (quantized copy of the tied embedding at load; BF16 gather table untouched; MixedLinear slot at both call sites; ModelArgs-level so all subcommands honour it). Measured greedy ladder same-session: bf16 145.7 -> q8 head 163.4 -> q4k head 172.7 -> q4k head + existing q4k-mlp-q8-text body manifest = 189.6 tok/s (+30%). Quality advisory: q8 head tracks greedy 637 chars; q4k-full forks early, text fully coherent. Full quantized spec stack: greedy 190-192, spec 112 math (0.59x) — ratio drop as predicted; tau dips under drafter-target quantization mismatch (tides 1.03) — counter is tree/tau work + a quantized-verify cost model. REMAINING (formalization, not perf): Q4KFullText policy variant + from_source manifest formats for artifact hygiene; quant-quality run with the fallback ladder; update the cost-model artifact for quantized verify costs.

## Implementation plan (agent-verified, 2026-07-10 evening)

Checkpoint has NO lm_head.weight (tied: minicpm.rs:453-461 always clones the embedding) — quantized head = quantized COPY of embed_tokens; the BF16 table stays for the gather. Plan: (1) Q4KFullText policy variant (quant_convert.rs:18-34) skipping the sensitivity-set gate (protections advisory per campaign), skipping in_proj_a/b (32KB, decay-gate sensitivity); NEW `{q4k,q6k,q8_0}_from_source` manifest formats referencing the source safetensors — avoids a 143MB artifact duplicate AND the existing lmbq->f32->GGML double quantization (quantized_linear.rs:172), and gives the q6k/q8 head fallback ladder for free. (2) lm_head slot: Linear -> MixedLinear (minicpm.rs:445, .apply at 537/549), artifact hook after the layer pass; optionally consume the QMatMul's F32 logits directly (sampling casts anyway - saves a 248k cast/token). (3) quality run advisory-except-collapse with --head-format fallback rung. Fused DeltaNet kernels need ZERO changes (they consume projection outputs; MixedLinear must keep returning input dtype). Expected: weight reads 1502->~422 MB/token; decode ~7.0 -> 4.5-5.3 ms (190-220 tok/s) BEFORE the fork's BF16-activation work; +Step-4 fork => 220-260. Caveat for the spec lane: target quantization raises greedy proportionally more than verify — pair with drafter-side quantization (cut-drafter-propose-cost) so spec keeps pace.

## Update (decode audit, 2026-07-10)

No wiring exists yet for the headline item: `lm_head` is a plain `candle_nn::Linear` built from tied embeddings (minicpm.rs:445,453-461) and `apply_quantized_text_artifact` (minicpm.rs:501-506 -> qwen35.rs:1360-1369) only iterates decoder layers — the 508 MB/token read (34% of all weight bytes, ~1.45 ms) has no quantization hook. Add a MixedLinear slot / replacement path for lm_head as part of this ticket. Also: `MixedLinear::from_qtensor` sets `force_f32_input` (quantized_linear.rs:39-59), so the quantized path adds 2 casts per projection (~50 extra dispatches/token model-wide) — the quantized gemv should accept BF16 activations (bf16-activation-quantized-matmul-metal) or activations should ride F32 through the layer.

## Goal

Cut per-token weight reads from ~1.5 GB BF16 to ~0.45 GB by quantizing every text linear AND the lm_head (248k vocab x 1024 = 0.51 GB BF16, currently protected) to Q4K, lifting the bandwidth roofline from ~270 to ~900 forwards/s. Campaign quality bar: quality is reported, not gating; protections become advisory.

## Acceptance

- New policy `q4k-full-text` covering all text linears + lm_head (tied embedding stays BF16 for the gather; only the matmul view is quantized).
- Per-tensor fallback ladder (q6k/q8) applied only where generation collapses outright (empty/looping output), chosen by the quality harness, not by logit-delta thresholds.
- `quant-quality` report for the policy (advisory) plus decode/prefill bench vs dense and vs the old q4 policies.
- Memory + bytes-per-forward accounting in the manifest so the roofline note can compute the new ceiling.
