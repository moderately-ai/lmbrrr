---
id: quantize-full-text-decoder-q4-incl-lm-head
title: Quantize full text decoder to Q4 including lm_head
status: todo
priority: p1
dependencies: []
related: [bf16-activation-quantized-matmul-metal]
scopes: [quantization, runtime/candle, evals]
shared_scopes: [docs/research]
paths: [src/quant_convert.rs, src/quant_sensitivity.rs, src/quantized_linear.rs, src/main.rs, evals/**, docs/research/q4-full-decoder-policy.md]
tags: [quantization, performance, campaign-1000]
---
## Goal

Cut per-token weight reads from ~1.5 GB BF16 to ~0.45 GB by quantizing every text linear AND the lm_head (248k vocab x 1024 = 0.51 GB BF16, currently protected) to Q4K, lifting the bandwidth roofline from ~270 to ~900 forwards/s. Campaign quality bar: quality is reported, not gating; protections become advisory.

## Acceptance

- New policy `q4k-full-text` covering all text linears + lm_head (tied embedding stays BF16 for the gather; only the matmul view is quantized).
- Per-tensor fallback ladder (q6k/q8) applied only where generation collapses outright (empty/looping output), chosen by the quality harness, not by logit-delta thresholds.
- `quant-quality` report for the policy (advisory) plus decode/prefill bench vs dense and vs the old q4 policies.
- Memory + bytes-per-forward accounting in the manifest so the roofline note can compute the new ceiling.
