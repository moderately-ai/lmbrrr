---
id: cut-drafter-propose-cost
title: Cut drafter propose cost (8.7 ms/round > a full greedy token for a 2-layer drafter)
status: todo
priority: p1
dependencies: []
related: [bf16-activation-quantized-matmul-metal, remeasure-spec-round-cost-model]
scopes: [inference/speculative, runtime/candle]
shared_scopes: [docs/research]
paths: []
tags: [speculative, performance, campaign-1000]
---
## Goal

Propose costs 8.7 ms/round — more than a full 24-layer greedy token (6.9 ms) for a 2-layer drafter — almost certainly dominated by the 248k-vocab reads: lm_head [248094,1024] bf16 (508 MB) once per proposal plus markov_w2 [248094,256] (127 MB) per Markov step. At the chain cap of ~2.27 tokens/round, every ms off propose is ~13 tok/s of spec throughput. None of these levers touch output correctness: drafts are verified, so a marginally worse draft distribution costs only tau.

## Levers (shape decided by remeasure-spec-round-cost-model's breakdown)

- Quantize the frozen drafter's lm_head and markov_w2 post-hoc (q4k/q8, no retraining); relates bf16-activation-quantized-matmul-metal (else the F32 cast tax eats the win).
- Narrowed-vocab / top-k draft sampling (correct only the top-K of base logits before the Markov bias; fidelity caveat documented).
- Batch the per-step Markov lm_head applications where the chain allows.
- Remaining draft-side dispatch cleanup after the KV-cache/SDPA/packed-readback pass (commit 5218684).

## Acceptance

- Propose <= ~5 ms/round at gamma=8-width proposals, drafter parity harness still within thresholds (or updated thresholds with justification), tau on the fixed prompt suite within noise of today's 2.1.
- Spec wall-rate A/B per measurement protocol; results into the cost-model artifact.
