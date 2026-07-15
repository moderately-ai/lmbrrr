---
id: ingest-external-dspark-head
title: "FEATURE/SPIKE: ingest the external Bonsai DSpark head + reconcile with our DsparkDrafter"
status: todo
priority: p2
dependencies: [gguf-loader-qwen35-hybrid]
related: [ternary-bonsai-27b-support, spec-loop-economics-recovery]
scopes: [runtime/candle]
shared_scopes: []
paths: [src/dspark.rs]
tags: [ternary-bonsai, model-compat, dspark]
---
## WHY

`Bonsai-27B-dspark-bf16.gguf` (arch `dspark`, bf16, 3.646B) is an **independently-trained full-DSpark head** — a rare external reference for the faithful-DSpark directive. Loading it (a) tests our `DsparkDrafter` against weights we didn't produce, and (b) surfaces exactly which DSpark mechanisms our implementation already has vs is missing. This is both a compatibility win and a spec-conformance check.

## REFERENCE HEAD ANATOMY (from the GGUF)

6 transformer blocks (`blk.0-5`, Qwen3-style GQA 40:4 + QK-norm) + own `token_embd`/`output`/`output_norm` + DSpark machinery:
- `fc` [25600→5120] = **5-layer feature fusion** over target hiddens `target_layers [1,16,31,46,61]`.
- `markov_head_a/b` [256,248320] = rank-256 low-rank bigram bias; `markov_rank 256`.
- `confidence_head` [5376→1] over hidden⊕markov (`confidence_head_with_markov`).
- `log_snr_fc1/fc2` = log-SNR (diffusion) conditioning, min/max ±9.
- `block_size 4`, `mask_token_id 248319` = masked-block (non-AR) drafting.

## WORK ITEMS

1. GGUF → `DsparkConfig`/`DsparkDrafter` name map (`dspark.fc`, `dspark.markov_head_a/b`, `dspark.confidence_head`, `dspark.log_snr_fc1/2`, `dspark.hidden_norm`, `blk.N.*`). `src/dspark.rs` already models block_size / mask_token_id / markov_rank / target_layer_ids / confidence-with-markov — confirm each maps.
2. **Gap analysis vs our DsparkDrafter**: does ours support (a) **5-layer feature fusion** (fc 25600 = 5×hidden — our MTP fc is 2×hidden; DsparkDrafter's target_layer_ids suggests yes, verify), (b) **log-SNR conditioning**, (c) **masked-block diffusion** drafting? Enumerate present vs missing; missing pieces become follow-up tickets.
3. Requires the target running to produce the 5 target-layer hiddens the fc consumes ([[qwen35-27b-config-scaleup]]) — can validate the head forward against a synthetic hidden stack first.

## DONE-WHEN

The external head loads into our drafter (or a documented list of missing mechanisms is filed as follow-ups), and a forward on target hiddens produces sane draft logits. Confirms/extends the faithful-DSpark posture with an external witness.
