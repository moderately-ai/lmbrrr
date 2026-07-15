---
id: design-ternary-bonsai-e2e-bringup
title: "DESIGN: end-to-end Ternary-Bonsai-27B spec-decode bring-up + eval"
status: todo
priority: p3
dependencies: [qwen35-27b-config-scaleup, ingest-external-dspark-head, metal-ternary-matmul-kernel]
related: [ternary-bonsai-27b-support]
scopes: [runtime/candle, evals]
shared_scopes: []
paths: []
tags: [ternary-bonsai, model-compat, dspark]
---
## WHY

The capstone that ties the pieces together and proves the compatibility claim: a ternary 27B target + its bf16 DSpark head running speculative decode in lmbrrr, with an eval confirming coherence and measured acceptance. Also the place to design the integration surface so the next external model reuses it.

## WORK ITEMS (design, then execute)

1. **Wiring**: ternary target ([[qwen35-27b-config-scaleup]] + [[metal-ternary-matmul-kernel]]) as the verify model, external DSpark head ([[ingest-external-dspark-head]]) as drafter, sharing the 5 target-layer hidden taps the head's fc needs. Reconcile with our existing MTP/DSpark loop (which taps verify-pass hiddens) — the tap points differ (5 fixed layers vs last-layer).
2. **Tokenizer**: gpt2 BPE, 248320 vocab, `qwen35` pre, bos 248044 — load from the GGUF tokenizer metadata; confirm chat template (embedded in the GGUF) round-trips.
3. **Scope cut**: text-only decode — the VLM `mmproj` projector is a non-goal for bring-up.
4. **Validation**: (a) target greedy output coherent + matches the F16 reference within ternary error on a few prompts; (b) end-to-end spec decode produces identical committed text to target-greedy with a measured mean-accepted-length; (c) rough decode tok/s on the M-series vs a llama.cpp baseline for the same GGUF (apples-to-apples, per the eval-apples-to-apples methodology).

## DONE-WHEN

Ternary-Bonsai-27B generates coherent text under lmbrrr spec decode on the M-series, validated vs the F16 reference and with a tok/s number; the integration path is documented as the template for future external models. Feeds the broader model-compat theme.
