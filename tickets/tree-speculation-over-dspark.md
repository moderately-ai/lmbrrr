---
id: tree-speculation-over-dspark
title: Tree speculation over DSpark
status: todo
priority: p1
dependencies: [integrate-dspark-block-runner]
related: [remeasure-spec-round-cost-model]
scopes: [inference/speculative, runtime/candle, evals]
shared_scopes: [docs/research]
paths: [src/main.rs, src/qwen35.rs, evals/dspark/**, docs/research/tree-speculation-dspark.md]
tags: [speculative, dspark, campaign-1000]
---
## Board revision (2026-07-10 evening)

With tau frozen at ~2.1 (position acceptance [0.63,0.28,0.21,0.11,0.04], chain cap ~2.27 tokens/round), this is the largest no-training tokens-per-round lever. Target reset: tau_eff >= 3.0 (the old 4.5 assumed a stronger drafter); compute the go/no-go from the rebuilt cost table, not the superseded one. The target is BF16, not quantized — the premise stands because the fused chunk kernel (l<=12) made verify tokens cheap, but note it handles LINEAR chunks: tree verification multiplies chunk cost per flattened path, so the "measure both" clause below is the crux. Add: rederive the trajectory-oracle noise bound if tree verification changes numerics (envelope already 0.5/0.75).

## Goal

Raise effective accepted length past chain tau by verifying a tree of DSpark candidates per round. Post-fusion, verify chunk marginal cost is small, so branching where the Markov head is uncertain converts cheap verify compute into accepted tokens.

## Acceptance

- Draft-tree construction from Markov-head top-k branching at low-confidence positions (budgeted by the verification throughput table's measured chunk costs).
- Tree verification through the target: full-attention layers via a tree attention mask; DeltaNet layers via per-path chunk re-advance or path flattening — measure both, document which wins and at what tree depth the recurrent-state cost caps the tree.
- Longest-accepted-path selection with exact state re-advance for the chosen path, gated by the multi-round greedy oracle.
- Report tau_eff vs chain tau per prompt class; target tau_eff >= 4.5 on math/code.
