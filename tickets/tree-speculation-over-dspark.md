---
id: tree-speculation-over-dspark
title: Tree speculation over DSpark
status: todo
priority: p1
dependencies: [integrate-dspark-block-runner]
related: [profile-dspark-verification-throughput-table]
scopes: [inference/speculative, runtime/candle, evals]
shared_scopes: [docs/research]
paths: [src/main.rs, src/qwen35.rs, evals/dspark/**, docs/research/tree-speculation-dspark.md]
tags: [speculative, dspark, campaign-1000]
---
## Goal

Raise effective accepted length past chain tau by verifying a tree of DSpark candidates per round. On a bandwidth-bound quantized target, verifying 16-32 tree tokens costs barely more than one token, so branching where the Markov head is uncertain converts nearly-free verify compute into accepted tokens.

## Acceptance

- Draft-tree construction from Markov-head top-k branching at low-confidence positions (budgeted by the verification throughput table's measured chunk costs).
- Tree verification through the target: full-attention layers via a tree attention mask; DeltaNet layers via per-path chunk re-advance or path flattening — measure both, document which wins and at what tree depth the recurrent-state cost caps the tree.
- Longest-accepted-path selection with exact state re-advance for the chosen path, gated by the multi-round greedy oracle.
- Report tau_eff vs chain tau per prompt class; target tau_eff >= 4.5 on math/code.
