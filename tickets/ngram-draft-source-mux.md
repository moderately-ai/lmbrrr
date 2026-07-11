---
id: ngram-draft-source-mux
title: "Draft-free n-gram speculation: prompt-lookup + verify-logit token recycling"
status: done
priority: p1
dependencies: []
related: [verify-logit-token-recycling-draft-source]
scopes: [inference/speculative]
shared_scopes: [docs/research]
paths: []
tags: [speculative, campaign-1000, frontier-survey]
---
## Goal
Chain-mode draft sources requiring no training: (1) prompt-lookup (suffix n-gram match against prompt+generation, propose following span, cap 12 tokens to fit closed-form rollback); (2) token recycling (adjacency matrix of top-k candidates from verify logits we already compute, <2 MB). Drafts fire only on match -> strict greedy floor, no probe tax. Mux per-round under the existing hardware-aware scheduler (RASD pattern): n-gram match > DSpark drafter > greedy.

## Acceptance
- CPU-side lookup structures; no new Metal kernels; exact argmax verification unchanged.
- Measured per protocol on the 4-class suite: expect 1.5-2.5x on summarization/code/grounded, ~1.0-1.2x prose floor; must never lose to greedy on any class.
- Composes with (not replaces) the DSpark drafter; scheduler EV gate decides per round.
- Refs: PLD (apoorvumang), vLLM spec-decode blog, Token Recycling 2408.08696, RASD 2503.03434, Snakes & Ladders (SSM chain-verify precedent).
