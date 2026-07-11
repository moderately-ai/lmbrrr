---
id: tree-speculation-over-dspark
title: Tree speculation over DSpark
status: todo
priority: p2
dependencies: [integrate-dspark-block-runner]
related: [remeasure-spec-round-cost-model]
scopes: [inference/speculative, runtime/candle, evals]
shared_scopes: [docs/research]
paths: [src/main.rs, src/qwen35.rs, evals/dspark/**, docs/research/tree-speculation-dspark.md]
tags: [speculative, dspark, campaign-1000]
---
## Board revision (2026-07-10 evening)

With tau frozen at ~2.1 (position acceptance [0.63,0.28,0.21,0.11,0.04], chain cap ~2.27 tokens/round), this is the largest no-training tokens-per-round lever. Target reset: tau_eff >= 3.0 (the old 4.5 assumed a stronger drafter); compute the go/no-go from the rebuilt cost table, not the superseded one. The target is BF16, not quantized — the premise stands because the fused chunk kernel (l<=12) made verify tokens cheap, but note it handles LINEAR chunks: tree verification multiplies chunk cost per flattened path, so the "measure both" clause below is the crux. Add: rederive the trajectory-oracle noise bound if tree verification changes numerics (envelope already 0.5/0.75).

## Progress (2026-07-10 night, commit cfeb779) + target-side execution plan

Drafter side LANDED: propose_branching returns the alternate chain from the runner-up Markov token at position 0 (parity 8/8 unchanged; zero cost on the default path). TARGET-SIDE PLAN (next fresh-context session): (1) plumb an optional external attention mask through forward_all_logits -> FullAttention (the masked-SDPA chunk path already takes an arbitrary materialized mask — build the ancestor mask for the flattened [anchor, a1..aw, b1..bw] layout where b* rows attend history+anchor only; siblings share RoPE positions). (2) GatedDeltaNet::forward_tree: run the fused chunk kernel on segment 1 [anchor, a1..aw] from S0, reconstruct state@prefix-1 from segment 1's OWN capture via the existing select_verify_state formula, run the chunk kernel on segment 2 [b1..bw] from that seeded state — 2 dispatches + reconstruction per layer, no weight re-reads (projections/MLP/lm_head run once over the flattened l=2w+1 <= 12). (3) Runner: verify both branches in the one forward; pick the longer-accepted path; KV compaction for the winner via slice_set (loser rows overwritten by next round); DeltaNet state = select at the winner's prefix within its segment capture. (4) Scheduler extension: branch only when position-0 survival is in the mid band (0.3-0.7) where the runner-up carries real mass. (5) Gates: state-integrity oracle (tree off), a new tree-vs-chain equivalence check (tree with branch pruned == chain bitwise), rederive the noise bound if needed. Cheap interim (optional): second-chance draft-skip — when the bonus equals alt_tokens[0], seed the next round's proposal from alt_tokens[1..] and skip the ~5ms draft (EV ~ +3-4%).

## Mechanism (spec-loop analysis, 2026-07-10 evening — the design decision)

Naively flattened tree chunks are UNFIXABLE for DeltaNet: the WY solve couples all chunk positions, so sibling tokens contaminate delta_i, and the closed-form reconstruction can only select a PREFIX. Path-by-path re-advance re-reads weights per path — dead. The viable mechanism is SEGMENT DECOMPOSITION with closed-form branch seeding: order nodes as contiguous root-to-leaf segments; run all linear projections once over the flattened [1,N,1024] (correct — each node's hidden depends only on ancestors); full-attention layers take an ancestor mask (the masked path accepts arbitrary masks; siblings share a RoPE position; winner-path KV compacted via slice_set); DeltaNet segments run the fused chunk kernel per segment, each non-root segment seeded from the branch-point state via the EXISTING select_verify_state formula on segment 1's capture. No weight re-reads. Host-composed branch restart costs ~2 ms today; in-kernel restart (the formula IS the kernel's chunk-end update — fork work: segment descriptor input) gets ~0.3-0.5 ms/branch. Drafter side is nearly free: base_logits/h_k computed once per block; a k-ary branch is one extra q8 Markov gemv + confidence eval, placed where calibrated confidence is low. Shapes within the l<=12 kernel budget: top-2@pos1 x depth 5, or top-3@pos1 x depth 3. Projected tau_eff ~2.45-2.6 (top-2), ~2.8 cap with this drafter. EV clears only after the L1/L2 round-floor work lands AND restart is in-kernel; host-composed is break-even.

## Goal

Raise effective accepted length past chain tau by verifying a tree of DSpark candidates per round. Post-fusion, verify chunk marginal cost is small, so branching where the Markov head is uncertain converts cheap verify compute into accepted tokens.

## Acceptance

- Draft-tree construction from Markov-head top-k branching at low-confidence positions (budgeted by the verification throughput table's measured chunk costs).
- Tree verification through the target: full-attention layers via a tree attention mask; DeltaNet layers via per-path chunk re-advance or path flattening — measure both, document which wins and at what tree depth the recurrent-state cost caps the tree.
- Longest-accepted-path selection with exact state re-advance for the chosen path, gated by the multi-round greedy oracle.
- Report tau_eff vs chain tau per prompt class; target tau_eff >= 4.5 on math/code.
