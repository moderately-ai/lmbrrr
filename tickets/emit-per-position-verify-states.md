---
id: emit-per-position-verify-states
title: Emit per-position states from verify chunks to eliminate re-advance
status: done
priority: p1
dependencies: []
related: [implement-speculative-state-rollback, tree-speculation-over-dspark, integrate-dspark-block-runner]
scopes: [runtime/candle, inference/speculative]
shared_scopes: [docs/research]
paths: [src/qwen35.rs, src/main.rs, docs/research/per-position-verify-states.md]
tags: [speculative, dspark, performance, campaign-1000]
---
## Outcome (2026-07-10)

Landed (commit 847563d) as lazy closed-form reconstruction (not full materialization): verify chunks retain S0/kc/delta/gcs/conv-window under a runner-controlled capture flag; partial accept computes S_j at the accepted position (one narrow+matmul per layer) and truncates full-attention KV to snapshot_len+prefix (the chunk's rows are causally valid — kept, not rewritten). Legacy restore+re-advance behind LMBRRR_READVANCE_ROLLBACK=1. Design cross-checked by a fresh-agent audit before implementation (closed form == chunk-end update at j=C-1). Gates: 33/33, state-integrity oracle both paths/prompts, envelopes unchanged. Measured: rollback 25.1 -> ~2 ms; round-1 drafter 0.59x -> 0.85x greedy (math γ3, 66.4 tok/s), 0.41x -> 0.63x (tides). Break-even tau lowered as designed; verify (~24.6 ms) now dominates the round -> fuse-deltanet-decode-step-kernel / chunk-kernel lane.

## Goal

Remove the rollback re-advance forward entirely by making verify chunks emit restorable per-position DeltaNet states, so a partial accept selects the state at the accepted position instead of restoring the pre-chunk snapshot and re-running the accepted prefix.

## Context (measured, 2026-07-10)

The stub runner (docs/research/speculative-state-rollback.md) showed speculation breaks even only at tau ~= 4-5 when every round pays a rollback + re-advance; at tau ~= 3 it is 0.79x greedy. The re-advance is a full chunk forward of the accepted prefix. The chunked delta rule already computes cumulative decays and pseudo-values per chunk; materializing S at each position (or at least the accepted position, computable as gamma_k * S0 + (K[..k] * decay)^T Delta[..k] after the acceptance decision) makes rollback state-selection nearly free. The full-attention side is already free (KV truncate). This lowers the drafter-quality break-even from tau ~= 4-5 toward ~3 and removes the per-round rollback tax that tree speculation would otherwise multiply per path.

## Acceptance

- Chunked delta rule exposes a way to reconstruct the recurrent state at an arbitrary position k inside the last chunk without re-running the prefix (closed-form from the chunk intermediates, computed lazily after the acceptance decision).
- Conv state at position k likewise reconstructed from the chunk input window (last kernel-1 tokens up to k).
- `dspark-run` partial-accept path uses state selection instead of restore + re-advance; the corruption-invariance oracle still passes and per-round timings show the re-advance term eliminated.
- Re-measure the stub-runner table; document the new break-even tau.
