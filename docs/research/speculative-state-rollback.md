# Speculative State Rollback and Multi-Round Loop

Date: 2026-07-10

Ticket: `implement-speculative-state-rollback`

## What landed

- `TruncatableKvCache` (src/qwen35.rs) replaces candle_nn's KvCache in `FullAttention`: grow-on-demand capacity, `slice_set` append, `truncate(len)` rewind. Also fixes a memory problem — candle's cache preallocates max_position_embeddings (262k) per layer. Strict logits parity is unchanged.
- `Qwen35TextModel::snapshot_decode_state` / `restore_decode_state`: DeltaNet conv/recurrent tensors snapshot as Arc clones (they are replaced by assignment, never mutated in place), full-attention layers snapshot only their KV length (rollback rewinds; the re-advance chunk overwrites the stale slice). Snapshots are effectively free.
- `dspark-run` subcommand: the real multi-round speculative loop with a stub drafter. Chunks follow the DeepSpec convention ([anchor, d1..dγ] fed; logits at position i verify draft i+1; bonus token becomes the next anchor). Partial accept restores the snapshot and re-advances the accepted prefix in one chunk; full accept keeps the advanced state (fast path). No prompt re-prefill ever.

## The oracle

Equality with the plain greedy baseline cannot be the blocking gate: chunk-path and decode-path logits legitimately tie-flip (observed once in 160 tokens). The blocking oracle is **corruption-pattern invariance**: under exact-match acceptance every committed token is target-chunk-derived, so runs whose stub drafts are corrupted at different periods must produce identical output iff state rollback is sound. `dspark-run` executes corrupt-every ∈ {0, 3, 5} and fails hard on divergence.

Results (BF16, Metal, γ=8): oracle **passed** on both prompts — long run: 54/54 rounds rolled back under corrupt-every=3 with output still token-identical to the uncorrupted run; advisory baseline match 160/160 (long) and 12/13 (short — one expected tie-flip).

## First end-to-end speculative wall-clock (long prompt, 160 tokens)

| Stub pattern | τ (mean) | rounds | rollbacks | tok/s | vs 55.7 baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| perfect drafter | 8.89 | 18 | 0 | **145.1** | **2.6×** |
| corrupt every 5 | 5.0 | 32 | 32 | 73.7 | 1.32× |
| corrupt every 3 | 2.96 | 54 | 54 | 43.9 | **0.79×** |

Readings:

1. **The runner-loop ceiling on today's BF16 target is ~145 tok/s** (perfect drafter, γ=8). Quantization and GEMV/fusion work multiply this ceiling directly.
2. **Break-even is τ ≈ 4–5 when every round pays a rollback + re-advance.** These stub patterns force a rollback every round (worst case); a real drafter's full-accept rounds take the fast path. Still, this sets a hard drafter-quality bar for `benchmark-full-dspark-speedup` and quantifies the re-advance tax.
3. Future optimization (filed as a note on the tree-speculation ticket's territory): the chunked delta rule could emit per-position intermediate states during verification, making rollback state-selection free and removing the re-advance forward entirely — worth ~30% at low τ.

The trained drafter drops into this verified loop via `integrate-dspark-block-runner` (backbone/Markov/confidence inference replacing the stub).
