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

## Oracle v2: state-integrity form (2026-07-10, later the same day)

The bitwise form above is **prompt-sensitive by construction**, root-caused when a new validation prompt ("Explain how tides work.", 96 tokens) failed on every binary back to the exact commit validated above, rebuilt against crates.io candle — deterministic, identical divergence at token 69, coherent text on both sides. Bitwise cross-pattern equality holds only if no committed token in the horizon sits within kernel noise of its runner-up under *any* tested chunk split. The math prompt above happens to have no such token in 160 tokens; the tides prompt has one at 69. Any kernel change (gemv routing, fork rebase, future fusion/quantization) re-rolls which prompts carry sub-noise ties, so the bitwise gate degrades into a per-prompt coin flip precisely as the campaign changes kernels — while telling you nothing about rollback.

The v2 gate (commit e4e6327) tests the property rollback must actually guarantee: **the target's logits at a committed position depend only on the prefix, never on the chunk split**, so across corruption patterns the top-8 logit values at every shared committed position must agree to within kernel noise. A real restore bug perturbs the whole trajectory (argmax flip or not) and fails loudly; a token divergence is benign only when both runs' top-2 margins sit inside the noise bound, after which the streams legitimately fork and comparison stops. This is strictly more sensitive than the bitwise form (it catches sub-argmax state corruption at every shared position) and robust to kernel churn.

Measured calibration (BF16, Metal, both prompts): max top-8 trajectory deviation across all shared positions = 0.25 (tides) / 0.375 (math); every observed token divergence is a genuine tie (one side at margin 0.0, other ≤ 0.375); noise bound set at 0.75 (~6 BF16 ulps of a top logit near 32) with all measurements reported in the JSON for ongoing calibration. Verdict: **rollback machinery confirmed sound under the fork kernels** — all divergences ever observed are tie-flips, trajectories agree within 3 ulps.

Protocol lesson recorded: validating a numerics-affecting change on a *new* prompt without first running the unchanged binary on that prompt (the control) wasted a full investigation cycle blaming the change for a pre-existing prompt property.

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
