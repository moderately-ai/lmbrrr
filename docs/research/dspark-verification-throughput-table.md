> **SUPERSEDED (2026-07-10):** pre-fusion measurements, ~5x off current kernels. The live artifact is `artifacts/spec-round-cost-model.json` (see ticket remeasure-spec-round-cost-model).

# DSpark Verification Throughput Table

Date: 2026-07-10

Ticket: `profile-dspark-verification-throughput-table`

Command: `cargo run --release --features metal -- verify-table --output target/verify-throughput-table.json` (BF16, Metal, single request; 1 warmup + 5 measured iterations per cell, medians; chunk content is the real greedy continuation per profile).

## Measured T_verify(γ)

| γ | short (ctx 27) | medium (ctx 53) | long (ctx 102) | chunk tok/s (long) | per-token efficiency vs γ=1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18.2 ms | 16.3 ms | 16.9 ms | 59 | 1.0 |
| 2 | 31.1 ms | 31.4 ms | 33.9 ms | 59 | 1.0 |
| 4 | 44.6 ms | 44.5 ms | 43.0 ms | 93 | 1.6 |
| 8 | 67.0 ms | 68.7 ms | 68.8 ms | 116 | 2.0 |
| 16 | 117.7 ms | 119.1 ms | 117.4 ms | 136 | 2.3 |
| 32 | 214.3 ms | 213.9 ms | 209.5 ms | 153 | 2.6 |

Context length (27–102 tokens) has no measurable effect at these sizes. The curve fits **T_verify(γ) ≈ 11 ms + 6.3 ms·γ**: a fixed ~11 ms floor plus a strongly linear marginal cost.

## The headline finding

The marginal verify token costs ~6.3 ms — 37% of a full decode step — because the seq>1 path through the 18 GatedDeltaNet layers is a sequential per-token loop (`recurrent_delta_rule`), so a "parallel" chunk is mostly serial. There is no cliff to schedule around; it is a slope.

Consequence for the campaign, worked through: with today's target, a chain-DSpark round at γ=8 costs ~1 ms draft + 68 ms verify; even at a strong τ=5 that is 13.8 ms/token ≈ 72 tok/s — barely above the 60 tok/s greedy baseline. **Speculation cannot pay on this target until the marginal verify token gets much cheaper.** `optimize-deltanet-chunked-prefill-and-verify-throughput` (chunked delta rule + grouped conv, then a fused Metal kernel) is therefore the campaign's critical-path ticket, ahead of drafter quality: the attention layers and MLPs already batch well, so a chunked DeltaNet should push per-token efficiency from 2–2.6× toward the 5–15× a bandwidth-bound verify implies.

## Scheduler contract

`target/verify-throughput-table.json` rows carry `median_verify_seconds` per (profile, γ); the scheduler consumes `T_round(γ) = T_draft + T_verify(γ)`. Single-request only; batched verify lands with `batched-multi-stream-decode-runner`, and this table must be regenerated under the q4 policy.

## Update 2026-07-10: after the chunked DeltaNet recurrence

`optimize-deltanet-chunked-prefill-and-verify-throughput` landed the same day (see docs/research/deltanet-chunked-recurrence.md). The regenerated table now fits **T_verify(γ) ≈ 15.6 ms + 0.80 ms·γ** — the marginal verify token dropped 6.3 → 0.80 ms, γ=32 chunks verify at 784 tok/s (12.5× per-token efficiency), and the "speculation cannot pay" conclusion above is retired: a γ=8 round at τ=5 now projects ~158 tok/s on the BF16 target. `target/verify-throughput-table.json` holds the post-chunking numbers; the pre-chunking artifact is preserved in this note's table.
