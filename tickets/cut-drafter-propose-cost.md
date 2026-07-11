---
id: cut-drafter-propose-cost
title: Cut drafter propose cost (8.7 ms/round > a full greedy token for a 2-layer drafter)
status: in-progress
priority: p1
dependencies: []
related: [bf16-activation-quantized-matmul-metal, remeasure-spec-round-cost-model]
scopes: [inference/speculative, runtime/candle]
shared_scopes: [docs/research]
paths: []
tags: [speculative, performance, campaign-1000]
claimed_from: todo
assignee: claude
lease_expires_at: 1783738814
---
## Outcome update (2026-07-10 night)

With the fork mv batching (ec0f74e5), q8 heads are clearly positive: gamma4+q8+--schedule = 115.9 tok/s math (0.78x greedy), draft 4.9 ms/round, tau unchanged. Acceptance target (propose <= ~5ms) MET at the operating point. Remaining stretch: Markov row-norm pruning; gamma8 blocked on the quantized mm path (see bf16-activation ticket).

## Progress (2026-07-10 late, commit ca4715c)

Landed: --drafter-quantize (q8_0/q4k/q6k post-hoc head quantization; tau-IDENTICAL at q8 on math — the risk didn't materialize), width>=1 floor, and the measured operating point gamma=4 + threshold 0.3: math ratio 0.51 -> 0.62 (91.4 tok/s, draft 5.3 ms). SURPRISE FINDING: q8 heads are time-NEUTRAL today (draft 4.7 vs 5.3 bf16) because the F32-cast + per-row quantized-mv taxes eat the byte win; gamma=8+q8 draft ballooned to 21 ms — a live measurement of the fork's per-row mv dispatch loop (fwd_mv re-dispatching per row). q8 flips clearly positive when bf16-activation-quantized-matmul-metal lands (now measurably the gating item for BOTH quant lanes). Remaining scope: Markov row-norm pruning (stretch, ~-1 ms), and re-A/B q8 after the fork work.

## Goal

Propose costs 8.7 ms/round — more than a full 24-layer greedy token (6.9 ms) for a 2-layer drafter — almost certainly dominated by the 248k-vocab reads: lm_head [248094,1024] bf16 (508 MB) once per proposal plus markov_w2 [248094,256] (127 MB) per Markov step. At the chain cap of ~2.27 tokens/round, every ms off propose is ~13 tok/s of spec throughput. None of these levers touch output correctness: drafts are verified, so a marginally worse draft distribution costs only tau.

## Cost model (agent-verified, 2026-07-10 evening)

markov_w1 stays BF16 (index_select gather, 512 B/row — same rule as the target embedding); confidence head [1,1280] stays dense. Implementation shape: switch lm_head (dspark.rs:261) and markov_w2 (263-266) slots to MixedLinear with a load-time `quantize_heads: Option<GgmlDType>` running quantize_onto on the mmapped checkpoint tensors — no artifact/manifest, no retraining. Risk is tau, not correctness (a flipped argmax at dspark.rs:430 cascades through prev_id and the confidence feature at 432): measure tau before/after on the fixed prompts; per-head q8_0/q6k fallback; break-even roughly delta-tau <= 0.25 per ms saved at the current round shape.

## Levers (shape decided by remeasure-spec-round-cost-model's breakdown)

- Quantize the frozen drafter's lm_head and markov_w2 post-hoc (q4k/q8, no retraining); relates bf16-activation-quantized-matmul-metal (else the F32 cast tax eats the win).
- Narrowed-vocab / top-k draft sampling (correct only the top-K of base logits before the Markov bias; fidelity caveat documented).
- Batch the per-step Markov lm_head applications where the chain allows.
- Remaining draft-side dispatch cleanup after the KV-cache/SDPA/packed-readback pass (commit 5218684).

## L1 specifics (spec-loop analysis, measured round math from target/dspark-dopt-math.json)

markov_w2 dominates: 127 MB PER Markov step = ~1.02 GB/round at gamma=8 (~3.4-4.4 ms) — bigger than the 508 MB lm_head gemm. q8_0 tier recommended (quant bench: Q8_0 decode-MV 1.78x; near-lossless for an additive low-rank bias and an argmax-only head). Combine with gamma default 4 in the runner (backbone still runs the full block of 8 — training distribution — but Markov/argmax/confidence run per-gamma; accepted>=4 in only 5/77 rounds, tau cost ~0.04): draft 8.75 -> ~4.3-4.7 ms. Stretch: exact Markov pruning via |bias_i| <= ||w2_i||*||w1[prev]|| row-norm bound -> gathered gemv over a provably-safe candidate set (~3.5 ms draft). Also take the L3 policy knobs here (one-liners, measured together): --confidence-threshold 0.3 (EV positive down to p~0.12 with rollback this cheap) and width = max(1, ...) (22/77 rounds pay full draft for 1 token at width 0).

## Acceptance

- Propose <= ~5 ms/round at gamma=8-width proposals, drafter parity harness still within thresholds (or updated thresholds with justification), tau on the fixed prompt suite within noise of today's 2.1.
- Spec wall-rate A/B per measurement protocol; results into the cost-model artifact.
