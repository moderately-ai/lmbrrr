# DSpark Corpus Scaling — Measured Rounds

Date: 2026-07-10 (round 1)

Tickets: `scale-dspark-training-corpus-modal` (execution), `spike-training-scale-up-strategy-for-dspark-rounds-2` (strategy — this doc is its evidence base)

## Rounds so far

| Round | Corpus | Steps (epochs) | Final loss | τ math | τ tides | Pos-0 accept | Spec/greedy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (smoke) | 491 convs | 24 (4) | ~2.5 | 1.02 | — | 0.8% | 0.27× |
| 1 | 20,899 convs (19,726 cached) | 380 (10) | ~1.8, still descending | **2.25** | 1.55 | 63% / 42% | 0.49× @γ8, **0.59× @γ3** |

Round-1 details (γ=8, math): accepted histogram [26,25,5,7,5,1,2,0,0], position acceptance [0.63, 0.28, 0.21, 0.11, 0.04, 0.03, 0, 0]; advisory 160/160 exact-greedy agreement. Training: H100:4, global batch 512, ~11 s/step after compile, ~80 min wall, torch.compile on. Pipeline wall-clock: ~50 min regen on 8 H100 shards (one shard preempted and restarted, roughly doubling its time), ~10 min cache build, ~80 min train.

## γ sweep with the round-1 drafter (quiet machine, baselines ~78 tok/s)

| γ | τ math | ratio math | τ tides | ratio tides |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1.93 | 0.58 | 1.50 | 0.42 |
| 3 | 2.13 | **0.59** | 1.54 | 0.41 |
| 4 | 2.22 | 0.57 | 1.54 | 0.39 |
| 6 | 2.35 | 0.56 | 1.54 | 0.39 |
| 8 | 2.25 | 0.51 | 1.55 | 0.39 |

τ saturates by γ≈3 (position acceptance decays fast), so smaller γ wins on wall rate today. Round shape at γ=3: draft 5.4 ms + verify 24.9 ms + re-advance 14.8 ms ≈ 45 ms for ~2.13 tokens ≈ 21 ms/token. The re-advance term is the target of `emit-per-position-verify-states` (projected ≈ 0.9× greedy at unchanged τ); the confidence head + scheduler then replace static γ.

## Scaling read (one data point is not a curve, but)

491 → 20.9k conversations (42×) moved τ 1.02 → 2.25 on math. The paper's τ 5–6 on 4B targets used corpora orders of magnitude larger; PerfectBlend totals ~1.4M conversations, of which we used ~1.5%. Round-1 loss was still descending at epoch 10 — the run is data- AND epoch-underfed. Per-epoch checkpoints (DeepSpec `checkpointing_steps="epoch"`, landed) plus weights-only warm start (`draft_init_checkpoint`, landed, inert) are in place for round 2; the spike ticket owns the round-2 spec (corpus size/mix, warm vs fresh ablation, epoch budget, eval protocol) and the launch is gated on user sign-off.

## Non-training τ multipliers queued (cheaper than data)

1. `emit-per-position-verify-states` — kill the per-round re-advance (in progress).
2. `calibrate-dspark-confidence-head` + `implement-dspark-hardware-aware-prefix-scheduler` — adaptive proposal length from the trained confidence head.
3. `relaxed-typical-acceptance-mode` — accept within the target's typical set (+20–50% τ per plan; quality bar allows it, ships behind a flag with a quality report).
4. `tree-speculation-over-dspark` — after the above land.
