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

---

# Rounds 2+ strategy (spike deliverable, 2026-07-11)

Everything below is the round-2 plan; **nothing launches without user sign-off on this section.** All four non-training multipliers above have since landed and been measured (rollback <1 ms; scheduler + calibration live; relaxed acceptance behind a flag; tree mechanism verified, EV pending in-kernel restart) — the spec lane's remaining deficit is τ itself.

## Where speculation stands after this session's kernel work

Greedy is 196.9 tok/s on the spec-lane manifest and **223.8 on `q4k-full-text`** (new record); the multi-column kernels cut verify chunk costs ~2× (l=2: 15.7 → 7.5–8.5 ms). Width-4 break-even is now **3.11 tokens/round** against the 196.9 bar (the capstone had measured 4.07, with width 2 mathematically impossible). Round-1 delivers 2.13 math / 1.25–1.45 prose. The verify-cost half of the gap is solved; τ is what's left.

Two fresh findings that reshape round 2:

1. **Drafter–target quantization mismatch is large.** τ_math drops 2.13 → **1.69** when the target moves to the deeper-quantized `q4k-full-text` manifest. The deployment target is quantized; round-1 traces were BF16. Round-2 traces must come from the deployment-config target (quantized weights, greedy decoding — we verify exact argmax, so temp-0.7 sampling trains mass off the mode the verifier checks). SGLang data-gen can't replicate Metal kernels exactly; the right approximation is GGML-equivalent quantized weights in the generator, which captures the weight-noise distribution.
2. **STS is overconfident in-loop** (scheduler predicted >3.5 expected tokens/round on math; realized 2.13). Refit cumulative survival per round from held-out eval traces of the deployed configuration, never train.

## Scaling estimate (2-point fit, wide error bars — measuring the slope IS round 2's job)

491 → 20.9k (~1.6 decades) moved τ_math 1.02 → 2.25: **≈ +0.75/decade** (prose ≈ +0.35/decade).

| corpus | τ_math est. | τ_prose est. | vs break-even 3.11 |
| --- | --- | --- | --- |
| 20k (measured) | 2.25 | 1.55 | below |
| 100k | ~2.7 ± 0.3 | ~1.8 | below, closing |
| 500k | ~3.2 ± 0.5 | ~2.1 | math at/above |
| paper regime (M-scale + tree + relaxed) | 5–6 | — | well above |

100k alone likely does not flip the spec lane; it measures the slope with full per-epoch τ curves and validates the quantized-trace fix (which may be worth several tenths of τ at deployment by itself). 500k is where chain-math plausibly clears break-even; tree and relaxed acceptance multiply from there.

## Round-2 spec (proposed)

- **Corpus 100k** from Open-PerfectBlend: 50% math+code (τ-elastic, matches the paper's domain finding), 30% general instruction, 20% expository/summarization (our weakest classes).
- **Traces from the deployment-config target** (GGML-equivalent quantized weights in the generator, greedy).
- **Warm start ablated before betting**: 20k→+20k warm (`draft_init_checkpoint`, committed inert) vs 40k fresh, identical eval; winner carries the 100k run.
- **15–20 epochs** (round-1 loss still descending at 10 — epoch-underfed), cosine decay, warm starts at 0.3× peak LR; select the checkpoint by held-out τ per epoch, not loss.
- **Ablations bought**: `block_size 16` (tree-width headroom, marginal cost) and `num_draft_layers 2 vs 4` (paper favours 2 at small data; re-check at 100k). Not bought: capture-set changes (invalidates warm start), long-context knobs (round 3).
- **Eval protocol** (prerequisite; `add-minicpm-dspark-evaluator`): fixed held-out suite, 4 domains × 3 lengths × 3 prompts = 36, reporting τ / position-acceptance / tokens-per-round per domain on the deployment target; STS fit from this split only.

## Budget (Modal, H100 ≈ $4/GPU-hr; round-1 reference: 20k regen ≈ 8 GPU-hr ≈ $32, train ≈ $6)

| item | GPU-hr | ≈ cost |
| --- | --- | --- |
| warm-start ablation pair + 20k quantized-target regen | ~14 | ~$56 |
| 100k regen (quantized target, 8–16 shards) | ~40 | ~$160 |
| 100k train, 15–20 epochs | ~8 | ~$32 |
| block_size-16 + draft-layers ablations | ~16 | ~$64 |
| **round-2 total** | **~78** | **~$310** |
| (contingent) 500k regen + train | ~220 | ~$880 |

Data-gen is embarrassingly parallel: shard count buys wall-clock, not dollars.

## Gates

- **Launch gate: user sign-off on this plan.** Nothing above runs until then.
- **Round-3 gate (data-driven):** if 100k shows ≥ +0.4 τ_math over round 1 on the deployment target, buy the 500k run. If not, data alone is not the path — the budget pivots to tree in-kernel restart + relaxed-typical acceptance, which multiply whatever τ exists.
- Every round evaluates on the fixed suite, controls first, per the measurement protocol.
