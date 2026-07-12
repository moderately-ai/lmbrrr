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

## Round-2 results (2026-07-11)

Executed variant: vLLM regen (native MiniCPMV4_6 serving, 6,597 tok/s at concurrency 64; 40k conversations in 21:38) against the fakequant deployment-config target; NVMe-staged training at ~5 s/step. Ablation arms: A = warm start from round-1 (20k corpus, 0.3x LR, 18 epochs), B = fresh init (40k corpus, lr 6e-4, 18 epochs). Both evaluated by the Modal MiniCPM evaluator on the fakequant target (gsm8k:20, mt-bench:10, greedy).

| arm | gsm8k tau | mt-bench tau | pos-0 accept (gsm8k) |
| --- | --- | --- | --- |
| round-1 step_380 (rebased on fakequant target) | 2.08 | 1.56 | — |
| A: warm, 20k, 18 ep (plateaued by ep 12) | 2.69 | 1.71 | — |
| **B: fresh, 40k, 18 ep** | **3.54** | **2.08** | **78.2%** |

**Verdict: fresh wins decisively.** Warm-starting saved compute but capped the ceiling (its per-epoch tau plateaued by epoch 12 while B kept improving); at 2x corpus the fresh run beats warm by +32% gsm8k tau and round-1 by +70%. Local validation on the real Metal runner corroborates the direction (warm arm measured tau 2.72 fixed-gamma on math locally vs the evaluator's 2.69 on gsm8k — the fakequant trace pipeline transfers almost exactly; B's local numbers pending checkpoint download).

**Round-3 gate: cleared 3x over.** +1.46 tau_math over round-1 vs the +0.4 criterion. Config for round 3: FRESH init (not warm), 100k-500k corpus, same deployment-config trace path, argmax confidence labels (already merged in the DeepSpec fork), per-epoch held-out tau selection.

Deployment sequencing before the default drafter flips to B: download step_1386, local fixed-gamma tau A/B vs round-1, argmax-event STS fit (the warm arm's fit measured +8.2% held-out over round-1 scheduled; rerun the same 10-minute procedure for B), scheduled validation must beat round-1 scheduled and B fixed-gamma, multi-class sweep.

## Round-3 results (2026-07-12): 120k fresh, 10 epochs, 8x H100 fused pipeline

Pipeline note: the volume-copy stage was removed mid-round after two failures (ENOSPC at ~600 GiB of stale caches, then a ~100 MB/s degrading single-stream copy of the 810 GiB cache). The fused stage (commit b0175b1) builds the cache on container-local NVMe and trains from it directly in one 8x H100 container; the volume receives only checkpoints. Wall clock: regen 67 min + prep 14 min + train 3.6 h (5.3 s/step, 1.94x the 4-GPU rate at fixed global batch).

Per-epoch held-out gsm8k tau (fakequant deployment target, greedy, n=20):

| epoch | tau | delta |
| --- | --- | --- |
| 1 | 2.251 | — |
| 2 | 2.841 | +0.590 |
| 3 | 3.310 | +0.469 |
| 4 | 3.608 | +0.298 |
| 5 | 3.634 | +0.026 |
| 6 | 3.741 | +0.107 |
| 7 | 3.864 | +0.123 |
| 8 | 3.909 | +0.045 |
| 9 | 3.812 | -0.097 |
| 10 | **3.914** | +0.102 |

Curve flattens at tau ~3.85-3.91 from epoch 7 (epochs 8/10 statistically tied at n=20). mt-bench final: **2.302** (pos-0 61.3%). Epoch-4 already exceeded the 40k arm's 18-epoch final — corpus size dominates epoch count decisively.

**Corpus scaling law (gsm8k tau, fresh init, deployment-config traces):** 40k -> 3.54 (18 ep); 120k -> 3.91 (10 ep, plateau by 7). Marginal: **+0.37 per 3x corpus**, and cheaper per point than epochs. Round-1 gate comparison: +1.83 over round-1's 2.08 (gate was +0.4).

**500k recommendation:** the slope supports one more scale step. With the fused pipeline the cost estimate drops from ~$880 to ~$330-380 (regen 4.5 h single-GPU, prep in-container, ~6 epochs at 8x H100 with the per-epoch plateau stop; projected tau ~4.3). One constraint to resolve first: a 500k cache is ~3.3 TiB on NVMe — above the 1.5 TiB this round used; either confirm Modal's ephemeral-disk ceiling supports ~3.5 TiB or cap the corpus at ~350-400k (~2.4-2.6 TiB). Epoch budget: 6 with the plateau stop armed.

## Round-4 results (2026-07-12): 400k fresh, 6 epochs, plateau stop armed

Corpus capped at 400k per the NVMe ceiling analysis above. Per-epoch held-out gsm8k tau (fakequant deployment target, greedy, n=20): 3.361 → 3.94 → 4.20 → 4.18 (dip, strike one) → 4.37 (reset) → **4.41 final** (step_4638). mt-bench final **2.58**. Confidence head well-calibrated: ece 0.033 gsm8k / 0.025 mt-bench, auc 0.859 / 0.899 — the truthful per-position Platt fit rides on clean raw signal.

**Corpus scaling law update (gsm8k tau, fresh init, deployment-config traces):** 40k → 3.54, 120k → 3.91 (+0.37 per 3×), 400k → 4.41 (**+0.50 per 3.33×**). The slope did not flatten at this step — it steepened slightly (the 6-epoch plateau-stop budget was also tighter than round-3's 10, and e6 was still +0.04, so 4.41 may modestly undershoot the checkpoint family's ceiling).

**Held-out validation vs the round-3 bundle** (local Metal runner, post-K1/K2/K5 fused stack, truthful STS refit from 958 records, rotated 3 reps): math 219.3 → 228.2 tok/s (+4.1%, tau 2.93 → 3.76), qa 160.3 → 175.8 (+9.7%, tau 1.12 → 1.39), translation flat (tau 1.00 both — hysteresis floor), summarization −3.5% (≈2σ, same content); coding/writing DIVERGENT CONTENT (excluded per protocol). **Shipped: `target/dspark-drafter-round4/` is the deployed bundle.** Note the conversion gap: tau gains (+28% math) convert to single-digit tok/s because chunk-verify costs and the greedy floor bound the exchange rate — the binding constraint is now kernel-side (q4_K SoA repack), not tau.

**Beyond-400k decision (slope rule):** slope healthy → the tau path is more corpus, NOT the MTP head. Full-scale (~1.4M PerfectBlend) implies a ~9 TiB cache — far past the NVMe ceiling; requires the cache redesign (sharded/streaming prep or capture compression) before any launch. Tracked on `dspark-cache-redesign-beyond-400k`.
