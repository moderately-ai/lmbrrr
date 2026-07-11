---
id: spike-training-scale-up-strategy-for-dspark-rounds-2
title: "Spike: training scale-up strategy for DSpark rounds 2+"
status: done
priority: p2
dependencies: []
related: [scale-dspark-training-corpus-modal, train-dspark-semi-autoregressive-drafter]
scopes: [docs/research]
shared_scopes: []
paths: []
tags: [research, dspark, training]
---
## Goal

Plan rounds 2+ properly before spending Modal budget: a written strategy (docs/research/dspark-corpus-scaling.md) that turns the round-0 -> round-1 data point (491 convs/tau 1.02 -> 20k convs/tau 2.25 math, 1.55 tides) into a deliberate scaling program instead of "train bigger and hope".

## Questions to answer

- **Scaling curve shape**: tau vs corpus size and tau vs epoch (per-epoch checkpoints now persist — evaluate tau on 2-3 checkpoints from round 1's own epochs retroactively? Only step_380 was saved for round 1; round 2 gets full curves). Where does 20k -> 100k -> 500k plausibly land against the break-even bar and the paper's tau 5-6?
- **Corpus composition**: PerfectBlend is ~1.4M conversations — what domain mix (math/code vs chat) matches our eval targets? DSpark paper's tau is domain-dependent (math/code highest). Should the corpus be regenerated greedily or with the current temp-0.7 sampling (drafter learns the target's sampled distribution vs its greedy mode — we verify greedily)?
- **Warm start vs fresh**: draft_init_checkpoint plumbing is committed (DeepSpec db36013, inert). Ablate on a cheap pair (e.g. 40k fresh vs 20k->40k warm) before betting the big run on it.
- **Training budget allocation**: at fixed GPU-hours, more epochs on less data vs fewer epochs on more data (round-1 loss was still descending at epoch 10 — under-trained on its own corpus?). LR schedule for warm-started runs.
- **Config ablations worth buying**: num_draft_layers (paper Fig. 3: 2 > 5-layer DFlash, but check at higher data), block_size (train 16 for tree-spec headroom later?), capture-layer set, num_anchors/max_length coverage of long contexts.
- **Data-gen throughput/cost**: 20k took ~50 min on 8 H100 shards (regen) + 10 min cache; extrapolate 100k/500k, decide shard counts, and price the rounds in Modal credits. Regeneration is embarrassingly parallel — the only real cost is $ not wall-clock if shard count scales.
- **Eval protocol**: fixed held-out prompt suite (domains x lengths) + the evaluator ticket (add-minicpm-dspark-evaluator) so tau comparisons across rounds are stable; STS calibration data sourced from the eval split, never train.

## Acceptance

- docs/research/dspark-corpus-scaling.md with: the scaling-curve analysis, a concrete round-2 spec (corpus size/mix, warm-start decision, epochs, ablation list with GPU-hour prices), and the eval protocol.
- Round-2 launch decision explicitly gated on this doc (user sign-off).
