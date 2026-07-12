# 1000 tok/s campaign log — measured composition and standings

Running log of the single-stream campaign's measured multipliers. All figures are same-session Spec-Bench held-out numbers on the deployment config (q4k-full-text + q4k head, fakequant-consistent drafter training) unless noted. Protocol: rotated arms, 3 reps, suite harness (`evals/run_spec_suite.py`).

## Standings (2026-07-12)

| config | math (q404) | coding (q124) | 6-class mean |
| --- | --- | --- | --- |
| round-1 drafter, pre-scheduler-fix stack | 144.4 | 131.1 | 130.3 |
| + rate-based skip-hysteresis (135409a) | 157.5 | 145.4 | 138.5 |
| round-2 stack (B drafter + balanced STS + FR-Spec 32k + refit cost model) | 182.8 | 168.2 | 150.7 |
| **round-3 stack (120k drafter, same recipe)** | **204.0** | **187.8** | **155.4** |
| **+ in-loop cost model + truthful gamma6 STS (2026-07-12)** | **208-231 (3q)** | **185.7** | **+1.2% vs same-day incumbent** |

Greedy floor (no speculation): ~215 tok/s device-resident. The speculative stack now beats greedy by ~2x on math/coding and the 6-class mean sits at 72% of the greedy floor because weak classes (translation, summarization) still run near-greedy — correctly, per the scheduler's own economics.

## Measured stage contributions this week

- **DeltaNet v2 chunk kernels** (fork 3781e79f): verify chunks -2.5% (l=2) to -9.1% (l=12); dspark math +4.4% end-to-end. Decode stays on v1 (~0.9 ms/token whole-layer kernel; the 3-dispatch v2 loses there). Falsified en route: the "decode kernel = 2.8 ms/token" diagnosis (host-profiler backpressure artifact).
- **Scheduler: rate-based skip-hysteresis**: +19% on the round-1 stack mean (130.3 -> 138.5 was the same-arm gate; qa/coding/math +9-11%). Two deficit-weighted refinements measured NEGATIVE (146.5 / 153.7 vs 155.4 champion) — the full-absolution reset is load-bearing.
- **Corpus scaling (the dominant lever)**: fresh-init tau on the fakequant target: round-1 2.08 -> 40k 3.54 -> 120k 3.914 (gsm8k; mt-bench 1.56 -> 2.08 -> 2.302). +0.37 tau per 3x corpus; epoch 4 at 120k beat 18 epochs at 40k. 400k in flight.
- **Argmax-event STS calibration**: uncalibrated round-2 lost to round-1 (154.3 vs 160.7); calibrated it won (+8.2% held-out). Calibration is a gating multiplier, not a nicety. Flow codified in `evals/fit_sts.py`.
- **FR-Spec 32k draft vocab** (assistant-ranked, control tokens pinned): tau exactly unchanged in all 6 classes, +11.6% mean, draft cost 5.37 -> 3.47 ms. The 8k profile locates the cliff (94.3% coverage -> -12% tau on math/coding).

- **Cost-model contract fix (2026-07-12)**: the scheduler's kernel-time table missed two structural effects, measured in-loop across 1,109 rounds (18 calibration questions): drafted rounds run 0.1-0.7 ms UNDER the synchronized table (queue overlap hides host work), while no-draft rounds run 0.7-1.1 ms OVER it (the bare per-round host cost). The composite understated greedy pace ~19% and biased admission narrow — the mechanism the decision-optimal (upward-biased) STS was accidentally compensating for. Shipped: per-l in-loop-refit verify table + explicit greedy_step_ms (v32k-inloop artifact) + truthful gamma-6 STS; validation A/B: +1.2% 6-class mean, math +2.6%, coding +4.2%, weak classes parity. The truthful-calibration flow is now deployable as-is for round-4.
- **Measurement confound found**: different width schedules invoke different chunk-length kernels; +/-ulp logit noise flips near-ties and diverges committed text between arms (e.g. qa q324: 81 vs 52 tokens) — tok/s deltas on such questions are content luck, not economics. The suite harness now prints DIVERGENT CONTENT on affected questions; averaging over per-class 3 questions bounds the residual luck.

## Recorded negatives with unlock conditions

- **Tree speculation**: tau-positive (coding 2.44 -> 3.10) but cost-bound (-20% tok/s); unlocks when verify chunks get ~2x cheaper.
- **Token recycling**: no margin threshold reaches viability at current verify costs (break-even ~77% acceptance at depth 1; drops to ~50% at roofline verify cost).
- **q4_K matvec micro-optimizations**: three falsifications (row tile, load shape, half accumulate) pin the kernel as integer-unpack-pipe-bound (99 GB/s effective vs 262 q8_0 / 358 dense on the same shape).
- **Q8 lm_head swap**: wins isolated (-0.42 ms) but loses -1 to -6% in-stream; also improves tau (head fidelity shifts near-ties) — relevant to future head work.

## Deployment convention (locked in 2026-07-12)

A drafter deploys as one directory: `model.safetensors + config.json + sts.json + draft_vocab.json + cost_model.json`. `dspark-run --drafter DIR --quantized-manifest M` reproduces the full stack with zero spec flags (gamma 6, schedule/pld/recycle default on; `--flag=false` ablates; explicit artifact flags override the bundle). Round-4 deploys by assembling its own bundle dir through the truthful flow: unscheduled gamma-6 records on the calibration split → `evals/fit_sts.py` → held-out validation vs the incumbent bundle.

## Open levers, quantified

1. **Round-4 drafter (400k, in flight)**: projected tau ~4.2-4.3; deploys through the codified STS + validation flow.
2. **simdgroup-matrix q4_K rewrite**: eliminate per-element integer unpack; bounded ~1 ms/token on the lm_head plus chunk-cost reductions that unlock tree (+tau) and recycling.
3. **Certified sub-vocab target head**: ~1.1-1.2 ms/token ceiling at 99.45% coverage with ~1/180 fallback rate; shares head-clustering analysis with (2).
4. **Beyond 400k corpus**: needs the >3 TiB cache redesign (sharded/streaming or capture compression).

Rough composition to 1000 single-stream on structured domains: greedy floor to ~280-300 f/s via (2)+(3), times tau_eff 3.5-4 via round-4 + unlocked tree = 950-1200. The bottleneck order is now kernels-then-tree, with corpus scaling still buying tau cheaply until it flattens.
