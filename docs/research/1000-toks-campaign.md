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
| + K1/K2/K5 kernel fuse (mc hoist, mm routing, width fusion, v2 decode) | 219.3 (r3 arm) | 185.5 | greedy floor 215 → **246-249** |
| **round-4 stack (400k drafter, tau 4.41, truthful STS refit) — SHIPPED** | **228.2 (+4.1%)** | divergent | qa +9.7% (tau 1.12→1.39); summ −3.5%≈noise |

Greedy floor (no speculation): **~247 tok/s** bench-mode after the K1/K2/K5 fuse (was ~215). τ is now abundant (4.41 gsm8k) relative to what the kernel economics can convert — math τ +28% bought +4.1% tok/s — so the binding constraint has flipped to chunk-verify cost and the mv floor (`q4k-soa-plane-repack`). Weak classes run near-greedy, correctly, per the scheduler's own economics.

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
- **q4_K matvec micro-optimizations**: three falsifications (row tile, load shape, half accumulate) pin the kernel as integer-unpack-pipe-bound (99 GB/s effective vs 262 q8_0 / 358 dense on the same shape). Three more this session (u32 unpack, dot(), float4 loads) closed the schedule class.
- **q4_K SoA plane-split repack (2026-07-12)**: the layout hypothesis is dead too. Planes (16B headers | 128B quants, perfect 128B cache-line alignment) vs 144B AoS: +3.7% lm_head, +1-2% projections, bitwise-identical outputs (fork 00adb831, `qmv-soa` gate task). With registers unconstrained (maxThreads 1024) and 62k TGs, the limiter is integer-dequant ISSUE RATE — no data-layout or schedule change fixes arithmetic. Absolute-time check: q4_K AoS is already the fastest head per token (142MB @ 1.16ms vs q8_0 1.28ms vs bf16 1.42ms), so the q8_0-head fallback is falsified a priori. **The floor moves by reading fewer weights, not by reading them faster.**
- **Q8 lm_head swap**: wins isolated (-0.42 ms) but loses -1 to -6% in-stream; also improves tau (head fidelity shifts near-ties) — relevant to future head work.

## Deployment convention (locked in 2026-07-12)

A drafter deploys as one directory: `model.safetensors + config.json + sts.json + draft_vocab.json + cost_model.json`. `dspark-run --drafter DIR --quantized-manifest M` reproduces the full stack with zero spec flags (gamma 6, schedule/pld/recycle default on; `--flag=false` ablates; explicit artifact flags override the bundle). Round-4 deploys by assembling its own bundle dir through the truthful flow: unscheduled gamma-6 records on the calibration split → `evals/fit_sts.py` → held-out validation vs the incumbent bundle.

## Open levers, quantified (updated 2026-07-12 evening — post K3/K6 falsifications)

1. ~~Round-4 drafter~~ SHIPPED (tau 4.41 gsm8k final; `target/dspark-drafter-round4/` is the default bundle).
2. ~~q4_K SoA repack~~ FALSIFIED (+3.7% vs the ≥1.5-2× gate; issue-rate-bound). GEMV-speed work on 4-bit formats is closed.
3. ~~Certified sub-vocab target head~~ **FALSIFIED — both bound families** (offline sim on 844 real decode hiddens, harness validated at 0.96 argmax agreement vs the production trace): cluster Cauchy-Schwarz f=0.996 scored; SVD-W128 + per-row tail norms f=0.994. Structural: decode queries sit 91% outside the table's top-128 subspace, the 248k tied table's spectrum is nearly flat (row tail-norm 0.77), the head runs at cosine ~1e-3 — every norm-product bound is 50-100× the logit gaps. Bit-exact head reduction on this model is dead, not tuning-limited.
4. ~~Eager skip-hysteresis~~ FALSIFIED (A/B wash; the drafted-vs-parked pace gap was selection bias — the scheduler's economics were already right). Constants now CLI-ablatable.
5. **Host encode path — the biggest surviving exact lever**: bench `decode_model` wall shows ~1.4ms/token of CPU spent encoding the forward (~27% of the realized greedy step). Fixes are candle-core encode-path surgery and further op-count reduction — invasive, unproven, plausibly −0.3..−0.7ms. Draft cost (3.47ms, dispatch/host-bound) is the same class of fix on the spec side.
6. **Beyond-400k corpus** (`dspark-cache-redesign-beyond-400k`): slope +0.50/3.33× still steep; buys τ, converts at the kernel exchange rate (+28% τ bought +4.1% tok/s on math this round).
7. **Quality-trading options (product decisions, not optimizations)**: fixed 32k target-head sub-vocab (−1.05ms ⇒ +20-25% floor, changes committed outputs; would ship behind a flag with a quality report, like `--accept-margin`); relaxed-typical acceptance (already built behind a flag).

**Honest standings after the falsification sweep**: the exact-decoding greedy floor is essentially at its structural limit (~247 bench / 5.09ms in-loop: GPU stack 1.4 + head 1.2 + argmax 0.25 + host ~1.4-2.2). With host-path work plus one more corpus round, the defensible exact-stack horizon is ~300-400 tok/s on structured domains — not 1000. Reaching materially past that requires trading exactness (lever 7), batching (a different product), or a different target model/hardware. The 1000 target should be renegotiated against that menu.
