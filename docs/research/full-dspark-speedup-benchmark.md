# Full DSpark speedup benchmark — capstone report

Measured 2026-07-11 on the reference host (M4 Max binned, 32 GPU cores, Metal). All configs share the same prompt matrix, max-new-tokens 128, BF16 activations, one warm iteration discarded implicitly by taking medians over 2 iterations (per-iteration spread was <1.5% everywhere — see raw artifacts `target/cap-*-{1,2}.{json,stderr}`).

## Configurations

| id | description |
| --- | --- |
| greedy-bf16 | `lmbrrr run`, fused decode path, BF16 everything |
| greedy-q4k | greedy + `--quantize-lm-head q4k` + q4k-mlp-q8 text body manifest |
| dspark-fixed | `dspark-run --gamma 4 --confidence-threshold 0.3 --drafter-quantize q8-0`, round-1 drafter (step_380) |
| dspark-sched | dspark + `--schedule` (Appendix-A prefix scheduler, STS calibration, measured cost model, greedy-fallback hysteresis) |
| dspark-sched-q4k | dspark-sched + quantized target + `--cost-model spec-round-cost-model-q4kfull.json` |

Prompt classes: math word problem, expository (tides), code (merge sorted lists), summarization (Romeo & Juliet).

## Results (median tok/s over 2 iterations)

| config | math | tides | code | summary | mean |
| --- | --- | --- | --- | --- | --- |
| greedy-bf16 | 147.4 | 146.5 | 147.6 | 145.3 | **146.7** |
| greedy-q4k | 189.3 | 189.6 | 189.9 | 185.5 | **188.6** |
| dspark-fixed | 98.8 | 71.2 | 81.1 | 59.4 | 77.6 |
| dspark-sched | 116.5 | 120.7 | 122.2 | 118.7 | 119.5 |
| dspark-sched-q4k | 148.9 | 154.5 | 156.3 | 143.8 | 150.9 |

Per-class committed tokens/round (fixed γ4): math 2.13, code 1.68, tides 1.45, summary 1.25. Scheduler skip rates: BF16 17–79% of rounds drafted nothing; quantized 83–86%. Exact-greedy reconstruction: math 128/128; other classes diverge at kernel-noise ties only (advisory, both margins under LOGIT_NOISE_BOUND — same behaviour as the state-integrity oracle baseline).

## Verdict against the gate

**DSpark does not beat the fused greedy baseline on this hardware/model pair with the round-1 drafter.** Best spec config (sched-q4k, 150.9 mean) sits 20% below greedy-q4k (188.6); best BF16 spec (119.5) sits 19% below greedy-bf16 (146.7). The gate ("beat 136–146 outright") fails.

The scheduler itself is working exactly as designed: it detects that drafting doesn't pay and suppresses it (its win over fixed-length is +54% on the worst class, and it lifts every class), converging toward greedy-with-overhead. The residual 20% gap on the quantized arm is precisely attributed: on math, 127 rounds with 109 skips leaves 18 hysteresis probe rounds costing 0.207 s of draft+verify for zero accepted tokens — that alone accounts for the 0.852 s vs 0.677 s pure-greedy wall difference.

## Why: the break-even arithmetic

From the measured cost models (`spec-round-cost-model{,-q4kfull}.json`), a spec round at width w costs `draft_ms + verify_ms[w+1]` and must commit `round_ms × greedy_rate` tokens to break even:

| target | width 2 | width 4 | width 8 |
| --- | --- | --- | --- |
| BF16 (bar 146.7) | need 2.84 / max 3 | need 2.93 / max 5 | need 3.33 / max 9 |
| q4k (bar 188.6) | **need 3.95 / max 3 — impossible** | need 4.07 / max 5 | need 4.48 / max 9 |

Two multiplicative causes, ranked:

1. **The l≥2 verify cliff (dominant, structural).** Verify at chunk length 2 costs 14.1 ms BF16 / 15.7 ms quantized vs 6.5 / 5.0 ms for a single decode step — the small/mid-m matmul problem (quantized matmul has no batched path above m=1; dense tile GEMM is ~2× slow at m=2–9). If verify at l=5 cost ~1.25× a decode step instead of ~2.2×, the BF16 width-4 break-even would drop from 2.93 to ~1.94 tokens/round — which the round-1 drafter **already delivers on math (2.13)**. Speculation viability on this model is gated on `bf16-activation-quantized-matmul-metal`, not primarily on drafter quality.
2. **Round-1 drafter τ (secondary).** 1.25–2.13 tokens/round across classes. Even with a perfect verify path, tides/summary (1.25–1.45) stay under break-even; round-2 training (spike-training-scale-up, gated on sign-off) and tree speculation target this factor.

The historical framing matters: speculation was worth 17→87 tok/s against the pre-fusion 66 tok/s baseline. Fusion and quantization then raised greedy 2.9× (66→188.6), moving the goalposts faster than the unchanged drafter could pay back. That is the expected dynamic from the DSpark paper's own model — speedup is `(tokens/round) / (round_cost / decode_cost)` — and both terms degraded against speculation here.

## DSpark vs recurrent EAGLE

Not a fair fight and reported as such: the EAGLE lane was de-risked as a baseline only (single-round probe harness, overfit smoke drafter, never given a real training round). On the capstone prompts the EAGLE smoke drafter accepts **0/4 drafted tokens** (math and tides probes, `target/cap-eagle-*.json`) versus DSpark round-1's 1.25–2.13 committed tokens/round end-to-end. DSpark wins the drafter comparison trivially, but the honest statement is that *neither* speculation lane beats fused greedy on this pair today.

## Failure modes documented (per acceptance)

- **Probe overhead when drafting never pays:** hysteresis probes every 8th round cost ~20% on the quantized arm. Mitigation for a future pass: exponential probe backoff, or letting the scheduler consult the STS positional prior to disable probing entirely for a prompt class.
- **Verify cliff at l≥2:** see break-even table; the binding constraint on the quantized target (width 2 mathematically cannot break even).
- **Class-dependent τ:** math 2.13 vs summary 1.25 — draft acceptance collapses on open-ended prose; consistent with the paper's domain spread.
- **Kernel-noise tie flips:** greedy reconstruction is exact on math and diverges only at sub-noise-bound ties elsewhere; benign under the state-integrity oracle's criteria.

## What single-stream-1000 needs from here

The multiplier decomposition stands: quantized forwards ~190/s × τ_eff ≥ 5 ≈ 1000. In dependency order: (1) small-m matmul package — flattens the verify cliff AND unlocks quantized aggregate scaling past 845; (2) round-2 drafter training (awaiting sign-off) — lifts τ on structured classes toward the paper's 3–5; (3) tree speculation — converts per-path τ into τ_eff on a verify path that the small-m fix has made cheap. Aggregate-1000 is already achieved (1530 tok/s BF16 N=32, `target/mb3-bf16.json`).
