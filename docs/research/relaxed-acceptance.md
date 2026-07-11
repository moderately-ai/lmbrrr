# Relaxed Typical Acceptance

Date: 2026-07-10

Ticket: `relaxed-typical-acceptance-mode`

## What landed

`dspark-run --accept-margin <m>`: a draft token survives verification while its target logit is within `m` of the target's top logit (Medusa-style typical acceptance); committed tokens are the drafts, so outputs legitimately diverge from greedy. Exact argmax remains the default. Composes with `--confidence-threshold`; the JSON notes that confidence_records under the relaxed rule need their own STS fit before mixing with exact-rule calibration data.

## Measured (round-1 drafter, τ≈2, threshold 0.4, γ=8)

| margin | math τ | math ratio | tides τ | tides ratio | greedy prefix match |
| --- | ---: | ---: | ---: | ---: | --- |
| exact | 2.13 | 0.86 | 1.38 | 0.68 | 160/160, 50/96 |
| 1.0 | 2.06 | 0.83 | 1.40 | 0.68 | 83/160, 50/96 |
| 2.0 | 2.12 | 0.89 | 1.38 | 0.68 | 57/160, 50/96 |
| 4.0 | 2.27 | **0.95** | 1.49 | 0.72 | 27/160, 14/96 |

## Quality report

Outputs at m=2–4 stay coherent (arithmetic correct, explanations sound) but carry token-level artifacts: both math runs commit "seconds/promote" where greedy says "seconds/prompt" — a near-tie substitution that is plausible in logit space and wrong in context. The tides m=4 text reads clean.

## Reading

The plan's +20–50% τ estimate assumed rejections concentrate at near-ties. At τ≈2 this drafter's misses are mostly NOT near-ties (when it's wrong, it's wrong by a lot), so relaxation buys only +3–9% wall rate at visible-artifact margins. Guidance: keep exact acceptance the default at current drafter quality; revisit margins 1–2 after each training round — as position acceptance climbs, the remaining rejections shift toward genuine ties and this multiplier grows into its estimate. The margin should eventually be recalibrated against the measured logit-noise scale (chunk-split noise is ~0.375, so margins ≤0.5 are noise-dominated, not semantic).
