# adapt-margin suite gate (2026-07-19)

## Setup
- M3 Pro, Q2_0 target + Q8_0 drafter, planar mm2d, N=96
- 6 short class prompts × 2 reps (rotated arms)
- Arms: fixed m1 (`--no-adapt-margin`), global `--fast` (m3), hard adapt `1,2`, soft15 `0,1.5,1,3`, soft20 `0,2,1,3`

## Overall (mean of class medians)

| arm | mean tps | mean PPL | vs m1 tps | vs m3 PPL |
|---|---:|---:|---:|---:|
| m1 fixed | 18.84 | 1.106 | — | — |
| m3 `--fast` | 20.15 | 1.165 | +7.0% | — |
| hard `1,2` (can exact) | 16.56 | 1.184 | **−12.1%** | +1.6% |
| **soft15 `0,1.5,1,3` (DEFAULT)** | **19.83** | **1.131** | **+5.3%** | **−2.9%** |
| soft20 `0,2,1,3` | 18.84 | 1.110 | +0.0% | −4.8% |

## Gates for default-on
- tps ≥ m1 + 2%: **PASS** (+5.3%)
- no class PPL > m3 + 2%: **PASS** (empty worse set)
- hard adapt: **FAIL** (fact −33%, summarize −36% tps; summarize PPL +24% vs m3)

## soft15 per-class

| class | Δtps vs m1 | ΔPPL vs m3 | notes |
|---|---:|---:|---|
| math | +4.8% | 0% | mostly fast |
| code | +0.0% | −5.2% | wash speed, better PPL |
| prose | +18.6% | −2.8% | big win |
| fact | −3.9% | −8.4% | slightly slower, much better PPL than m3 |
| summarize | +4.8% | 0% | |
| translate | +9.1% | 0% | |

## Decision
Default `gguf spec` → soft adapt `0,1.5,1,3`.  
`--no-adapt-margin` for fixed m1; `--fast` / `--exact` unchanged.

Raw rows: `evals/adapt/soft_suite_rows.json`.

## v3b default re-baseline (prose N=128, 3 reps, post-default-flip)

| arm | median tps | accept |
|---|---:|---:|
| plain | 14.47 | — |
| exact | 15.35 | 2.342 |
| m1 `--no-adapt-margin` | 18.20 | 2.969 |
| **default soft** | **20.16** | 3.448 |
| `--fast` | 20.10 | 3.448 |

Default matches `--fast` speed on this prompt with soft schedule (0 exact / ~10 base / ~17 fast).
