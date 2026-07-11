# Draft-free speculation: prompt-lookup + token recycling — investigation and design

2026-07-11. Two zero-training draft sources landed behind flags (`--pld` commit afc13dc, `--recycle` commit 7e699e2), both verified by the standard exact-argmax chunk path (greedy floor preserved by construction). This doc records the measured findings and the design reasoning, because both differ from the reference methods in ways our cost model forced.

## Prompt-lookup: measured findings

Method: match the trailing n-gram (3–4, fallback scan) of the committed sequence against prompt+history; propose the tokens that followed the previous occurrence; verify as a normal chunk. Rotated same-session A/B, 3 reps, SD 0.2–3.1 tok/s (quiet machine).

**Finding 1 — the first measurement was noise.** A single cold run showed −4% on summary; replication flipped it to +5.3%. Recorded as a protocol reminder: nothing inside the ±10% floor from one run means anything.

**Finding 2 — short spans win everywhere; default span is 2.** Pre-registered prediction that copy-heavy text would favour wide spans was **falsified**: on a re-quote-the-function code prompt, accepted tokens were identical (15) at span 2, 4, and 8 — only the rejected tail grew, and span 8 cost −15%. The model re-quotes in short bursts. Cost model agrees: break-even acceptance is 0.99 tokens at span 2 but 2.41 at span 8.

**Finding 3 — gating to scheduler-skip rounds is mandatory.** Ungated PLD preempts strong drafter rounds: math (drafter τ 2.05) went −13% ungated, **+3.5% gated**; summary (drafter τ 1.10, gate effectively open) held +5.3% either way. PLD benefit anti-correlates with drafter strength, so it fires only where the scheduler's skip-hysteresis says drafting doesn't pay. Corollary for round-2+: a stronger drafter closes the gate more often and PLD's contribution compresses — that is correct behaviour, not regression.

**Bonus finding — copies are more greedy-exact than the drafter.** PLD-math matched the true greedy prefix 128/128 vs the scheduled drafter's 72 (quantized-drafter-head tie flips). Draft-free rounds avoid an entire noise source.

| class | control | gated PLD |
| --- | --- | --- |
| summary | 123.4±0.8 | **129.9±0.2 (+5.3%)** |
| math | 148.3±3.1 | **153.6±2.5 (+3.5%)** |

(code + tides pending quiet machine; both predicted ≈neutral-to-positive under the gate.)

## Token recycling: design (measurement pending)

Reference (arXiv 2408.08696) banks top-k candidates from every verify pass and drafts an ~80-node **tree** through the table for ~2×. Two deliberate departures, both forced by our constraints:

1. **Chains, not trees.** Wide tree verification is structurally expensive on the linear-attention target (per-branch state seeding; the two-branch tree measured break-even at best). Short chains ride the existing ≤12 rollback for free. Consequence: expect a modest, class-dependent gain, not the paper's 2×.
2. **Hard margin gating is the viability condition.** At measured verify costs (l=2 ≈ 1.77× a decode step), a depth-1 draft needs **~77% acceptance** to beat plain greedy. A Markov-1 table only clears that on near-deterministic continuations — exactly the rows where the banked top-1/top-2 logit margin is large. Chains extend only while the margin holds (default 6.0 logits, `--recycle-margin`); the expected firing rate is a minority of skip rounds, by design. The naive fire-on-any-entry variant provably loses on the same math that burned ungated PLD.

**Exact harvest at ~zero cost.** Naive top-k means shipping l×248k logits to host. Instead: reduce each row to 970 chunk-maxima on device (12 KB readback), then — since the k-th largest global value is exceeded by at most k−1 others, at most k chunks can have a maximum ≥ it — gather only the top-k chunk-maxima chunks (one batched index_select, 8 KB/row) and finish exactly on host. Tie semantics equal candle argmax at both stages (unit parity test vs naive full top-k). The harvest also returns the row argmax, replacing `argmax_tokens` on recycle rounds; the table warms from every round type.

Mux order per round: PLD match (contextual verbatim copy) → recycled chain (statistical, margin-gated) → trained drafter (scheduler EV) → greedy. All sources share one verify body and one gate.

## Full matrix + WHY investigation (2026-07-11, quiet machine)

First full matrix (SD 0.3–1.8) showed PLD +3.9% summary / +1.0% math / −2.7% code / **−12.8% tides**, and the recycle arm dragging 1–7 points everywhere — including classes where it never accepted a token. Three hypotheses were isolated and tested:

**H-A confirmed — tides −12.8% was scheduler dynamics on a diverged trajectory, not mechanism cost.** Only 3 PLD rounds fired (~1% direct cost). A PLD verify chunk's batched-matmul numerics flipped a near-tie at ~token 26 onto a different (equally greedy-valid) text; on it the scheduler chose width>0 three times as often (18 vs 6 rounds, 12/18 fully rejected — the STS overconfidence signature), and every width>0 decision reset the skip-hysteresis, buying ≥3 more drafter probes. Drafter invocations doubled (27→51); the extra draft time (+105 ms) was the entire wall delta. Corollary: the three arms produced three different texts at 160/139.6/143.5 tok/s — **on weak-drafter classes, trajectory luck swings throughput ±13%, dwarfing any mechanism effect**. Arm deltas there conflate the two; round accounting, not tok/s, attributes mechanism cost.

**H-B confirmed — the recycle drag was a per-round harvest tax.** With recycling armed but proposals impossible (margin 999999, zero rounds fired), throughput still fell −2.1% (code) / −3.5% (summary): the two-stage top-k's second device round-trip ran on every verify round.

**H-C negative — no margin threshold rescues Markov-1 recycling at current verify costs.** Acceptance at margin 6/9/12 on math: 45%/57%/50% — non-monotone, trajectory-dominated, far below the 77% depth-1 break-even.

### Fixes landed (both confirmed by an after-matrix)

1. **Harvest gated on `copy_gate_open`** — rounds where the scheduler considers drafting profitable skip the harvest. Code both-arm recovered +7.0%; with the gate closed the table stays cold and recycling correctly goes quiescent.
2. **Evidence-based hysteresis reset** — `consecutive_zero_widths` resets only on actual draft-token acceptance; a fully-rejected width>0 round counts as a realized zero. Tides pld recovered +9.3% (139.6→152.6), and the **default path improved too**: ctl +4.7% summary, +1.0% tides, +0.9% math, −0.6% code — weak-drafter classes stop paying the reset→3-probe cycle for overconfident width decisions.

### Verdicts (all provisional on the verify-intercept fix)

Every economic verdict here is priced against a verify chunk that costs 1.77× a decode step at l=2 where the roofline says ~1.05× (small-l quantized matmul doesn't batch-scale — see gemv-width-splitk-concurrency). At roofline cost the depth-1 break-even drops from ~77% to ~50% acceptance, which recycling's measured 43–57% *would* clear, and PLD/tree economics improve likewise. Standing results at current costs: gated PLD is +3.7% summary / ≈0 math, code / −5.6% tides (trajectory-dominated); recycling is a small net drag and stays default-off. **Re-run this matrix after the kernel fix and cost-model re-fit before treating any of these as final** (user directive: fix known performance issues before judging techniques).
