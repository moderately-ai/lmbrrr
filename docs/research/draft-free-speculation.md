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

## Bench matrix queued (quiet machine)

{ctl, pld, pld+recycle} × {summary, tides, math, code} × 3 reps rotated; margin sweep {4, 6, 9} on the best class; then re-run against the round-2 drafter (staged at `target/dspark-drafter-round2-warm`, identity STS until the argmax refit) to measure the gate interaction. Negative results get recorded and the flag stays default-off.
