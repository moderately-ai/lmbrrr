---
id: eval-wave-2026-07-rollup
title: "EVAL WAVE 2026-07: rollup + execution order (start here)"
status: todo
priority: p1
dependencies: []
related: []
scopes: [evals]
shared_scopes: []
paths: []
tags: [eval-wave]
---
Umbrella for the post-fanout evaluation wave (2026-07-13). Every child ticket is written executor-grade: preconditions, exact commands, gates with pass criteria, decision rules, and where to record. READ FIRST: eval-protocol-ambient-control (the measurement discipline every other ticket depends on), then docs/research/metal-decode-utilization.md section 7 (standing instruments) and docs/research/1000-toks-campaign.md (standings + falsification ledger — do NOT re-run dead experiments; the ledger includes: q4_K schedule/layout/format micro-opts, SoA, nsg threadgroup geometry, magic-number dequant (analytical), certified sub-vocab head, fixed 32k target head (quality-rejected), scoped barriers, CPB=1000, eager enqueue, one-round-lag as default, tree bands under truthful STS).

EXECUTION ORDER (dependency + value):
1. eval-protocol-ambient-control (p1, infra — everything else uses it)
2. q4k-mv-round3-production-arbitrated (p1 — N_DST=2 / rt2 / combined, production-arbitrated; procedure in its comments)
3. greedy-host-path-deferred-readback (p1 — procedure in its comments)
4. m3-macos26-eval-suite (p1, BLOCKED on machine access — unblock the moment credentials arrive; highest ceiling)
5. gemv-width-splitk-concurrency (p2 — mc column-parallelism eval, procedure in comments; re-opens tree/recycling economics -> re-run their break-evens after)
6. fused-gemv-argmax-head (p2 — exact, small, zero quality risk)
7. megakernel-stage1-drain-probe (p2 — sizes the comb prize; promotes or kills starvation-free fusion)
8. dspark-ondevice-chunk-assembly adaptive-sync EV sim (p2 — offline first; procedure in its comments)
9. CPB=100 re-anchor (fold into any quiet session: env A/B CANDLE_METAL_COMPUTE_PER_BUFFER=50 vs default on the current binary, palindrome 6 rounds, medium+long; expected +3-4%, flag if it fails to reproduce)
10. metal-gpu-timestamp-meter (p3 — build when convenient; several evals get sharper with it)

STANDING RULES FOR ALL EVALS: (a) every kernel change ships bit-identical or margin-gated per the protocol — nextest 59/59 + stub oracle (dspark-run without --drafter, corrupt 0/3/5 invariant, 0.75 bound) + tree-check + text-compare; (b) after ANY shipped perf change: in-loop cost refit (run_spec_suite.py --split calibration --per-class 3 --reps 1 --gamma 6) and convergence check; (c) after standings-relevant changes: validation refresh (--split validation --per-class 3 --reps 3) + campaign-log row; (d) record every verdict (positive OR negative) as a ticket comment with receipts — negatives go to the falsification ledger.
