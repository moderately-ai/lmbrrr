---
id: eval-protocol-ambient-control
title: "EVAL INFRA: control-normalized benching protocol (fixes the +/-35% micro-bench drift)"
status: todo
priority: p1
dependencies: []
related: []
scopes: [evals]
shared_scopes: []
paths: []
tags: [eval-wave, infrastructure]
---
WHY: the isolated micro-bench (metal_benchmarks qmv/nsg-sweep) measured 121 GB/s then 77-79 GB/s on IDENTICAL code across sessions (2026-07-12 vs -13) at modest load — +/-35% ambient drift. Cross-session absolute rates from that harness are meaningless; it can only support WITHIN-RUN comparisons. Any claim under ~1.5x MUST be arbitrated in production (bit-identical kernel swap + text-identical gate + bench A/B).

PROTOCOL (mandatory for every future bench session, micro or production):
1. PRECONDITIONS: `uptime` load < 4.0; `ps aux -r | head -8` shows no cargo/rustc/agents/other compute; power adapter connected; note whether the display is on (GPU watchdog + WindowServer contention differ display-off); never run builds concurrently with benches; first runs after any kernel change are shader-compile transients (discard).
2. SESSION CONTROL: at session START and END run `cd ~/workspace/github.com/huggingface/candle/candle-metal-kernels && ./target/release/examples/metal_benchmarks qmv 2>&1 | head -2` and record the lm_head m=1 GB/s line. If END deviates from START by >10%, the session's micro numbers are VOID — rerun. Never compare micro GB/s across sessions; compare only ratios within one session.
3. PRODUCTION ARBITRATION (for kernel variants): all candidate kernels must be bit-identical per row (verified by the bench task's bitwise gate); then swap into production via env var (see q4k-mv-round3 ticket), run the rotated bench A/B (palindrome A B B A A B, warmup 1, iterations 3, --profile medium --profile long), decide on median steady_state_tokens_per_second with non-overlapping ranges.
4. RECORD: every bench session's control values + verdicts go in a ticket comment; standings-affecting results also go to docs/research/1000-toks-campaign.md.

DELIVERABLE for this ticket: add the START/END control to the bench scripts in scratchpad usage (a 5-line wrapper script committed to evals/bench_session.sh that prints uptime, runs the control, then execs the given command, then re-runs the control), and record one calibration session demonstrating the control catching drift. DONE-WHEN: evals/bench_session.sh exists, is used by the round-3 eval, and the protocol paragraph above is copied into docs/research/metal-decode-utilization.md section 7.
