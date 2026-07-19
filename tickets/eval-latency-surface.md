---
id: eval-latency-surface
title: "EVAL: latency surface — TTFT, prefill throughput, decode-vs-position curve, 1000-token sustained runs"
status: todo
priority: p2
dependencies: []
related: [program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [evals]
shared_scopes: []
paths: []
tags: [eval-wave]
---
WHY: every standings number is ONE point on the latency surface — steady-state decode tok/s at short prompts and <=long-profile generations, on a cold-ish chip. The campaign is literally named 1000-toks and we have no standing 1000-token measurement; we have no TTFT or prefill row (bench already computes prefill_tokens_per_second and decode_time_to_first_token_seconds — src/commands/run_bench.rs:279-282 — we just never sweep or record them); and the hybrid architecture means decode cost GROWS with position for the 6 full-attention layers (sdpa over KV) while 18 DeltaNet layers are O(1) — nobody has measured the slope.

PROCEDURE (measurement only, no code changes expected; per eval-protocol-ambient-control):
1. PROMPT SET: synthesize prompts of ~128 / 1k / 4k / 16k tokens (python: take a long public-domain text, tokenize with the model tokenizer, truncate to N, detokenize; record ACTUAL prompt_tokens from the bench report, not the target). Store under evals/prompts/latency-surface/.
2. SWEEP: for each prompt length run `./target/release/lmbrrr bench --quantized-manifest artifacts/minicpm-v46-q4k-full-text/manifest.json --quantize-lm-head q4k --warmup 1 --iterations 3 --profile medium` with the prompt override flag (check bench's prompt arg in src/main.rs; if bench can't take a prompt file, add --prompt-file — small change, gate with evals/run_gate_battery.sh). Record: prefill tok/s, TTFT, steady decode tok/s.
3. GENERATION-LENGTH AXIS: one run per prompt length with --max-new-tokens 1000 (or a 1000-token profile if profiles are hardcoded — src/commands/run_bench.rs:541 lists short/medium/long; adding a `xl` profile is in scope). Record tok/s over the run (first 100 vs last 100 tokens if the report exposes per-token timing; else split via two runs) — this captures BOTH position-dependent attention cost and thermal droop. Peak memory: prefix the command with `/usr/bin/time -l` and record maximum resident set size per prompt length.
4. SPEC ARM: repeat step 2 at 128 and 4k with --drafter artifacts/dspark-drafter-round4 — does speculation's advantage hold at depth (verify chunks touch attention KV too)?

DELIVERABLE: the surface table (prompt_len x {TTFT, prefill tok/s, decode tok/s early, decode tok/s late, peak RSS, spec tok/s}) as a comment + campaign-log section. ROUTING RULES: decode at 16k >= 25% slower than at 128 -> file an sdpa_vector lane ticket (2-pass tuning / KV layout) with the curve as receipt; TTFT dominated by shader compile on first run -> file an MTLBinaryArchive pipeline-cache ticket (kills first-run compile; product TTFT lever); thermal droop >10% within 1000 tokens -> quote SUSTAINED numbers in standings going forward, not burst (amend eval-protocol-ambient-control).
