---
id: eval-harness-validity-fixes
title: "EVAL INFRA p1: fix the measurement harness — two HIGH-severity metric bugs + observability (all standings ride on this)"
status: todo
priority: p1
dependencies: []
related: []
scopes: [runtime/candle, evals]
shared_scopes: []
paths: []
tags: [eval-wave, infrastructure]
---
A full-file audit of run_bench.rs / generate.rs / run_spec_suite.py / dspark.rs found real defects in the metrics every decision rides on. Fix in this order; each fix changes metric values, so VERSION the metrics (add steady_state_tokens_per_second_v2 etc. alongside the old field for one transition period) and never compare v1-to-v2 across sessions.

HIGH 1 — steady-state window arithmetically inconsistent on the device-chain path: metric = (N-1)/(decode_elapsed - first_token_after_prefill), but on the greedy device chain first_token_after_prefill is set at the FIRST 8-TOKEN FLUSH (generate.rs ~202/229), i.e. the subtracted window contains ~7 forwards while the numerator subtracts 1 token -> ~+6% inflation at N=128, scaling with READBACK_EVERY/N. FIX: count tokens actually inside the measured window (subtract the flush size, not 1), or set first-token time at the first token's own flush position. Cross-arm ratios were fair only when both arms used identical N and READBACK_EVERY — note this in the campaign log.

HIGH 2 — EOS asymmetry: the device chain runs up to READBACK_EVERY-1 forwards past EOS; their GPU time lands in decode_elapsed but the tokens are not counted -> an arm whose (tie-flip) text hits EOS mid-batch is charged phantom token-times. FIX: (a) record an eos_overshoot_forwards counter per run, (b) subtract overshoot forwards' estimated time OR end the window at the last counted token's flush; (c) bench has NO content-divergence detection across arms — generated_token_ids is already in the JSONL, add an offline divergence check to evals (compare across arms, flag like the suite does).

MED — suite tokens_per_second is END-TO-END, not decode: wall clock starts before prefill, includes the prefill-produced token, AND includes two JSON file loads (StsCalibration + RoundCostModel, dspark.rs ~299-310) inside the timed window. FIX: move file loads out of the wall; emit decode-only tok/s as a separate field; keep the e2e number under an honest name (effective_tokens_per_second — the ecosystem is converging on exactly this metric, keep both).
MED — suite divergence check uses rep-1 text only and arm order never permutes within (rep,qid): store per-rep committed-text hashes and check all reps; rotate arm order per rep. MED — dspark greedy baseline runs COLD (first forwards of the process: pipeline creation, heap growth) while the spec loop runs warm -> in-report baseline comparison biased pro-spec; add one untimed warmup generation before the baseline.
LOW — decode_time_to_first_token means different things on chain vs non-chain paths (8-token flush vs true first token): rename per-path or normalize. LOW — suite silently averages 1-of-3-rep arms with pstdev(n=1)=0: print rep counts, use stdev.

OBSERVABILITY (cheap, do in the same pass): per-token timestamp vector into GenerationStats (decode_start.elapsed() is already computed per token) -> report p50/p95/p99 inter-token gap + max stall — spec decode commits in bursts and deferred-readback batches flushes, so JITTER becomes a standing gate column (mean tok/s can improve while streaming gets chunkier); raw per-round wall ms in dspark reports (residuals are cost-model-relative today, raw walls make reanalysis free); getrusage ru_maxrss + Metal allocator stats in every report; git rev + fork pin of the binary in report JSON; loadavg + thermal-pressure snapshot at run start (makes the quiet-machine protocol auditable post hoc).

GATES: nextest 59 -> rebuild (nextest clobbers the metal binary) -> stub oracle -> golden text byte-identical (these are timer/reporting changes; ANY text change = bug). Then ONE quiet rotated bench + suite calibration run to establish the v2 baselines, and a campaign-log note that v1 numbers carry the ~+6% chain inflation. Full audit details with file:line refs are in the session log of 2026-07-13; re-derive from the code if unclear — do not trust this summary over the code.
