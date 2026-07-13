---
id: eval-vision-path-smoke
title: "EVAL (scope decision needed): vision-path perf smoke — encode latency + parity re-check on the deployed pin"
status: todo
priority: p3
dependencies: []
related: []
scopes: [evals]
shared_scopes: []
paths: []
tags: [eval-wave, scope-question]
---
SCOPE QUESTION FOR THE USER: the campaign is text-decode; MiniCPM-V-4.6's vision path was ported and parity-validated (tickets validate-minicpm-image-parity, validate-minicpm-vision-feature-parity — both done) but has had ZERO attention through five waves of kernel/dispatch changes. Nothing in the gate battery exercises it, so a regression (or a shader that no longer compiles) would be invisible. If images matter to the product, TTFT includes vision encode and it belongs on the latency surface.

IF IN SCOPE: (a) SMOKE: re-run the parity harness from the done tickets (evals/minicpm_v46_image_oracle.py + the runner's image path — read those tickets' comments for the exact commands; they are the source of truth) on the CURRENT deployed pin; pass = parity within the tolerances those tickets established. (b) PERF ROW: time vision encode for one 448x448 and one high-res tiled image, 5 reps median, quiet machine; record ms + where it lands relative to text TTFT. (c) Add the smoke to evals/run_gate_battery.sh --extended so vision can never silently rot again.
IF OUT OF SCOPE: comment 'text-only confirmed' + close; leave a note in the rollup that vision is explicitly unmaintained.
