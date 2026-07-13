---
id: eval-distribution-cost
title: "EVAL p3: distribution cost census — deployable bytes, quantize-from-HF time, first-launch compile"
status: todo
priority: p3
dependencies: []
related: []
scopes: [evals]
shared_scopes: []
paths: []
tags: [eval-wave]
---
WHY: everything a user needs lives in target/ and is 'unrebuildable without hours of work' (per the gate-battery ticket) — that statement is a distribution problem wearing a dev-workflow hat. Nobody has quantified what shipping this stack costs.

CENSUS (measurement only): (a) deployable byte census: release binary size, manifest tensor bytes, drafter bundle, rankings/cost-model JSONs, tokenizer — the download a user pays; (b) timed end-to-end quantize-from-HF-weights run (the `quant` command path) on the M4 and once on the M3 box — the alternative to downloading the manifest; (c) first-ever-launch TTFT breakdown: weight load (cold FS cache — run once after `purge`), runtime shader compile of our kernel library, prefill — vs warm relaunch (coordinates with eval-latency-surface's cold-start routing rule; llama.cpp/MLX both ship PREBUILT metallibs and llama.cpp split theirs into 20 parallel-compiled libraries specifically to kill this tax — if our runtime compile costs >1s, file an MTLBinaryArchive/precompiled-metallib lever ticket with this census as receipt).
DELIVERABLE: the three tables as a comment here; no shipping decisions in this ticket.
