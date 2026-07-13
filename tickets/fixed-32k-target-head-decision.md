---
id: fixed-32k-target-head-decision
title: "DECISION: 32k sub-vocab target head (quality trade for ~30-40% floor)"
status: todo
priority: p2
dependencies: []
related: []
scopes: [model/minicpm, inference/speculative]
shared_scopes: []
paths: []
tags: [campaign-1000, decision]
---
USER DECISION REQUIRED (changes committed outputs). Post-wave trace prices this as BY FAR the largest remaining floor lever. EVIDENCE: lm_head q4_K (248094 rows, 143MB) = 41% of the step's device read (352.6MB total, counters CSV), in the most launch/issue-limited buffer. Fixed 32k sub-vocab head (7.6x fewer rows, 143->~19MB) removes ~124MB = 35% of device reads -> step 2.68->~1.74ms -> ~575 tok/s GPU, est ~400-450 bench (from ~335-345 today). Every EXACT lever combined (fusion ~+5-8%, mc verify-parallelism spec-only, adaptive-sync ~0.5ms/drafted round) sums to less. COST: changes outputs on out-of-vocab tokens (unlike drafter FR-Spec 32k which is verification-exact). Ships behind a flag + quality report (like --accept-margin). NOTE the CERTIFIED bit-exact sub-vocab head was falsified both bound families (certified-subvocab-head, closed) — this is the UNCERTIFIED quality-trade version. GATE if pursued: build behind flag, measure floor delta + quality ladder (quant-quality + 6-class held-out) -> present tradeoff. DO NOT build without user sign-off (plan Part III 7b menu item b).
