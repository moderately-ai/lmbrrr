---
id: eval-realworld-conditions
title: "EVAL p3: real-world conditions — battery/Low Power Mode, contended machine (re-opened from rollup rejection)"
status: todo
priority: p3
dependencies: []
related: []
scopes: [evals]
shared_scopes: []
paths: []
tags: [eval-wave, hardware]
---
RE-OPENING the rollup's 'energy — no product signal' rejection with evidence: the product goal is ordinary MacBooks, and (a) Low Power Mode on M4-generation chips cuts GPU power ~3x and GPU perf to ~69% — and many users run LPM-on-battery as a standing setting; on M3-generation the same mode measured NO GPU change (generation-dependent, must be measured not assumed); (b) macOS lets battery and adapter power modes differ, so 'it was fast when I tested it' and 'it is slow on my lap' are both true; (c) users run next to Chrome/Slack — our quiet-machine protocol measures a machine state users never have.

PROCEDURE (one session per machine, cheap): bench medium profile x 3 reps under arms: {plugged+automatic, battery+automatic, battery+LPM, plugged + browser playing 1080p video, plugged + one background CPU spinner}. Report tok/s ratio vs quiet baseline per arm. Optional: powermetrics J/token alongside (relative numbers only — its GPU energy model is documented as unreliable; validate against wall power once if we ever publish energy claims).
DELIVERABLE: the degradation table -> a short 'what users should expect' section in the campaign log, plus a decision: if LPM cuts us ~30%, whether to detect-and-warn (ProcessInfo.isLowPowerModeEnabled) in the product. No optimization work in this ticket.
