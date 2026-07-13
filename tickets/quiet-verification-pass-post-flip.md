---
id: quiet-verification-pass-post-flip
title: "Quiet-machine verification pass: 305-bench confirm, pacing A/B, campaign standings refresh"
status: done
priority: p2
dependencies: []
related: [tree-speculation-over-dspark]
scopes: [evals]
shared_scopes: [docs/research]
paths: []
tags: [campaign-1000]
---
Everything measured after ~18:30 on 2026-07-12 ran under elevated ambient load (user builds + Xcode replay). Re-verify on a quiet machine (protocol: dossier §7, memory rule 'wait for quiet machine'):

1. Bench greedy on q4k-full-text at defaults (touched 305.1±1.6 under load; the manifest-flip floor claim should be re-anchored clean).
2. The CPB pacing variants (feeds metal-wave3-commit-pacing).
3. Rotated 6-class suite refresh of the shipped stack (round-4 bundle + q4full + r4q4f cost model) → update the campaign-log standings table with clean numbers.
4. Tree EV re-check on the new economics (greedy 4.03ms in-loop, cheaper chunks — the 'mid-band ties coding' verdict from the K4 break-even is stale; comment outcome on tree-speculation-over-dspark).
