---
id: eval-cold-resume-wire-dvfs
title: "EVAL p2: cold-resume, memory wiring, and GPU ramp — the bursty-interactive latency taxes"
status: todo
priority: p2
dependencies: []
related: []
scopes: [evals, candle-fork]
shared_scopes: []
paths: []
tags: [eval-wave, hardware]
---
WHY: interactive use is BURSTY (type, wait, read), and three documented macOS mechanisms tax exactly that pattern — none measured on our stack: (1) macOS Tahoe UNWIRES GPU memory after ~1s of idle; llama.cpp had to add a residency-set keep-alive heartbeat (PRs #17766, tuned in #24074) because the next burst pays re-wire latency; (2) our fork may not use MTLResidencySet at all (macOS 15+ API; audit needed) so weights may be soft-purgeable under pressure; (3) GPU DVFS ramp on short bursts is publicly UNMEASURED (our research sweep found no credible primary data) — cheap novel measurement.

PROCEDURE:
1. RESIDENCY AUDIT (read code, no bench): grep the fork's metal layer for MTLResidencySet / setPurgeableState / heap usage; record exactly how weights+pools are kept resident and whether the Tahoe idle-unwire applies to us. 30 minutes, do first.
2. COLD-RESUME CURVE: one process, generate 32 tokens, idle T seconds, generate 32 more; T in {0.2, 0.5, 1, 2, 5, 30, 120}; metric = first-token latency of the second burst vs T. Discontinuity around 1s = idle-unwire; around minutes = pool eviction/page-out. Run on M4 Max AND the M3 box (macOS 26 — where Tahoe unwire was reported).
3. DVFS RAMP (novel): powermetrics --samplers gpu_power -i 50 during the first ~300ms of a burst after each idle gap; plot GPU frequency ramp. Arms: display on vs lid-closed-external-display vs Game Mode on. CAVEAT: powermetrics energy numbers are relative-only (its GPU power model is documented as unreliable); frequency is the trustworthy signal.
4. MITIGATION EVAL (only if step 2 shows a cliff): residency-set + heartbeat in the fork (copy llama.cpp's design), re-run the curve, ship if the cliff closes and steady-state is unaffected (gates: golden text, bench within noise).
RECORD: curves + audit note here; if the idle-unwire cliff is real on macOS 26 it goes in the campaign log as a product-latency landmine BEFORE we recommend the macOS 26 upgrade for packed_numeric.
