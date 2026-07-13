---
id: eval-target-tier-hardware
title: "EVAL p1: target-tier hardware runs — M3 Pro box (available NOW via Tailscale) + fleet transfer audit"
status: todo
priority: p1
dependencies: []
related: []
scopes: [evals]
shared_scopes: []
paths: []
tags: [eval-wave, hardware]
---
WHY: every standings number, the CPB=100 knee, READBACK_EVERY=8, kernel-variant rankings, and the DSpark cost model + STS calibration are artifacts of ONE machine (M4 Max, ~546 GB/s, 40 GPU cores). The fleet spans 8x in bandwidth (M1 Air 68 -> M4 Max 546) and 4-5x in GPU cores. Biggest transfer risk: our GEMV threadgroup geometry is tuned to saturate 40 cores and may UNDER-OCCUPY an 8-14-core GPU, making cheap machines worse than the bandwidth-proportional prediction. Second risk: the scheduler's round-admission economics (fixed host cost vs kernel ms) rebalance completely when kernels are 3-4x slower but host cost is similar.

AVAILABLE NOW — M3 Pro box access recipe: ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519_thomas_work_laptop_m4_pro tsanterre@100.66.57.91 (Tailscale node thomass-macbook-pro-2; wake it via user if offline — 1h sleep timer as of 2026-07-13). Specs: M3 Pro, 14-core GPU, 18GB RAM (GPU wired budget ~13.6GB), macOS 26.5.2, Rust 1.97 installed, fork cloned at ~/lmbrrr-work/candle. NOTE the M3 Pro is the fleet's bandwidth OUTLIER (150 GB/s, a 25% regression vs M2 Pro) — compute-rich/bandwidth-poor, which makes it a GOOD stress test for the occupancy question and a BAD proxy for bandwidth scaling; label results accordingly.

PROCEDURE (M3 box, in order):
1. MICRO: run metal_benchmarks nsg-sweep + qmv (already building at ~/lmbrrr-work/candle/candle-metal-kernels). Compare achieved GB/s vs the 150 GB/s roof AND vs our M4 Max fractions: if our kernels reach a similar fraction-of-roof (~80%), geometry transfers; if the fraction collapses, we have an occupancy problem on small GPUs — capture which shapes suffer (expect lm_head to survive, short rows to suffer) and file a geometry-scaling item on q4k-mv-round3.
2. PRODUCT: copy the lmbrrr repo + the target/ artifacts (manifest ~1GB, drafter bundle, cost model, rankings — rsync from the M4; they are NOT in git), cargo build --release, run the gate battery, then bench medium+long and a suite validation pass. Record burst AND a 10-min sustained run (fanned chassis — expect mild droop).
3. ECONOMICS: in-loop cost refit ON THE M3 (run_spec_suite.py --split calibration) -> does the scheduler still choose sane widths with 3-4x slower kernels? Record the refit constants next to the M4's — the RATIO of fixed-host-cost to verify-slope is the number that decides whether DSpark helps at all on base-tier machines (the llama.cpp MTP-on-Metal net-loss report shows spec decode LOSING on an M1 Max because verify overhead exceeds the gain — our economics must be re-derived per tier, not assumed).
4. SCALING CHECK: tok/s ratio M3/M4 vs bandwidth ratio 150/546=0.275 vs core ratio 14/40=0.35 — which one predicts us? (Bandwidth -> we are memory-bound and healthy; cores -> we are occupancy/compute-bound on small GPUs and the kernel program has a second front.)
FLEET GAPS (needs purchase/borrow — user decision): the minimal matrix wants an M2/M4 AIR (fanless throttle, 100-120 GB/s, the modal user machine) and ideally an 8GB config for the memory floor. File the ask with the user after the M3 results bound the risk.
RECORD: full table + refit constants + verdicts here + campaign log; version stamps per protocol.
