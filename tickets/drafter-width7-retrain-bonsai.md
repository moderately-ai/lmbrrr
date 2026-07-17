---
id: drafter-width7-retrain-bonsai
title: Width-7 Bonsai DSpark drafter retrain (fill the flat m=8 verify tile)
status: in-progress
priority: p1
dependencies: []
related: [ternary-bonsai-27b-support, ternary-decode-profile-optimize, dspark-cache-redesign-beyond-400k]
scopes: [evals]
shared_scopes: []
paths: []
tags: [route-map, acceptance, research]
---
Bucket A / A1. Retrain the Bonsai DSpark drafter at block_size 7 so verify m=8 = the SAME flat mm2d tile as today's m=5 (free on the verify matmul).

ECONOMICS SHARPENED by the v2 round anatomy (2026-07-17, MEASURED in-loop): round wall ~229 ms FLAT in accepted-count; propose is only 30 ms (13%) — the old "drafter-time-bound -> +10-13%" model was wrong about the binding term. And **18/29 margin-3.0 rounds saturate the width-4 cap** (exact: 12/38): the acceptance run-length distribution is cap-truncated. INFERRED from the measured tail: width-7 propose ~52 ms, verify flat -> round ~252 ms; alpha 3.45 -> ~4.5-4.9 -> **~22-23.5 tok/s (+15-22%)** if per-position acceptance holds beyond the cap. The retrain checkpoint's tau-eval decides.

Status: prepare_cache disk blocker FIXED (fused prep_and_train at the 3 TiB ephemeral ceiling, commit 9c19a24 — the 38k cache extrapolates to ~1.6 TiB, it filled the old 1 TiB stage at ~60%). PRODUCTION ROUND LAUNCHED 2026-07-17 (app ap-89auisxIlBubJbljnDrfKj): cache on NVMe -> train8 6 epochs (exp dspark_block7_bonsai_r1) -> auto-eval on the final ckpt. Then: convert via evals/dspark/convert_dspark_gguf.py -> width-7 GGUF -> M3 drop-in (block_size read from GGUF metadata, dspark.rs:104). See memory bonsai-acceptance-drive-plan.
