---
id: drafter-width7-retrain-bonsai
title: Width-7 Bonsai DSpark drafter retrain (fill the flat m=8 verify tile)
status: todo
priority: p1
dependencies: []
related: [ternary-bonsai-27b-support, ternary-decode-profile-optimize, dspark-cache-redesign-beyond-400k]
scopes: [evals]
shared_scopes: []
paths: []
tags: [route-map, acceptance, research]
---
Bucket A / A1. Retrain the Bonsai DSpark drafter at block_size 7 so verify m=8 = the SAME flat mm2d tile as today's m=5 (free on the verify matmul). Agent model: at p~=0.8, alpha 3.0->4.16 (+~0.9 accepted tok/round), but DRAFTER-time-bound -> net ~+10-13% tok/s, not +30%.

Status: pipeline PROVEN on Modal (5-stage green on the width-7 config); production retrain BLOCKED on the prepare_cache OOM (38k cache x2 staging > 1 TiB ephemeral_disk). Unblock = cap max_samples or bump disk, then cache -> train8 6 epochs -> eval tau -> width-7 GGUF for the M3. See memory bonsai-acceptance-drive-plan.
