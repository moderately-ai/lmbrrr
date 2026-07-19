---
id: eagle3-drafter-upgrade
title: EAGLE-3-style upgrade of the DSpark/Bonsai drafter (raise the acceptance vector)
status: todo
priority: p1
dependencies: []
related: [train-dspark-semi-autoregressive-drafter, implement-real-eagle-recurrent-drafter, design-full-dspark-drafter, program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [inference/speculative]
shared_scopes: []
paths: []
tags: [route-map, acceptance, research]
---
Bucket A / A2. The ONLY lever that raises p itself instead of rearranging a fixed acceptance budget. EAGLE-3 (arXiv 2503.01840): (a) drop the feature-prediction constraint / predict tokens directly, (b) training-time-test = feed the drafter its own multi-step rollout during training, (c) fuse low/mid/high target hidden features into the draft head. +1.4x in-paper (accepted len 4.83->6.62, Vicuna-13B). COMPOSES with DSpark (do not replace it). The difference between 'fill 8 rows, saturate tau~4.5' and 'tau~6 makes crossing into the second verify tile pay'. Larger retrain effort; aligns with the DSpark full-implementation directive.
