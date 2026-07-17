---
id: prompt-class-adaptive-drafting
title: Prompt-class-adaptive drafting (recover the prose acceptance gap)
status: todo
priority: p2
dependencies: []
related: [relaxed-typical-acceptance-mode, integrate-dspark-adaptive-scheduler]
scopes: [inference/speculative]
shared_scopes: []
paths: []
tags: [route-map, acceptance, research]
---
Bucket A / A4. Measured: acceptance decay is prompt-class-specific (prose tau~2.3 exact vs math ~3.7 already saturated). Adapt block width / accept-margin per detected class so the prose spans (the only real headroom) get a wider or more relaxed budget while low-entropy code/math stay exact. Cheap, no retrain; composes with the width-7 drafter and margin acceptance.
