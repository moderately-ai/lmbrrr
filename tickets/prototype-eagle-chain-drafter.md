---
id: prototype-eagle-chain-drafter
title: Prototype EAGLE-style chain drafter
status: todo
priority: p2
dependencies: [record-hidden-state-traces]
related: []
scopes: [inference/speculative, model/qwen]
shared_scopes: [evals]
paths: []
tags: [speculative, eagle, prototype]
---
Prototype a direct-token chain drafter over fused target hidden states. Start with chain verification, report accepted length and draft latency, and require exact greedy output reconstruction before speed claims.
