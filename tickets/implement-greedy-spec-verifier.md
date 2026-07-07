---
id: implement-greedy-spec-verifier
title: Implement greedy speculative verifier harness
status: todo
priority: p1
dependencies: [design-speculative-decoding-lab]
related: [audit-minicpm-mtp-weights]
scopes: [runtime/candle, inference/speculative]
shared_scopes: [evals]
paths: []
tags: [speculative, verifier, implementation]
---
Add a text-only verifier command that accepts proposed draft token sequences, verifies them with the target model in greedy mode, emits accepted length and verifier timing, and proves exact reconstruction plus intentional suffix rejection.
