---
id: record-hidden-state-traces
title: Record Qwen3.5 hidden-state traces for drafters
status: done
priority: p1
dependencies: [implement-greedy-spec-verifier]
related: []
scopes: [runtime/candle, model/qwen, inference/speculative]
shared_scopes: [evals]
paths: [src/main.rs, src/qwen35.rs, src/minicpm.rs, docs/research/hidden-state-trace-export.md, tickets/record-hidden-state-traces.md]
tags: [speculative, traces, qwen]
---
Add optional hidden-state capture for selected Qwen3.5 layers and export prompt ids, generated ids, selected hidden states, logits, and timing artifacts for draft-model training and acceptance analysis.
