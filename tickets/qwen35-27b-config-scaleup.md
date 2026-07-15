---
id: qwen35-27b-config-scaleup
title: "FEATURE: scale qwen35.rs to the 27B config + M-series memory/perf envelope"
status: todo
priority: p2
dependencies: [gguf-loader-qwen35-hybrid]
related: [ternary-bonsai-27b-support]
scopes: [runtime/candle]
shared_scopes: []
paths: [src/qwen35.rs]
tags: [ternary-bonsai, model-compat]
---
## WHY

Our qwen35 stack is validated at 0.8B (MiniCPM-V-4.6: hidden 1024). Bonsai-27B is the same hybrid arch at hidden 5120 / 64 layers / GQA 24:4 / head_dim 256 / ffn 17408 / full_attn_interval 4 / mrope sections [11,11,10,0]. The model code is config-driven, so this is mostly verification + fixing any hardcoded 0.8B assumptions, plus confirming the memory/perf envelope is viable on the M-series (ternary target ~7 GB weights + KV cache at ctx up to 262144).

## WORK ITEMS

1. Audit `qwen35.rs` for dimensions/counts assumed from the 0.8B config (head_dim, rotary_dim, group_count, ssm inner/state, mrope sections) and make them fully config-sourced.
2. Confirm GQA 24:4 with head_dim 256 (note MiniCPM used different head_dim) flows through attention + the mm2d routing shapes.
3. Memory: KV-cache sizing for 64 layers at large ctx; decide a practical decode ctx cap for M-series RAM. Ternary weights ~7 GB + activations/KV must fit.
4. A tiny generation smoke on the 27B (ternary or F16) confirming coherent output and no shape panics.

## DONE-WHEN

The 27B qwen35-hybrid config loads and decodes coherently within a documented M-series memory envelope; no 0.8B-specific hardcodes remain.
