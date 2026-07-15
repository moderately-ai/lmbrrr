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

## VERIFIED DERIVATIONS (2026-07-15, from full reads of `qwen35.rs` constructors vs the Bonsai GGUF)

`qwen35_config_from_gguf` must produce (checked against `GatedDeltaNet::new`/`FullAttention::new`):
- `key_dim = linear_key_head_dim·linear_num_key_heads`, `value_dim = linear_value_head_dim·linear_num_value_heads`, `conv_dim = key_dim·2 + value_dim`. Bonsai: `linear_num_value_heads=48` (=`time_step_rank`), `linear_num_key_heads=16` (=`group_count`), `linear_{key,value}_head_dim=128` (=`state_size`) → `value_dim=6144`, `conv_dim=4096+6144=10240`.
- Full-attn `q_out = num_attention_heads·head_dim·2 = 24·256·2 = 12288` (the fused q+gate = Bonsai `attn_q`); `kv_out = num_key_value_heads·head_dim = 4·256 = 1024`; `head_dim=256`.
- `layer_types` from `full_attention_interval=4`: Full where `(i+1)%4==0`, else LinearAttention (confirm phase vs prism `qwen35.cpp`).
- Fields with no GGUF source, set explicitly: `hidden_act=silu`, `attention_bias=false`, `tie_word_embeddings=false` (`token_embd`/`output` are separate tensors).
- Transforms: `A_log = (-ssm_a).ln()` (model loads `A_log`, does `.exp()`; GGUF stores `ssm_a = -exp(A_log)`); `ssm_conv1d [d_conv, conv_dim] → (conv_dim, 1, d_conv)` (transpose).

Envelope: packed ternary body ~7 GB + `token_embd` (packed or ~2.5 GB dequant) + KV for 64 layers → cap decode ctx well under 262144 on the 18 GB M3.

## DONE-WHEN

The 27B qwen35-hybrid config loads and decodes coherently within a documented M-series memory envelope; no 0.8B-specific hardcodes remain.
