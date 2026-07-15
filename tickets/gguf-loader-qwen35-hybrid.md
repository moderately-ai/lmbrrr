---
id: gguf-loader-qwen35-hybrid
title: "FEATURE: GGUF model loader + name mapping for qwen35-hybrid"
status: todo
priority: p2
dependencies: []
related: [ternary-bonsai-27b-support, ternary-type42-dequant, qwen35-27b-config-scaleup]
scopes: [runtime/candle]
shared_scopes: []
paths: [src/qwen35.rs, src/pack.rs]
tags: [ternary-bonsai, model-compat]
---
## WHY

lmbrrr loads its model from safetensors + a quant manifest; Bonsai (and any llama.cpp-ecosystem model) ships a single GGUF. Candle can already READ GGUF (`pack.rs` uses `gguf_file::Content`), so this is a loader that: reads the GGUF metadata into a `Qwen35Config`, maps GGUF tensor names to our module tree, and builds the model. Reusable for every future external qwen35-hybrid model, not just Bonsai — the generic on-ramp for the compatibility theme.

## WORK ITEMS

1. Read `qwen35.*` metadata KVs → config (block_count, embedding_length, ffn, head_count/kv, key/value_length, full_attention_interval, ssm.*, rope.*, vocab). Independent of quant.
2. **Name map** GGUF → our VarBuilder names: `blk.N.attn_qkv/attn_gate/attn_norm/post_attention_norm`, `blk.N.ssm_{alpha,beta,a,conv1d,dt,norm,out}`, `blk.N.ffn_{up,gate,down}`, `token_embd`, `output`, `output_norm`. Reconcile our GatedDeltaNet/FullAttention tensor decomposition against the GGUF's fused `attn_qkv`/`ssm_*` (e.g. does GGUF fuse qkvz that we split?). Determine which layers are full-attention (every 4th) vs deltanet from `full_attention_interval`.
3. Wire quant: dense (F16) path first for validation, then the ternary path via [[ternary-type42-dequant]] (dequant-to-bf16 first, packed later).
4. A load-and-forward test on the small F16 reference (or a synthetic) verifying tensor shapes/roles line up.

## DONE-WHEN

`lmbrrr run` can load a qwen35-hybrid GGUF (F16 first) and produce a coherent forward, with the name map documented. Composes with [[qwen35-27b-config-scaleup]] for the 27B and [[ternary-type42-dequant]] for ternary.
