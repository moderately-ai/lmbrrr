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

## FULL RECIPE (2026-07-15) — arch confirmed identical to our qwen35.rs; verified against Bonsai GGUF + prism fork `src/models/qwen35.cpp`

Bonsai-27B IS the exact hybrid our `qwen35.rs` implements (gated DeltaNet + gated full-attention, QK-norm, conv). No new modeling — a name-map + a few weight *transforms*. Layer type by `full_attention_interval=4` (idx%4==3 → full-attn, else DeltaNet).

**Name map** (GGUF → our VarBuilder name):
- Globals: `token_embd`→`embed_tokens.weight`, `output`→`lm_head.weight`, `output_norm`→final `norm.weight`.
- Per layer: `attn_norm`→`input_layernorm`, `post_attention_norm`→`post_attention_layernorm`, `ffn_{gate,up,down}`→`mlp.{gate,up,down}_proj`.
- DeltaNet: `attn_qkv`→`in_proj_qkv`, `attn_gate`→`in_proj_z`, `ssm_beta`→`in_proj_b`, `ssm_alpha`→`in_proj_a`, `ssm_out`→`out_proj`, `ssm_norm`→`norm` (our GDN), `ssm_dt.bias`→`dt_bias`.
- Full-attn: `attn_q`(fused q+gate, 12288=2×6144)→`q_proj`, `attn_k`→`k_proj`, `attn_v`→`v_proj`, `attn_output`→`o_proj`, `attn_q_norm`→`q_norm`, `attn_k_norm`→`k_norm`.

**Transforms (the no-naive bits):**
1. `A_log = ln(-ssm_a)`. GGUF `ssm_a` is stored PRE-computed as `-exp(A_log)` (ref: `gate = softplus(alpha+dt) * ssm_a`), but our loader expects raw `A_log` and does `exp` itself (`a_log_exp_f32`, negated downstream). So invert: our `A_log` = `(-ssm_a).ln()`.
2. conv1d: GGUF `ssm_conv1d [d_conv=4, conv_dim=10240]` → our `(conv_dim, 1, d_conv)` = transpose(0,1).unsqueeze(1).
3. Keep the Q2_0 weights QUANTIZED (do NOT dequant to bf16 — see constraint below); norms/biases/`ssm_a`/`conv1d` are F32, load direct.

**DECISIVE CONSTRAINT: dev machine RAM = 38.6 GB.** A 27B dequantized to bf16 is ~54 GB → does NOT fit. So the model MUST run on the packed ternary weights (7.15 GB) via [[metal-ternary-matmul-kernel]] (peak ~8 GB per prism's numbers). This makes the Metal kernel a **feasibility gate**, not a perf option — the loader must build QTensors, not dequant-to-bf16.
