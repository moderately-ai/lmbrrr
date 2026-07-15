---
id: gguf-loader-qwen35-hybrid
title: "FEATURE: GGUF model loader + name mapping for qwen35-hybrid"
status: in-progress
priority: p2
dependencies: [linear-source-seam, metal-ternary-matmul-kernel]
related: [ternary-bonsai-27b-support, ternary-type42-dequant, qwen35-27b-config-scaleup, causal-text-model-generic-decode]
scopes: [runtime/candle]
shared_scopes: []
paths: [src/gguf.rs, src/qwen35.rs, src/main.rs]
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

## PLAN INTEGRATION (2026-07-15) — this ticket = plan L3 (`src/gguf.rs`) + L5 (`--gguf`)

Build on the [[linear-source-seam]]: this ticket is `GgufSource` (the second `LinearSource` impl) + `qwen35_config_from_gguf(&Content) -> TextConfig` + `tokenizer_from_gguf` + the `--gguf`/`ModelSource` entry in `main.rs`. `src/gguf.rs` is NEW (NOT an extension of `pack.rs`, which is a bitwise sidecar cache). NO new `LMBRRR_*` env keys — GGUF changes weights + shape config, not the runtime route config threaded via `ModelCtx`.

- **Fused-linear via requant**: the model uses fused `qkv_proj`/`gate_up`/`in_proj_qkvz`/`in_proj_ba`; Bonsai ships them separate. `GgufSource::fused_linear` builds the fused weight by dequant→cat→requant, which is **bitwise-identity for ternary** (values sit exactly on the grid, so `from_float` recovers the exact codes) with a bounded transient f32 (≤ ~300 MB per fused weight).
- **Quant split**: big weights stay packed Q2_0 via `ctx.quantized_linear`; small ones (`in_proj_ba`, `A_log`, `ssm_a`, `conv1d`, norms/biases) dequant to dense/F32.

## GGUF GROUND TRUTH (2026-07-15, parsed from the actual Ternary-Bonsai-27B-Q2_0.gguf: 851 tensors, 37 KV)

Metadata keys (all `qwen35.*` → TextConfig): `block_count`=64, `context_length`=262144, `embedding_length`=5120, `feed_forward_length`=17408, `attention.head_count`=24, `attention.head_count_kv`=4, `attention.key_length`=`attention.value_length`=256 (→head_dim), `attention.layer_norm_rms_epsilon`=1e-6, `full_attention_interval`=4, `ssm.conv_kernel`=4, `ssm.state_size`=128, `ssm.group_count`=16, `ssm.time_step_rank`=48, `ssm.inner_size`=6144, `rope.freq_base`=**1e7** (not 1e4!), `rope.dimension_sections`=[11,11,10,0], `rope.dimension_count`=64. vocab=248320 (tokenizer.ggml.tokens len), eos=248046, bos/pad=248044, tokenizer.ggml.model=gpt2, pre=qwen35, chat_template embedded. **Derived** (confirmed vs tensor shapes): linear_num_value_heads=48, linear_num_key_heads=16, linear_{key,value}_head_dim=128, value_dim=6144, conv_dim=key_dim(2048)*2+value_dim=10240 (= attn_qkv out ✓).

Name map (our module path → GGUF flat name), all linears **type 42 (Q2_0)**, norms/ssm_a/conv1d/dt_bias **type 0 (F32)**:
- `embed_tokens.weight`→`token_embd.weight` (Q2_0 — dequant to dense table at load), `norm.weight`→`output_norm.weight`, lm_head→`output.weight`.
- `layers.N.input_layernorm`→`blk.N.attn_norm`, `post_attention_layernorm`→`blk.N.post_attention_norm`, `mlp.{gate,up,down}_proj`→`blk.N.ffn_{gate,up,down}`.
- Full-attn: `self_attn.{q,k,v}_proj`→`blk.N.attn_{q,k,v}`, `o_proj`→`blk.N.attn_output`, `q_norm/k_norm`→`blk.N.attn_{q,k}_norm`.
- DeltaNet: `linear_attn.in_proj_qkv`→`blk.N.attn_qkv`, `in_proj_z`→`blk.N.attn_gate`, `in_proj_b`→`blk.N.ssm_beta`, `in_proj_a`→`blk.N.ssm_alpha`, `out_proj`→`blk.N.ssm_out`, `norm`→`blk.N.ssm_norm`, `dt_bias`→`blk.N.ssm_dt.bias`, `A_log`→`blk.N.ssm_a`, `conv1d.weight`→`blk.N.ssm_conv1d.weight`.
GGUF stores weights [in, out] (ne0=in); candle's tensor read yields (out, in) — matches our convention, no transpose. `ssm_conv1d` is (conv_dim, kernel) → reshape to the requested (conv_dim,1,kernel).

## CRITICAL CORRECTION: fused linears must byte-cat packed, NOT dequant→cat→requant

The plan's dequant→cat→**requant** is **lossy for ternary**: requant recomputes per-128-block `d = max|w|`; a dequantized block whose max code is 3 (value 2d) yields `d' = 2d`, which then remaps a code-2 value (d) to 2d — corruption. Fusion cats along OUTPUT ROWS, which never crosses the input-dim quant blocks, so each row's packed blocks are already correct. **Concatenate the raw Q2_0 block bytes** of the parts along dim 0 and rebuild the QTensor (`qtensor_from_ggml(Q2_0, cat_bytes, (out_total, in))`) — exact, no requant. Applies to the fused `qkv`, `qkvz`, `ba`, `gate_up`. `A_log` transform: read `ssm_a` (F32), our tensor = `(-ssm_a).ln()` so the model's `.exp()` recovers `-ssm_a`.

## IMPLEMENTED (2026-07-15) — `src/gguf.rs`

- `qwen35_config_from_gguf(&Content) -> TextConfig`: **DONE + verified** — `gguf::tests::bonsai_config_from_gguf` asserts every field vs the real 7 GB GGUF (passes; header-only so runs on any candle pin).
- `GgufFile` (parsed header + seekable reader) + `GgufSource` (the packed `LinearSource`): name translator (our module path → flat GGUF name, full map), `A_log = ln(-ssm_a)` transform, conv1d reshape, Q2_0 single-part → `ctx.quantized_linear` packed, **fused → packed byte-cat** (exact). Compiles clean.
- candle fork: added the missing `qtensor_from_ggml` Q1_0/Q2_0 arms (`ggml_file.rs`) — GGUF Q2_0 ingestion; without it `content.tensor` on a type-42 tensor hit the "unsupported" catch-all.

INTEGRATED (2026-07-15): candle fork pushed (`tomsanbear` `lmbrrr` @ f4eb38b2), lmbrrr's 4 rev pins bumped, builds clean. **GgufSource verified on real 27B weights**: `bonsai_gguf_source_fused_load` loads blk.0 `attn_qkv`+`attn_gate` packed and asserts the fused linear == the two split linears concatenated — **max abs diff 0.000e0 (bitwise exact)**. Byte-cat fusion + name map + `quantized_linear` + Q2_0 kernel all confirmed together.

REMAINING: (2) `tokenizer_from_gguf` (gpt2 BPE from `tokenizer.ggml.tokens`+`merges`, ChatML template from `tokenizer.chat_template`). (3) `--gguf` flag + `ModelSource` in `main.rs` (L5). (4) construct `Qwen35CausalLM` ([[causal-text-model-generic-decode]]) via `GgufFile::source` and decode. INTEGRATION GATE: commit the candle fork on the `lmbrrr` branch, `git push tomsanbear lmbrrr`, bump the 4 candle rev pins in `Cargo.toml` — user-gated per the commit/push rule.

## CANDLE-CORE ROUTING FINDINGS (from the metal-kernel full reads — must handle here)

1. **Prefill landmine**: candle-core `QMetalStorage::fwd` routes Q2_0 **m>1** to `call_quantized_matmul_mm_t` → the `Err` arm (no `mul_mm_q2_0`). The mv kernel already handles m>1 in ONE dispatch (each `tgpig.y` = one src1 row), so the fix is to route Q2_0 through `fwd_mv` for all m (small candle-core change: extend the `dim(Minus2)==1` short-circuit, or a `dtype-has-no-mm-kernel` predicate). Decode (m=1) already works. Do this before the first 27B prefill.
2. **Packed embedding**: `token_embd` kept packed Q2_0 would hit `call_quantized_get_rows` → the `Err` arm (no `get_rows_q2_0`). Decide: port `get_rows_q2_0` (prism has it) OR dequant `token_embd` to bf16 at load (vocab 248320 × 5120 × 2 B ≈ 2.5 GB — fits within the ~8 GB envelope, simplest). Lean dequant-at-load unless the 2.5 GB is tight.
