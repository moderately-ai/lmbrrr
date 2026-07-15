---
id: design-ternary-bonsai-e2e-bringup
title: "DESIGN: end-to-end Ternary-Bonsai-27B spec-decode bring-up + eval"
status: todo
priority: p3
dependencies: [gguf-loader-qwen35-hybrid, causal-text-model-generic-decode, qwen35-27b-config-scaleup, metal-ternary-matmul-kernel]
related: [ternary-bonsai-27b-support, ingest-external-dspark-head]
scopes: [runtime/candle, evals]
shared_scopes: []
paths: []
tags: [ternary-bonsai, model-compat, dspark]
---
## WHY

The capstone that ties the pieces together and proves the compatibility claim. Per the approved plan (`rippling-wobbling-dijkstra.md`) the bring-up is **target-first**: get the ternary 27B target generating coherent, byte-checked text at ≥ llama.cpp tok/s FIRST (L5 `--gguf` → `Qwen35CausalLM` via `GgufSource` → generic `generate_tokens`). The external bf16 DSpark head + speculative decode ([[ingest-external-dspark-head]]) is a **later optional layer**, added once the target path is solid. Also the place to design the integration surface so the next external model reuses it.

## PLAN GATES (target-first, before the DSpark layer)

- **Prefill routing** must be fixed first (see [[gguf-loader-qwen35-hybrid]] finding 1): Q2_0 m>1 routes to the tile-mm `Err` arm; route through `fwd_mv` (mv kernel handles m>1 in one dispatch).
- **E2E parity**: `lmbrrr run --gguf …Q2_0.gguf -p "…"` greedy → committed token ids **byte-match** the llama.cpp golden (`llama-cli -st --temp 0`, captured on the M3 at 13.7 tok/s).
- **Throughput**: decode tok/s on the M3 vs the 13.7 t/s llama.cpp bar (same model/prompt); ≥ parity, stretch = beat it.
- **Memory**: peak ≤ ~8–10 GB (packed weights resident, no bf16 expansion) — the feasibility gate.

## WORK ITEMS (design, then execute)

1. **Wiring**: ternary target ([[qwen35-27b-config-scaleup]] + [[metal-ternary-matmul-kernel]]) as the verify model, external DSpark head ([[ingest-external-dspark-head]]) as drafter, sharing the 5 target-layer hidden taps the head's fc needs. Reconcile with our existing MTP/DSpark loop (which taps verify-pass hiddens) — the tap points differ (5 fixed layers vs last-layer).
2. **Tokenizer**: gpt2 BPE, 248320 vocab, `qwen35` pre, bos 248044 — load from the GGUF tokenizer metadata; confirm chat template (embedded in the GGUF) round-trips.
3. **Scope cut**: text-only decode — the VLM `mmproj` projector is a non-goal for bring-up.
4. **Validation**: (a) target greedy output coherent + matches the F16 reference within ternary error on a few prompts; (b) end-to-end spec decode produces identical committed text to target-greedy with a measured mean-accepted-length; (c) rough decode tok/s on the M-series vs a llama.cpp baseline for the same GGUF (apples-to-apples, per the eval-apples-to-apples methodology).

## DONE-WHEN

Ternary-Bonsai-27B generates coherent text under lmbrrr spec decode on the M-series, validated vs the F16 reference and with a tok/s number; the integration path is documented as the template for future external models. Feeds the broader model-compat theme.

## MILESTONE (2026-07-15): COHERENT ON-DEVICE GENERATION

`bonsai_27b_generate` (ignored, on-demand): builds the 27B + `tokenizer_from_gguf` (gpt2 BPE + byte-level from the embedded vocab/merges), greedy-decodes a ChatML prompt → coherent text ("<think>\nHere's a thinking process:\n1. **Analyze User Input:** ..."). The full target path works end-to-end on-device.

BUG FOUND + FIXED: GGUF stores norm weights **already shifted** (llama.cpp folds the `+1`; `attn_norm` mean 0.97, `output_norm` mean 2.0 — not the raw `w-1` ≈ 0). lmbrrr's `Qwen35RmsNorm` zero-centred `+1` was double-shifting every layernorm → degenerate output. Fixed with `LinearSource::norms_pre_shifted()` (GgufSource=true, VarBuilderSource=false); `Qwen35RmsNorm::new` applies the shift only when not pre-shifted. Safetensors path unchanged.

REMAINING (target path polish): (1) `--gguf` CLI entry in `main.rs` so it's user-runnable (`lmbrrr run --gguf …`). (2) **byte-match** vs the llama.cpp golden (`llama-cli -st --temp 0`) on the M3 — the formal correctness gate; watch for the mrope-as-standard-rope approximation and the chat-template exactness. (3) throughput vs the 13.7 tok/s bar. (4) generic `generate_tokens` unification to bring the async-readback / fused-argmax fast paths to `Qwen35CausalLM`. Then the optional DSpark-head spec layer.
