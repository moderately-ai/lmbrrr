---
id: causal-text-model-generic-decode
title: "FEATURE: CausalTextModel trait + generic generate_tokens + Qwen35CausalLM (plan L4)"
status: todo
priority: p2
dependencies: [linear-source-seam]
related: [ternary-bonsai-27b-support, gguf-loader-qwen35-hybrid, qwen35-27b-config-scaleup]
scopes: [runtime/candle]
shared_scopes: []
paths: [src/qwen35.rs, src/generate.rs, src/minicpm.rs]
tags: [ternary-bonsai, model-compat]
---
## WHY

Plan L4. `generate_tokens` is currently hardwired to `&mut MiniCpmForConditionalGeneration` with a vision-coupled `forward(input, images, downsample_mode, offset)`. A GGUF text model has no vision prefill. Abstract the decode loop over a `CausalTextModel` trait (static dispatch — the loop is the hot path), and give the ternary model a lean `Qwen35CausalLM` (embed + `Qwen35TextModel` + final norm + `lm_head`) that REUSES `Qwen35TextModel` verbatim — it just owns the head `MiniCpm` currently wraps. Models specialize (MiniCpm keeps its vision-prefill `forward` as its own method); the shared decode path is written once.

## WORK ITEMS

1. `trait CausalTextModel`: the exact surface `generate.rs` calls — `clear_cache`, text-core `forward(input, offset) -> logits`, `supports_fused_head_argmax` + the fused-argmax/device-chain decode hooks (reads the model's `lm_head` QTensor), KV truncate/len. Extract by reading `generate.rs` in full first.
2. Make `generate_tokens<M: CausalTextModel>` generic (static dispatch); keep the async-readback / device-chain / fused-argmax fast paths behind trait methods.
3. Impl on `MiniCpmForConditionalGeneration` (its vision `forward(input, images, …)` stays a MiniCpm-only method; the trait's `forward(input, offset)` is the text core).
4. `Qwen35CausalLM::new(cfg, &source, ctx)` (built via [[linear-source-seam]]) — embed + `Qwen35TextModel` + norm + `lm_head` via `ctx.quantized_linear`; impl `CausalTextModel`.

## DONE-WHEN

`generate_tokens` is generic over `CausalTextModel`; MiniCPM decodes byte-identically to today through the trait (parity check), and `Qwen35CausalLM` is a working text-only impl ready for the GGUF loader to construct. Fused-argmax / deferred-readback fast paths preserved for both.

## PROGRESS (2026-07-15) — trait + Qwen35CausalLM DONE + 27B forward verified

- `CausalTextModel` trait (core surface `clear_cache` + `forward(input, offset) -> logits`) + `Qwen35CausalLM` (reuses `Qwen35TextModel` verbatim + the packed `output`-weight head) in `qwen35.rs`.
- **`bonsai_27b_forward` (ignored, on-demand): builds the FULL 27B via `GgufSource` (18.6s, ~7 GB packed) and runs a prefill forward — finite logits, valid argmax. The whole model runs on real ternary weights.**
- candle fork fix (@ ea3dc446, pin bumped): route Q1_0/Q2_0 matmul through `fwd_mv` for ALL m — the mv kernel services prefill (m>1) in one dispatch. This closed the prefill landmine.

REMAINING: the generic `generate_tokens` unification (bring the async-readback / fused-argmax fast paths to the trait; impl `CausalTextModel` for `MiniCpm` too and make the decode loop generic). Not needed for a first coherent-text run — a straightforward greedy loop over the trait suffices for E2E bring-up + byte-match; the fast-path unification is the follow-up.
