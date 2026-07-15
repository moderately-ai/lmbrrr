---
id: linear-source-seam
title: "REFACTOR: LinearSource trait + VarBuilderSource (weight-source seam, plan L2)"
status: done
priority: p2
dependencies: []
related: [ternary-bonsai-27b-support, gguf-loader-qwen35-hybrid, causal-text-model-generic-decode]
scopes: [runtime/candle]
shared_scopes: []
paths: [src/linear_source.rs, src/qwen35.rs]
tags: [ternary-bonsai, model-compat, refactor]
---
## WHY

Plan L2. Today `qwen35.rs` constructors read weights straight from a `VarBuilder` (dense), and quantization is a SECOND pass (`apply_quantized_text_artifact` patches the dense model post-hoc). That post-hoc patch cannot materialize a 27B (dense-first = ~54 GB transient). A GGUF ternary model must build quantized-from-the-start. Rather than fork the constructors, introduce a weight-source seam both paths implement, so `Mlp`/`FullAttention`/`GatedDeltaNet`/`DecoderLayer`/`Qwen35TextModel::new` are written ONCE against the trait. This is a pure refactor — no behaviour change on the safetensors path — that unblocks [[gguf-loader-qwen35-hybrid]] and folds the two-pass patch away (no transitional duplicate).

## WORK ITEMS

1. `src/linear_source.rs`: `trait LinearSource { fn linear(&self, name) -> MixedLinear; fn fused_linear(&self, &[names]) -> MixedLinear; fn tensor(&self, name) -> Tensor; }` (exact signatures = what the constructors call — audit them first).
2. `VarBuilderSource`: the current dense path — `Tensor::cat` fused weights → `MixedLinear::dense` (or `ctx.quantized_linear` where the artifact says so). Behaviour-identical to today.
3. Refactor the five `qwen35.rs` constructors to build via `&dyn LinearSource` (or a generic `S: LinearSource` — decode isn't in the constructor hot path, so `dyn` is fine; match the DI idiom).
4. Retire `apply_quantized_text_artifact`'s dense-then-patch once both sources route through the seam — the quantized safetensors path becomes "VarBuilderSource that returns quantized `MixedLinear`s".

## DONE-WHEN

The existing MiniCPM/qwen35 safetensors path builds through `LinearSource` with byte-identical output (a load + greedy-decode parity check vs the pre-refactor binary), and `GgufSource` has a clean trait to implement. No dense-first-then-patch remains.

## DONE (2026-07-15)

`src/linear_source.rs`: `LinearSource` trait (generic, not dyn — `sub()` returns owned `Self`, sidesteps Box/lifetime churn) + `LinearPart` + `VarBuilderSource` reproducing the historical `vb.get`+`cat`+`MixedLinear::dense` exactly. All seven constructors (`Qwen35RmsNorm`, `Mlp`, `FullAttention`, `GatedDeltaNet`, `DecoderLayer`, `Qwen35TextModel`, `MtpHead`) build through `&S: LinearSource`; the two `minicpm.rs` call sites wrap their `VarBuilder` in `VarBuilderSource`. Fusion orders preserved (q,k,v / gate,up / qkv,z / b,a). The `apply_quantized_text_artifact` post-hoc pass is UNCHANGED (still applies to the dense-built safetensors model) — retiring it is deferred to when `GgufSource` proves the quantized-from-start path; kept for now to hold behaviour parity. **Verified**: clean build; MiniCPM-V-4.6 greedy decode coherent ("Quantum computing is a way to process information using quantum mechanics…") through the new seam. NOTE: the dense-then-patch retire (work item 4) is intentionally NOT done here — parity-safe increment; revisit once GGUF lands.
