---
id: metal-wave1-gpu-stream-slimming
title: "Wave 1: GPU-stream slimming — bf16-dst quantized GEMV, DeltaNet cat elimination, rope-in-place"
status: todo
priority: p1
dependencies: []
related: [q4k-soa-plane-repack]
scopes: [candle-fork, runtime/candle]
shared_scopes: [docs/research]
paths: []
tags: [kernels, campaign-1000]
---
## Scope (four independent edits, one pin bump, one integration commit)

All evidence, sites, and procedures: docs/research/metal-decode-utilization.md (§5-final for the census, §7 for instruments).

1. **C1 — bf16-dst quantized GEMV variants (fork quantized.metal + routing)**: every quantized GEMV currently writes F32 (candle-core quantized/metal.rs:406 hardcode) and MixedLinear casts back (src/quantized_linear.rs:68) — ~100-130 cast_f32_bf16 dispatches/step ≈ 0.4-0.6ms. Needs bf16-dst KERNEL variants (kernels store via `device float*` — allocation-only change is insufficient). Scope mv + mc only: the m>=8 mm routing is lm_head-only and the HEAD MUST STAY F32 (argmax numerics), as must the drafter heads (dspark lm_head/markov) — MixedLinear grows a keep-f32 flag; default flips to bf16-out for body projections.
2. **O1 — DeltaNet cat elimination**: forward_fused_decode cats [qkvz|b|a] (qwen35.rs:1060) = 3 copy2d_bf16 × 18 layers = 54/step (dst_s≠d2 defeats the blit; census §5-final). Fix: gated_delta_v2 decode/prep/tree kernels take b/a as separate buffer inputs. b/a are DENSE-PROTECTED (decay gates) — do NOT row-concat them into the quantized projection.
3. **O2 — b+a single dense GEMV**: fuse in_proj_b + in_proj_a into one 64-row dense projection (18 tiny dispatches saved).
4. **C4 — partial-rotary rope-in-place** (candle-nn): rotary_dim=64 vs head_dim=256 forces narrow→contiguous→rope→cat per q,k per attention layer (~24 movement dispatches). Add a rope variant applying rotation over the first rotary_dim lanes in place; caller drops the split (qwen35.rs:286-298).

## Pre-build research (do first, ~1h)
- Confirm MSL float→bfloat store rounding == the cast kernel's rounding (if yes, C1 is bit-preserving and gets the text-identical gate).
- F32-consumer census: enumerate every MixedLinear call site that must keep F32 (lm_head, drafter heads; check recycle top-k and confidence paths).

## Gates
Per slice: nextest, stub oracle (0.75 bound), tree-check, drafter smoke text-compare. Perf: attribution is PACKAGE-level (individual effects < ±10% ambient floor) — one quiet-machine bench + in-loop refit AFTER the wave lands (procedure: dossier §7). Never share a slice with sync-machinery changes.
