---
id: fuse-deltanet-decode-step-kernel
title: Fuse DeltaNet decode step into one Metal kernel
status: in-progress
priority: p1
dependencies: []
related: [measure-metal-roofline-and-dispatch-overhead, optimize-deltanet-chunked-prefill-and-verify-throughput]
scopes: [runtime/candle, runtime/metal]
shared_scopes: [docs/research]
paths: [src/qwen35.rs, src/**, docs/research/fused-deltanet-decode-kernel.md]
tags: [performance, kernels, deltanet, campaign-1000]
claimed_from: todo
assignee: claude
lease_expires_at: 1783735054
---
## Outcome (2026-07-10): decode step DONE — 77 -> 145 tok/s (+88%)

Landed: fork kernel gated_delta_decode_bf16 (rev fdd06d7c) + lmbrrr src/fused_deltanet.rs wrapper (multi-output tensors built from MetalStorage; all-public APIs, no candle-core changes) + GatedDeltaNet::forward_fused_decode behind eligibility checks and LMBRRR_UNFUSED_DELTANET=1. Gates all green (tests/fixture/oracle both prompts/coherent-text drift advisory). Rotated same-binary A/B: 140.7-147.5 vs 69.8-79.5 tok/s, zero overlap — Stage-2 neighbourhood (~54% BF16 roofline) reached on this lever alone. NOTES: (1) trajectory-oracle envelope crept to 0.625/0.75 with numerics changes stacking — rederive the bound if another numerics change lands; (2) fork lib.rs root re-export of call_gated_delta_decode still missing (lmbrrr imports via kernels:: path) — fold into the next fork commit; (3) REMAINING SCOPE: the verify-chunk variant (chunked WY kernel) — verify is now ~25ms vs 7ms/token decode, the spec loop's dominant cost; and the packed single-gemv projection (4 gemvs + cat -> 1 gemv) for another ~3 dispatches/layer.

## Kernel design (2026-07-10, pre-implementation)

One dispatch per layer per decode token, grid = 16 threadgroups (one per v-head) x 128 threads (one per v_dim column). No cross-threadgroup dependency exists: TG h owns its head's q/k/v conv channels (3x128 of the 6144), its gate scalars, its 128x128 state slab, its norm+z gating.

Per-TG phases (barriers within TG only): (1) depthwise conv on the head's 384 channels (window 4, taps from constant buffer) + silu -> q,k,v vectors in TG memory; (2) gate scalars g = -a_log_exp*softplus(a_in+dt_bias), beta = sigmoid(b_in); (3) l2norm q,k (TG reduction); (4) delta rule with thread j owning state column j as a 128-float thread-local array (per-head state slab 64KB exceeds the 32KB TG memory, so state lives in device memory, row j contiguous under a [v,k]-transposed layout for coalescing): kv_mem[j] = dot(k, col_j) via k in TG memory, delta_j = (v[j]-kv_mem[j])*beta, col_j = exp(g)*col_j + k*delta_j, out[j] = dot(q, col_j); (5) group-RMSNorm over out (TG reduction, weight from buffer) * silu(z[head]) -> output slice.

Integration decisions:
- Fuse the four in_proj gemvs into ONE gemv first (concat weights at load: 6144+2048+16+16 = 8224 rows) - 4 dispatches -> 1 and gives the kernel a single packed activation input.
- States are NOT mutated in place: kernel writes new conv/recurrent buffers (pool makes the alloc cheap) and Rust reassigns - preserves the DecodeStateSnapshot replace-by-assignment invariant verbatim (no copy-on-snapshot needed).
- Recurrent state F32 in/out ([v,k] transposed layout, kept F32-resident per keep-deltanet-recurrent-state-f32); activations BF16 in, BF16 out; internal math F32.
- Vehicle: fork-side. New gated_delta.metal in candle-metal-kernels + call_gated_delta_decode, exposed as a candle_nn::ops function returning (out, new_conv_state, new_recurrent_state) built from MetalStorage directly (multi-output; CustomOp1-3's single-tensor contract doesn't fit). lmbrrr calls it from GatedDeltaNet when l==1 on Metal, env LMBRRR_UNFUSED_DELTANET=1 for the reference path.
- Gates: unfused-vs-fused logits parity within noise (advisory drift report per campaign policy), state-integrity oracle both prompts (0-draft rounds route l==1 through this kernel), 33/33 tests, rotated interleaved A/B on plain decode + spec loop.
- Layout note: the existing per-tensor layout is conv_state [1,6144,4], recurrent_state [1,16,128,128] (k,v); the kernel takes recurrent as [1,16,128(v),128(k)] - transpose once at load/first use, keep resident in kernel layout thereafter (decode-only; chunked path converts on read... simpler alternative if that couples the paths too tightly: transpose in/out inside the kernel epilogue, costs nothing vs the win).

## Scope correction (decode audit, 2026-07-10)

Static dispatch count per DeltaNet layer per token is ~95, not ~30, and the recurrent rule (qwen35.rs:1050-1085) is only ~29 of them. The rest: depthwise conv ~15 (incl. cat copy2d pair and state-update narrow/copy), gate chain 10 dispatches on 16-element tensors (qwen35.rs:779-784), output gating with 4 avoidable casts (F32->BF16 at 1084 immediately re-upcast inside norm.forward at 817; z silu F32 round-trip at 818 — BF16 usilu exists), l2norm 2x5 dispatches. A rule-only fusion leaves ~60 dispatches/layer behind. The kernel MUST cover conv + gates + l2norm + rule + group-RMS-norm + z-gating: inputs are the 4 gemv outputs + conv/recurrent state; outputs are the gated value + updated states. That takes a layer 95 -> ~10 dispatches and intermediate traffic ~16 MB -> ~2 MB; across 18 layers ~1500 dispatches and ~250 MB per token (~4-6 ms of the 15.8).

Snapshot caveat: `DecodeStateSnapshot` (qwen35.rs:304-312) relies on states being replaced by assignment, never mutated in place. An in-place fused kernel must switch the snapshot to copy-on-snapshot. See also keep-deltanet-recurrent-state-f32 (state should be F32-resident before/with this kernel).

## Goal

Collapse the single-token GatedDeltaNet decode step — conv state update + silu, gates (sigmoid/softplus-style), l2norm, recurrent delta rule, output RMSNorm + silu gate — from ~30 small tensor dispatches per layer into one (or two) custom Metal kernel dispatches per layer. 18 DeltaNet layers dominate the per-token dispatch budget at hidden 1024.

## Acceptance

- Implement as a Candle `CustomOp` with `metal_fwd`, compiling embedded MSL at runtime via `MetalDevice::new_library_with_source` (no candle-core fork required; UgIOp1 in candle-core/src/custom_op.rs is the template). State tensors updated in place via the inplace-op variant.
- Numerics: advisory drift report against the unfused path (logits parity + long-generation text diff) published with the change; drift is accepted under the campaign quality bar but must be visible.
- Measure decode tok/s before/after on the standard bench matrix; report the dispatch-count reduction per token.
- Keep the unfused path behind a flag for oracle comparisons.
