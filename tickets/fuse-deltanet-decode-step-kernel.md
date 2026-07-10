---
id: fuse-deltanet-decode-step-kernel
title: Fuse DeltaNet decode step into one Metal kernel
status: todo
priority: p1
dependencies: []
related: [measure-metal-roofline-and-dispatch-overhead, optimize-deltanet-chunked-prefill-and-verify-throughput]
scopes: [runtime/candle, runtime/metal]
shared_scopes: [docs/research]
paths: [src/qwen35.rs, src/**, docs/research/fused-deltanet-decode-kernel.md]
tags: [performance, kernels, deltanet, campaign-1000]
---
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
