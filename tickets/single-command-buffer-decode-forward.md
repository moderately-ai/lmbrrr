---
id: single-command-buffer-decode-forward
title: Single command buffer decode forward
status: done
priority: p2
dependencies: []
related: [fuse-deltanet-decode-step-kernel, measure-metal-roofline-and-dispatch-overhead]
scopes: [runtime/candle, runtime/metal]
shared_scopes: [docs/research]
paths: [src/qwen35.rs, src/minicpm.rs, src/main.rs, docs/research/single-command-buffer-decode.md]
tags: [performance, kernels, campaign-1000]
---
## Scope correction (roofline, 2026-07-10)

Measured dispatch launch cost is 2.2 us; ~550 launches ~= 1.2 ms of the 16.5 ms token (~7%) — real but secondary. The primary dense inefficiency is small-GEMV underutilization (MLP shapes at ~83 GB/s vs 350 achievable; see docs/research/metal-roofline-and-dispatch-overhead.md), tracked in fix-runner-hot-path-naive-ops and the fork kernel tickets. This ticket stays worthwhile for the ~1.2 ms plus sync elimination, but the stage gate is shared with those tickets rather than owned here.

## Update (decode audit, 2026-07-10)

Static count is ~2100 dispatches/token (not ~550): ~95 per DeltaNet layer x18, ~75-80 per full-attention layer x6, plus norms everywhere. At the default `compute_per_buffer=50` (commands.rs:18,136-141) that is ~42 command-buffer commits per token, each a commandBuffer+fence+commit cycle. Quick win to take immediately: `CANDLE_METAL_COMPUTE_PER_BUFFER=4096` so a token is 1-2 CBs — safety holds (intra-encoder hazards via auto_barrier, cross-encoder via fences; pool reuse is Arc-count-based). Related new tickets carry pieces of this end state: remove-per-token-decode-synchronize (the redundant second wait), two-stage-argmax-device-sampling (device-resident token + periodic eos readback + the per-token 4-byte `from_slice` upload at main.rs:5360 that allocates a fresh buffer every step), fix-metal-buffer-pool-purge-and-residency-commits (waits currently purge the pool, so fewer waits also fixes allocator churn).

## Goal

Encoder/command-buffer discipline for the fixed decode graph: encode the whole per-token forward into as few Metal command buffers as possible, keep buffers resident and pre-allocated, and eliminate host round-trips inside the forward (the only host sync per token should be the final sampled-token readback).

## Acceptance

- Audit where candle currently splits command buffers / syncs during one decode forward (the fork's `feat/metal-encoder-labels` branch helps attribute); document the count before/after.
- Remove per-token host work from the runner loop: cached causal-mask/none path, no CPU tensor construction per step beyond the 1-token input, argmax on device with a single scalar readback.
- Investigate replaying the fixed decode graph (indirect command buffers or persistent encoder reuse) and report feasibility within candle's Metal backend.
- Stage gate (with fuse-deltanet-decode-step-kernel): >= 150 tok/s BF16 single-stream, i.e. >= 55% of the BF16 roofline.
