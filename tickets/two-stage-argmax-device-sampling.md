---
id: two-stage-argmax-device-sampling
title: Two-stage argmax over 248k vocab + device-resident sampled token
status: todo
priority: p2
dependencies: []
related: []
scopes: [runtime/candle, candle-fork]
shared_scopes: []
paths: []
tags: [performance, decode-audit-2026-07-10]
---
## Goal

Two related tail-latency items on the sampling path:

1. src/main.rs:5279 argmax routes to `call_reduce_contiguous` with grid width = out_length = 1 threadgroup for the whole 248094-element row — one GPU core scanning 496 KB serially, ~30-80 us right before the per-token wait. Two-stage it (partial argmax per block + tiny final pass) or fuse into the lm_head gemv epilogue.
2. src/main.rs:5360 `Tensor::from_slice(&[next_token], (1,1), device)` allocates a fresh Metal buffer per token via `new_buffer_with_data` (always allocates, device.rs:253-267), inserts into the residency set (a commit), and is purged next sweep. Dies naturally with device-resident sampling: keep the sampled token id on device (u32 -> index_select embed) with a periodic every-K-tokens eos readback.

Also note: the non-greedy path (`LogitsProcessor::sample`) copies all 248094 logits to host as F32 (~1 MB) per token — flag this before anyone benchmarks with temperature > 0.

## Acceptance

- Related: single-command-buffer-decode-forward (device-resident sampling is part of its end state; this ticket carries the argmax kernel work).
- Strict parity (greedy path bit-identical); interleaved A/B.
