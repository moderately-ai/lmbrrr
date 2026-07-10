---
id: remove-per-token-decode-synchronize
title: Remove redundant per-token device.synchronize() in the generate loop
status: done
priority: p1
dependencies: []
related: []
scopes: [runtime/candle]
shared_scopes: []
paths: []
tags: [performance, decode-audit-2026-07-10]
---
## Outcome (2026-07-10)

Landed in lmbrrr 4c03195: per-token synchronize removed (comment documents the stat-attribution shift — decode_model_elapsed is now encode/queue time, the GPU wait lands in sampling_elapsed); trailing syncs removed from argmax_token/argmax_tokens (leading syncs kept — they isolate T_verify attribution for the scheduler table). Parity clean (fixture + 256-token text bit-identical). Rotated-order interleaved A/B: ~+1% steady decode combined with conv-tap precompute — far below the 1-2ms/wait estimate, because waitUntilCompleted on an already-completed queue is nearly free; the real second-wait cost was the pool purge, addressed separately.

## Goal

src/main.rs:5362-5365: `model.forward(...)` is followed by an explicit `device.synchronize()`, but sampling then calls `logits.argmax(...).to_scalar()`, whose `to_cpu` path already does a blit + `flush_and_wait_current`. Every token therefore pays two `waitUntilCompleted` round-trips (~1-2 ms each of OS notification latency per the fork's own comment at commands.rs:244-246) and two buffer-pool purges (see fix-metal-buffer-pool-purge-and-residency-commits). The explicit sync exists only to attribute `decode_model_elapsed`.

## Acceptance

- Drop the per-token `synchronize()` (or gate it behind a `--timing` flag); let the argmax readback be the only wait.
- Also fix the bench helpers `argmax_token`/`argmax_tokens` (main.rs:5644-5662) which sync twice.
- Interleaved A/B decode bench + strict logits parity (should be bit-identical — this changes no math).
