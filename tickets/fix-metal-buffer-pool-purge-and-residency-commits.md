---
id: fix-metal-buffer-pool-purge-and-residency-commits
title: Stop purging the Metal buffer pool on every wait; batch residency-set commits
status: todo
priority: p1
dependencies: []
related: []
scopes: [candle-fork]
shared_scopes: [docs/research]
paths: []
tags: [performance, decode-audit-2026-07-10]
---
## Goal

Fork fix (candle-core/src/metal_backend/device.rs:131-187): both `wait_until_completed` and `flush_and_wait_current` call `drop_unused_buffers`, which evicts every free buffer (strong_count==1) from both pools — exactly the buffers `find_available_buffer` (device.rs:472-486) would reuse. The decode loop waits 1-2x per token, so each token starts with an empty free pool: ~100-300 MTLBuffers re-created via `newBuffer` per token, each inserted into the residency set with a per-buffer `set.commit()` (residency_set.rs:31-43), then all removed again at token end (another commit each). Estimated 0.5-2 ms/token, and it defeats the pooling design entirely.

## Acceptance

- Decouple the sweep from waits: run `drop_unused_buffers` every N waits or on a high-water byte threshold, not on every wait.
- Batch residency updates: one `commit()` after a batch of addAllocation/removeAllocation calls instead of per-buffer.
- Validate in the candle repo (nextest Metal suite) before bumping the lmbrrr rev.
- Interleaved same-session end-to-end A/B in lmbrrr (thermal variance protocol); strict logits parity.
- Record result in docs/research and the fork upstream list (upstream-fork-kernels).
