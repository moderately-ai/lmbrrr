---
id: reduce-metal-dispatch-layer-overheads
title: Shave per-op constants in the fork dispatch layer (allocator scan, fence maps, encoder locks)
status: todo
priority: p2
dependencies: []
related: []
scopes: [candle-fork]
shared_scopes: []
paths: []
tags: [performance, decode-audit-2026-07-10]
---
## Progress (2026-07-10)

BTreeMap allocator landed (fork 733bfcfd, validated 128/128 Metal suite; lmbrrr pinned): `find_available_buffer` is a range(size..) early-exit scan. Measured decode-neutral today (CPU encode path is hidden behind GPU execution) — see fix-metal-buffer-pool-purge-and-residency-commits for the A/B. Remaining items (Commands mutex, encoder HashSet, fence-map scans) stay parked until fusion/single-CB work makes the CPU path measurable; measure with Instruments before touching.

## Goal

Per-op constants in the fork, multiplied by ~2100 dispatches/token today (shrinks as fusion lands, but the constants remain for every surviving op):

- `find_available_buffer` (device.rs:472-486) linearly scans all size buckets under a write lock on every allocation for best-fit — replace with a `BTreeMap` range lookup.
- Every op takes the global `Commands` mutex for the whole encode (commands.rs:160-190); every `set_input_buffer`/`set_output_buffer` does a HashSet insert under a second mutex (encoder.rs:119-149); every dispatch runs `auto_barrier`.
- Each new encoder (every ~50 ops at default, plus after every blit) waits on every fence in `prev_ce_outputs` (commands.rs:169-186) — O(outstanding output buffers) map scan and GPU-side serialization.

These are secondary to reducing op count (fusion tickets) — file order of attack accordingly.

## Acceptance

- BTreeMap allocator lookup landed and validated in-repo (nextest Metal suite).
- Fence-map/encoder-lock costs measured (metal-debug-labels or Instruments) and reduced where the win is real; no speculative churn.
- Interleaved A/B in lmbrrr; results in docs/research; upstream candidates noted in upstream-fork-kernels.
