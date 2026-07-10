---
id: keep-deltanet-recurrent-state-f32
title: Keep DeltaNet recurrent state resident in F32 (perf + compounding accuracy bug)
status: todo
priority: p1
dependencies: []
related: []
scopes: [runtime/candle]
shared_scopes: []
paths: []
tags: [performance, correctness, decode-audit-2026-07-10]
---
## Goal

src/qwen35.rs:1073-1084 (decode), 931-934/998 (chunked), 1020-1024/1045 (sequential): the recurrent state is upcast BF16->F32 at the start of every step and downcast F32->BF16 at the end. With state (1,16,128,128) F32 ~1 MB/layer that is 2 casts x ~3 MB x 18 layers ~= 54 MB of traffic + 36 dispatches per token. Worse, the recurrence loses mantissa bits every single token — a compounding accuracy bug for long generations, unlike a one-shot activation cast.

## Acceptance

- Store `recurrent_state` (and conv state if same pattern) in F32 permanently; remove the per-step cast pairs in all three paths (decode, chunked, sequential fallback).
- Caveat carried to fuse-deltanet-decode-step-kernel: `DecodeStateSnapshot` (qwen35.rs:304-312) relies on replace-by-assignment; any future in-place state update must switch the snapshot to copy-on-snapshot.
- Corruption-invariance oracle + strict parity vs a long-generation reference; interleaved A/B bench.
