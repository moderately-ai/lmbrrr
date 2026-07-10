---
id: keep-deltanet-recurrent-state-f32
title: Keep DeltaNet recurrent state resident in F32 (perf + compounding accuracy bug)
status: done
priority: p1
dependencies: []
related: []
scopes: [runtime/candle]
shared_scopes: []
paths: []
tags: [performance, correctness, decode-audit-2026-07-10]
---
## Outcome (2026-07-10)

Landed for the decode path only; chunked/sequential keep the one-cast-per-chunk BF16 store (cheap). The first attempt (F32 stores everywhere) turned the rollback oracle red and triggered a root-cause investigation that ended somewhere more valuable than the ticket: the oracle's bitwise cross-pattern equality was prompt-sensitive by construction (it demands no committed token sit within kernel noise of its runner-up under any chunk split — the validated math prompt had no such token in 160 tokens, the new tides prompt has one at 69 on every binary back to the validated commit on crates.io candle). Replaced by the state-integrity oracle (commit e4e6327): top-8 logit trajectories compared at every shared committed position against a measured noise bound (observed max 0.25/0.375, bound 0.75), token divergences benign only at genuine ties. Under the fixed oracle the decode-F32 store passes cleanly (trajectory envelope unchanged). Perf: neutral in rotated interleaved A/B (paired median -0.75 tok/s; 36 dispatches of ~2100 is under the machine noise floor) — landed on correctness grounds (no compounding BF16 rounding in the recurrence). Protocol lesson: always run the unchanged-binary control on a new validation prompt before attributing its failure to the change.

## Goal

src/qwen35.rs:1073-1084 (decode), 931-934/998 (chunked), 1020-1024/1045 (sequential): the recurrent state is upcast BF16->F32 at the start of every step and downcast F32->BF16 at the end. With state (1,16,128,128) F32 ~1 MB/layer that is 2 casts x ~3 MB x 18 layers ~= 54 MB of traffic + 36 dispatches per token. Worse, the recurrence loses mantissa bits every single token — a compounding accuracy bug for long generations, unlike a one-shot activation cast.

## Acceptance

- Store `recurrent_state` (and conv state if same pattern) in F32 permanently; remove the per-step cast pairs in all three paths (decode, chunked, sequential fallback).
- Caveat carried to fuse-deltanet-decode-step-kernel: `DecodeStateSnapshot` (qwen35.rs:304-312) relies on replace-by-assignment; any future in-place state update must switch the snapshot to copy-on-snapshot.
- Corruption-invariance oracle + strict parity vs a long-generation reference; interleaved A/B bench.
