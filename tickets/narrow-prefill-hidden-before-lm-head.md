---
id: narrow-prefill-hidden-before-lm-head
title: "Prefill: narrow hidden state to last position before lm_head"
status: todo
priority: p1
dependencies: []
related: []
scopes: [runtime/candle]
shared_scopes: []
paths: []
tags: [performance, decode-audit-2026-07-10]
---
## Goal

src/minicpm.rs:508-543: `forward` calls `forward_all_logits` (lm_head over the full sequence) then `narrow(1, seq_len-1, 1)` — every prefill runs an L x 248094 x 1024 matmul and materializes L x 248094 BF16 logits. A 1000-token prompt wastes ~0.5 TFLOP and a ~500 MB logits buffer (which the allocator's next-power-of-two rounding can double). Decode (l=1) is unaffected.

## Acceptance

- Narrow the hidden state to the last position before `lm_head` in `forward`; keep `forward_all_logits` for callers needing dense logits (spec verification uses per-position logits — do not regress the verify path).
- Prefill TTFT bench before/after; strict parity on the returned last-position logits.
