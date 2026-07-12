---
id: bf16-recurrent-state-storage
title: BF16 recurrent-state storage (f32 accumulate) in fused DeltaNet kernels
status: closed
priority: p3
dependencies: []
related: []
scopes: [runtime/metal, candle-fork]
shared_scopes: []
paths: []
tags: [kernels, frontier-survey]
closed_reason: wontdo
closed_note: upside <0.05 ms/token after the v2 kernels; bf16-state quality hazard documented upstream
---
## Goal
18 recurrent states are read+written f32 every token (~36 MB/token, ~5-6% of token time at current speed). Store bf16, accumulate f32 in registers; decay/gate math stays f32. Quamba (2410.13229) shows even int8 SSM state survives at 2.8B.

## Acceptance
- Generation-length drift sweep (the hybrid-serving literature warns about progressive state error - our fp32 ACCUMULATE addresses the warned failure mode; verify).
- Expect +2-3% honest; measure per protocol. Sequence AFTER regrid-fused-deltanet-decode-kernel (same kernel, avoid conflicts; the re-grid may change the calculus).
- Note: 2026-07-07 null result predates fused kernels; bytes now visible.
