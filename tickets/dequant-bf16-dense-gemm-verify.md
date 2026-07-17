---
id: dequant-bf16-dense-gemm-verify
title: "[REFUTED] Dequant-tile->bf16->dense simdgroup_matrix verify (MLX qmm-style)"
status: closed
priority: p3
dependencies: []
related: [bf16-activation-quantized-matmul-metal, eval-matmul2d-uint4b-tensor-op]
scopes: [runtime/metal]
shared_scopes: []
paths: []
tags: [route-map, kernel, refuted]
closed_reason: wontdo
closed_note: "MEASURED-REFUTED 2026-07-17 via gpudebug: MLX qmm f32_limiter 91.55%, 2.8x slower than mm2d at m<=8. metal_notes 15.E."
---
Bucket B / B1 + D3 (same finding). Dequantize each ternary tile once to bf16 in threadgroup memory, then run a clean dense simdgroup_matrix GEMM over the m rows (this IS MLX's affine_qmm_t). Three research agents rated it 'do first' (1.7x compute-bound precedent on M4 Pro PREFILL). MEASURED-REFUTED 2026-07-17 via clean gpudebug profile on the M3: f32_limiter 91.55% (bf16 executes on the f32 pipe -> pays FULL dense-GEMM FLOPs), wall ~3.1 ms gate_up at m=32 = ~2.8x slower than our mm2d's 1.10 ms; and MLX uses the slower qmv path below m=10 anyway. Also answers the 'use higher-precision units' (D3) question: precision-up does NOT beat the packed-operand mm2d at m<=8. See metal_notes.md 15.E. Closing won't-do.
