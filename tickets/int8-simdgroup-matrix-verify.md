---
id: int8-simdgroup-matrix-verify
title: "[REFUTED] int8/int4 integer simdgroup_matrix verify path"
status: closed
priority: p3
dependencies: []
related: [metal-ternary-matmul-kernel]
scopes: [runtime/metal]
shared_scopes: []
paths: []
tags: [route-map, kernel, refuted]
closed_reason: wontdo
closed_note: "REFUTED-HW: no integer matrix datapath pre-M5; int-mul 0.125x fp16 (Rigel 2606.12765). Revisit on M5 (see m5-matrix-unit-roadmap)."
---
Bucket B / B5. There is no integer-operand simdgroup_matrix / matmul2d on M3/M4 (Metal simdgroup_matrix is half/float only; Metal 4.1 matmul2d low-precision frontier is fp8/fp4/MXFP4, no integer accumulate). Synthesizing it on the ALU loses: I32/I16 multiply issues at 0.125x fp16 FMA. Rigel arXiv 2606.12765 (no dedicated matrix unit pre-M5); philipturner metal-benchmarks. Dedicated int8 matrix (~2x fp16) is an M5 story -> see m5-matrix-unit-roadmap. Closing won't-do (revisit on M5).
