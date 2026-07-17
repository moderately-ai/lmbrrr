---
id: algebraic-ternary-reframes-refuted
title: "[REFUTED] Algebraic ternary matmul reframes (sparsity / low-rank / WHT / mult-free)"
status: closed
priority: p3
dependencies: []
related: [metal-ternary-matmul-kernel]
scopes: [runtime/metal]
shared_scopes: []
paths: []
tags: [route-map, kernel, refuted]
closed_reason: wontdo
closed_note: "REFUTED-M3: sparsity/low-rank/WHT need absent HW; mult-free INCREASES GPU instrs (FairyFuse GPU 130x slower)."
---
Bucket B / B6. Bundle of algebraic reframes, all REFUTED for M3: (1) 2:4 structured sparsity on the ~33% zeros -- needs Sparse Tensor Cores M3 lacks; unstructured zeros serialize a SIMD warp (arXiv 2510.06957 is CPU/NEON scalar). (2) Low-rank + ternary residual -- M3 has no fast fp16 matrix unit either, adds a dense term without removing the ternary one (LQ-LoRA 2311.12023); only surviving angle is an ACCURACY lever (raise acceptance at fixed bits), not throughput. (3) Walsh-Hadamard -- only accelerates multiplication by a structured matrix, not arbitrary trained W (QuaRot-style rotation adds O(K log K), doesn't cut GEMM ALU). (4) Multiplication-free add/sub -- INCREASES GPU instructions (FMA already fuses the multiply); FairyFuse arXiv 2604.20913 GPU port is 130x slower than fp16 cuBLAS. Closing won't-do.
