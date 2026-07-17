---
id: m5-matrix-unit-roadmap
title: M5 neural accelerators / Metal 4 tensor ops (int8 verify) -- roadmap
status: todo
priority: p3
dependencies: []
related: [eval-matmul2d-uint4b-tensor-op]
scopes: [runtime/metal]
shared_scopes: []
paths: []
tags: [route-map, substrate, research]
---
Bucket D / D1 (roadmap, not an M3 lever). M5 ships the first dedicated matrix units on Apple GPUs (1024 FP16 FMA/cyc/core, 2048 INT8 OPS/cyc); A19 measures 7.5 TFLOPS FP16 / 13.5 TOPS INT8 (int8 ~1.8x fp16), MLX-on-M5 ~3.3-4x prefill over M4. On M5, int8-both-operands + the tensor unit engage and the ~3.5-4x fix the roofline predicts becomes reachable. Revisit the int8 verify path + matmul2d integer accumulate when M5-class hardware is in the fleet. Refs: Rigel arXiv 2606.12765, tzakharko A19/M5, Apple ML Research M5.
