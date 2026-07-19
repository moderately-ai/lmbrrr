---
id: sprinter-approx-verify-audit
title: SPRINTER approximate-verify-then-audit (exactness-breaking escape hatch)
status: todo
priority: p2
dependencies: []
related: [relaxed-typical-acceptance-mode, calibrate-dspark-confidence-head, program-full-bonsai-acceleration-program-2026-07-19-canonical]
scopes: [inference/speculative]
shared_scopes: []
paths: []
tags: [route-map, verify-structure, research]
---
Bucket C / C4. Cheap accept-PREDICTOR most of the time, fall back to exact GPU verify only on predicted rejects + periodic full audit. SPRINTER (arXiv 2502.04557): <1k-param single-layer verifier, 1.64x latency / 8.3x fewer FLOPs / ~11 accepted tok/cycle vs 2.17. NOT lossless -- output distribution becomes (1-eta_FP)*p + eta_FP*q, deviation scales with false-positive rate. The single biggest COMPUTE cut, in the same bounded-quality class as margin acceptance. Only if drift is acceptable; gate with the quality-reference battery. Companion refs: SpecPV 2512.02337, Speculative Verification 2509.24328.
