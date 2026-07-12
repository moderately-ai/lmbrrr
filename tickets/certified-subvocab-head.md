---
id: certified-subvocab-head
title: Certified-exact sub-vocabulary lm_head (CSV-Decode class) - measurement-gated
status: closed
priority: p1
dependencies: []
related: []
scopes: [runtime/metal, runtime/candle, candle-fork]
shared_scopes: []
paths: []
tags: [kernels, frontier-survey]
closed_reason: wontdo
closed_note: "falsified offline: cluster bounds f=0.996, SVD+tail bounds f=0.994; queries 91% out-of-subspace, bounds 50-100x the logit gaps"
---
## Goal
Bit-exact greedy head via cluster upper bounds (k-means over tied embedding, Cauchy-Schwarz cluster bounds, best-first opening, certified stop, <2% full fallback; ~18% of vocab scored). Tied anisotropic embeddings tighten bounds.

## GATE (do first, 10 minutes)
Measure the isolated q4_K lm_head GEMV + argmax. Our roofline doc shows the BF16 head matvec already at ~85% of bandwidth -> prize today may be only +7-16%, growing to ~1.3x as other kernels reach roofline. Only proceed if the measurement and the post-regrid token budget justify 1-2 weeks.

## Acceptance (if pursued)
- Offline: k-means C~2000 on embedding table; per-cluster centroid+radius metadata (~5 MB).
- Metal: gathered-row q4_K GEMV + fused argmax; full-head fallback on failed certification.
- Bit-exact greedy on the full suite (hard gate); spec verify handles l<=12 via per-position bounds + row-set union.
- Fallback design if bounds too loose at d=1024: SVD-Softmax W=128 preview + FEXIPRO tail-norm certification (validate offline on traced hiddens first).
