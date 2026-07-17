---
id: bitplane-popcount-twotier-verify
title: Bit-plane popcount ternary matmul + two-tier verify (int2/int4 activations)
status: todo
priority: p2
dependencies: []
related: [metal-ternary-matmul-kernel]
scopes: [runtime/metal, quantization]
shared_scopes: []
paths: []
tags: [route-map, kernel, research]
---
Bucket B / B3. The ONLY route that deletes the 2-bit unpack: w = w+ - w-, the packed 2-bit code IS the sign+mask planes; <a,w> = popcount(a & w+) - popcount(a & w-) with no multiply, no unpack. Needs int-K bit-sliced activations (we are bf16); cost ~ m*K so only int2 (K=2) / int4 (K=4) plausibly beat unpack at m=5-8. Keep committed argmax exact via TWO-TIER verify: cheap int2/binary popcount first-pass FILTER over all speculative rows -> re-verify survivors at fp16. Popcount is 0.125x fp16 per-instruction but packs 32 MACs/instruction. arXiv 2504.12285 / XNOR-popcount refs. Greenfield on Metal; gated on whether low-bit activations hold speculative acceptance (measure tau as the guardrail).
