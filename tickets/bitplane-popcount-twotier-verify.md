---
id: bitplane-popcount-twotier-verify
title: Bit-plane popcount ternary matmul + two-tier verify (int2/int4 activations)
status: done
priority: p2
dependencies: []
related: [metal-ternary-matmul-kernel]
scopes: [runtime/metal, quantization]
shared_scopes: []
paths: []
tags: [route-map, kernel, research]
---
MOTIVATION SHARPENED (2026-07-17): the strip-probes bound ALL in-family mm2d software levers at <=15-18% (the matmul2d op itself ceilings at ~50-53 GB/s at small M; fold epilogue only 7%). This bitplane/masked-add reformulation is now the ONLY untried software family that changes the compute structure (no MAC/MMA per weight). Bar to clear: beat 43 GB/s (shipped mm2d) at m=5-8 on 17408x5120; the roof it chases is the mv's 106 GB/s. Mind metal_notes 15.C: Apple GPUs hide latency via occupancy, not ILP — keep registers lean.

Bucket B / B3. The ONLY route that deletes the 2-bit unpack: w = w+ - w-, the packed 2-bit code IS the sign+mask planes; <a,w> = popcount(a & w+) - popcount(a & w-) with no multiply, no unpack. Needs int-K bit-sliced activations (we are bf16); cost ~ m*K so only int2 (K=2) / int4 (K=4) plausibly beat unpack at m=5-8. Keep committed argmax exact via TWO-TIER verify: cheap int2/binary popcount first-pass FILTER over all speculative rows -> re-verify survivors at fp16. Popcount is 0.125x fp16 per-instruction but packs 32 MACs/instruction. arXiv 2504.12285 / XNOR-popcount refs. Greenfield on Metal; gated on whether low-bit activations hold speculative acceptance (measure tau as the guardrail).
