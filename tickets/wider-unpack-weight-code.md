---
id: wider-unpack-weight-code
title: Wider byte-aligned ternary weight code (spend spare DRAM BW to cut unpack)
status: todo
priority: p2
dependencies: []
related: [metal-ternary-matmul-kernel, spike-ternary-type42-block-format]
scopes: [quantization, runtime/metal]
shared_scopes: []
paths: []
tags: [route-map, kernel, research]
---
Bucket B / B4. The wall is the per-weight 2-bit unpack+scale-fold stealing FMA issue slots; DRAM is only ~29% used (~3.5x spare). Trade that spare, coalesced-load-friendly bandwidth (the primitive Apple GPUs are good at) for a byte-aligned / partially-unpacked ternary weight code so one 32-bit lane load feeds several MACs with minimal bit-extraction. Most diagnosis-aligned UNTRIED kernel idea; measure issued ALU/LSU instruction count delta, not GB/s.
