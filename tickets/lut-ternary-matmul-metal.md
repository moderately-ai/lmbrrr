---
id: lut-ternary-matmul-metal
title: LUT / T-MAC / Four-Russians ternary matmul on Metal (probe)
status: todo
priority: p3
dependencies: []
related: [metal-ternary-matmul-kernel]
scopes: [runtime/metal]
shared_scopes: []
paths: []
tags: [route-map, kernel, research]
---
Bucket B / B2. Highest ALGEBRAIC ceiling on the ALU-unpack wall: precompute shared partial sums into a table (bit-serial sign+magnitude planes), replace g decode+adds with 1 lookup + 1 add; activations need NOT be quantized. T-MAC arXiv 2407.00088 (4x CPU), LUT-GEMM 2206.09557, FLUTE 2407.10960. VERDICT speculative->NEGATIVE for Metal: Apple GPUs lack a cheap data-indexed register shuffle (simd_shuffle indexes by lane, not data), so a lookup degrades to a divergent threadgroup gather + bank conflicts that refund the saved adds; LUT-Tensor-Core (arXiv 2408.06003) states software LUT loses to dequant on stock GPUs. ACTION before betting: run the CPU T-MAC path (proven) and/or a Metal prototype measured on ISSUED ALU+LSU instruction counts (gputrace), not GB/s.
