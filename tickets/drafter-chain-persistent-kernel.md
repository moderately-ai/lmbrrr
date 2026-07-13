---
id: drafter-chain-persistent-kernel
title: Markov chain as one persistent 1-TG kernel (12 serialized dispatches -> 1)
status: todo
priority: p1
dependencies: [megakernel-stage1-drain-probe]
related: []
scopes: []
shared_scopes: []
paths: []
tags: [spec-lane, metal, depth-wave]
---
The fused Markov chain is 2 dispatches/step x 6 steps, strictly serial: fenced 3.4 ms, of which isolated kernel exec is only ~78us/step — the boundary cost (~0.24-0.28 ms/serialized stage in-situ on M3) is ~90% of the wall. A single persistent kernel looping the gamma steps internally (threadgroup-scope sync only; the reduce stage is already 1-TG, and the partial stage's 64 TGs would fold into a strided loop in one TG or use the two-kernel structure within one dispatch via sequential grid... design per drain-probe stage-2) pays the boundary cost once. Expected: 3.4 -> ~0.5-1.0 ms fenced.

GATED ON: megakernel-stage1-drain-probe's decomposition of the 0.28 ms boundary into encode vs pipeline-drain vs commit share — if a 1-TG loop kernel recovers most of it in the synthetic, build the chain version.

Ship criteria: bitwise CPU-reference bench gate extended to the persistent variant (same markov-chain task), byte-identical integration, gate battery, M3 ladder + standings.
