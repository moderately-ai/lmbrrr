---
id: drafter-block-megakernel
title: "Drafter block megakernel: fused per-layer prep+attention+MLP dispatch (backbone depth ~30 -> ~8)"
status: todo
priority: p2
dependencies: [drafter-rope-in-place, drafter-chain-persistent-kernel]
related: []
scopes: []
shared_scopes: []
paths: []
tags: [spec-lane, metal, depth-wave]
---
If the depth-x-drain model survives the rope and chain results, the remaining backbone wall is ~30 barrier-separated serial stages x ~0.28 ms. The drafter is only 2 layers at m=12 (block_size), dense BF16, GQA sdpa — small enough for a fused per-layer kernel in the style the target's DeltaNet v2 already ships (prep+core+epilogue, two dispatches on one encoder). Target shape: qkv+qknorm+rope in one dispatch, sdpa (existing kernel), o+addnorm+gateup+swiglu+down+addnorm in one dispatch -> ~4 stages/layer. Model prices the backbone 8 -> ~3 ms.

DO NOT START until rope + chain land and re-measure: each removed stage must show ~the calibrated per-stage cost, otherwise the model is wrong again and this is a zero-EV kernel. This is the expensive lever — cheapest levers first, re-measure between each.

Ship criteria: bitwise gates per fused slice, gate battery, M3 ladder + refit + standings.
