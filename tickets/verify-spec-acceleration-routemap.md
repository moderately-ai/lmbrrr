---
id: verify-spec-acceleration-routemap
title: "EPIC: Bonsai verify + spec-decode acceleration route map (research wave 2026-07-17)"
status: todo
priority: p1
dependencies: []
related: [ternary-bonsai-27b-support, ternary-decode-profile-optimize, drafter-width7-retrain-bonsai, eagle3-drafter-upgrade, weaver-feature-reuse-adapter, prompt-class-adaptive-drafting, tree-speculation-over-dspark, gdn-rollback-free-masked-solve, kvbuffer-deferred-linattn-state, lut-ternary-matmul-metal, bitplane-popcount-twotier-verify, wider-unpack-weight-code, async-cross-engine-draft-verify, sprinter-approx-verify-audit, m5-matrix-unit-roadmap, relaxed-typical-acceptance-mode, ngram-draft-source-mux, metal-ternary-matmul-kernel, eval-matmul2d-uint4b-tensor-op, dequant-bf16-dense-gemm-verify, int8-simdgroup-matrix-verify, algebraic-ternary-reframes-refuted, paradigm-swaps-vs-dspark-refuted, component-aware-self-spec-refuted, eval-harness-validity-fixes]
scopes: [docs/research]
shared_scopes: []
paths: []
tags: [epic, route-map, research]
---
Index of every route surfaced by the 10-agent research wave (2026-07-17) for pushing Bonsai spec-decode past 19.17 tok/s on the M3, reconciled against session measurements. Throughput identity: tok/s ~= (mean accepted tokens/cycle) / (verify + propose + overhead/cycle). CUDA offload is deliberately EXCLUDED (Metal-only for now).

HEADLINE RECONCILIATION: the agents' single most-recommended kernel route (dequant-tile->bf16->dense simdgroup_matrix, = MLX qmm) was MEASURED-REFUTED this session (f32-bound 91.55%, 2.8x slower at m<=8). Measurement beats inference; see dequant-bf16-dense-gemm-verify + metal_notes 15.E.

BUCKET A -- ACCEPTANCE (the live lever): A1 drafter-width7-retrain-bonsai [IN-FLIGHT], A2 eagle3-drafter-upgrade [highest ceiling], A3 weaver-feature-reuse-adapter, A4 prompt-class-adaptive-drafting.
BUCKET B -- CYCLE COST (verify matmul SETTLED; rest is arch state): B2 lut-ternary-matmul-metal [spec->neg], B3 bitplane-popcount-twotier-verify [spec], B4 wider-unpack-weight-code [untried, diagnosis-aligned], B7 gdn-rollback-free-masked-solve [enabler, p1], B8 kvbuffer-deferred-linattn-state. Refuted: dequant-bf16-dense-gemm-verify (B1+D3), int8-simdgroup-matrix-verify (B5), algebraic-ternary-reframes-refuted (B6).
BUCKET C -- VERIFY STRUCTURE (compositional only): C1/C2 tree-speculation-over-dspark [+ gdn-rollback-free-masked-solve], C3 async-cross-engine-draft-verify [marginal single-M3], C4 sprinter-approx-verify-audit [exactness-breaking]. Refuted: paradigm-swaps-vs-dspark-refuted (C5), component-aware-self-spec-refuted (C6).
BUCKET D -- SUBSTRATE: D1 m5-matrix-unit-roadmap [roadmap]. (CUDA D2 excluded by decision. D3 higher-precision folded into the B1 refutation.)

DSPARK-vs-C: DSpark already folds EAGLE feature-conditioning + non-AR diffusion-block proposal; every C route that REPLACES it regresses acceptance. The non-regressive C = KEEP DSpark + add tree (C1/C2) + upgrade drafter training (A2).

RANKED ACTIONABLE: (1) A1 finish width-7 (disk blocker FIXED, production round IN FLIGHT on Modal 2026-07-17); (2) C1 Sequoia-DP'd depth-fill + 1 confidence-placed branch (the CORRECT tree the w=3 test got wrong); (3) B7+C2 Trees-from-Marginals rollback-free solve + lossless wide tree (top lossless lever, wants width-7 first); (4) A2 EAGLE-3 drafter (only true ceiling-raiser). Escape hatch if bounded drift OK: C4 / more relaxed acceptance. Full synthesis + arXiv refs in each child ticket; measured negatives kept as closed(wontdo) records.

TASK-0 CLEARED (2026-07-17): harness v2 + fresh rotated baseline CONFIRMS the standings — plain 14.42 / exact 14.67 (byte-match) / margin-3.0 19.18. New measured anchor for all route economics: round wall ~229 ms FLAT in accepted-count (verify 192 ms = 84%, propose 30 ms = 13%); 18/29 margin rounds saturate the width-4 cap. Every route's value = how much it moves (accept+1) or the 192 ms verify term; propose-side savings are capped at 13%.
