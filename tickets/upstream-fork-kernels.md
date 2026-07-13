---
id: upstream-fork-kernels
title: Upstream fork kernels to candle
status: todo
priority: p2
dependencies: []
related: [bf16-activation-quantized-matmul-metal, fuse-deltanet-decode-step-kernel, optimize-deltanet-chunked-prefill-and-verify-throughput, integrate-dspark-block-runner]
scopes: [candle-fork]
shared_scopes: [docs/research]
paths: [docs/research/candle-upstreaming.md]
tags: [fork, upstream]
---
## Goal

Upstream the campaign's candle fork work as PRs so the fork pin stays short-lived (README goal: "as we progress we'll upstream improvements"). No local speed value; keeps the dependency healthy.

## Acceptance

- `KvCache::truncate(n)` (rewind for speculative rejection) proposed upstream with tests.
- i32 cast surface (cast.metal + dispatch rows, 7b6d1981), gemv routing + export (81167a3e, ff3666ec), pool-sweep gating + batched residency commits (907dd0bf), BTreeMap ordered buffer pool (733bfcfd) proposed upstream.
- BF16-activation quantized matmul + batched GEMV Metal kernels proposed upstream.
- DeltaNet/GLA kernels offered upstream (or as a candle-transformers model contribution) once stabilized in lmbrrr.
- Track PR links and review status in the doc; follow upstream conventions, raise disagreements in PR discussion.
