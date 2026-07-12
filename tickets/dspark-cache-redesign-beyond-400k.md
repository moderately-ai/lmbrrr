---
id: dspark-cache-redesign-beyond-400k
title: DSpark training-cache redesign for beyond-400k corpora (sharded/streaming or compressed captures)
status: todo
priority: p2
dependencies: []
related: [scale-dspark-training-corpus-modal]
scopes: [evals]
shared_scopes: [docs/research]
paths: []
tags: [dspark, campaign-1000]
---
## Why

The corpus scaling law has not flattened: gsm8k τ 3.54 (40k) → 3.91 (120k, +0.37/3×) → **4.41 (400k, +0.50/3.33×)** on the fakequant deployment target. The slope rule from round 4 says the τ path is more corpus, not the MTP head. But the fused-pipeline cache is ~6.5 GiB per 1k conversations on NVMe: 400k ≈ 2.6 TiB was already near the ephemeral-disk ceiling, and full-scale PerfectBlend (~1.4M convs) implies ~9 TiB — structurally impossible with the build-whole-cache-then-train design.

## Options (from the round-3/4 pipeline analysis, docs/research/dspark-corpus-scaling.md)

1. **Sharded/streaming prep**: build cache shards on N containers, stream shards into the trainer per epoch window (dataloader reads shard k+1 while training on shard k). Keeps the fused-container win (no volume round-trip for the bulk data), removes the single-disk ceiling.
2. **Capture compression**: the cache stores per-position target hiddens for 5 target layers; quantizing captures (bf16 → fp8/int8 with per-tensor scales) halves-to-quarters the footprint — a 1.4M corpus at fp8 ≈ 2.3 TiB fits today's design. Needs a short numerics gate (train a 40k probe on compressed vs raw captures, τ must match within noise).
3. Combination: compression first (cheap to validate, biggest ratio), sharding only if full-scale still doesn't fit.

## Acceptance

- 40k probe: compressed-capture arm τ within noise of raw-capture arm (same epochs/seed policy) — the go/no-go for option 2.
- A costed launch spec for the next scale step (800k–1.4M) that fits the disk ceiling, with wall-clock and $ estimates in the style of the round-2 budget table.
- No training launch without user sign-off (standing rule).
