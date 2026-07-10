---
id: measure-metal-roofline-and-dispatch-overhead
title: Measure Metal roofline and dispatch overhead
status: done
priority: p1
dependencies: []
related: [profile-dspark-verification-throughput-table]
scopes: [runtime/candle, runtime/metal]
shared_scopes: [docs/research]
paths: [src/main.rs, docs/research/metal-roofline-and-dispatch-overhead.md]
tags: [performance, measurement, campaign-1000]
---
## Goal

Establish the denominator for the 1000 tok/s campaign on this host (binned M4 Max, 32 GPU cores, ~410 GB/s): how much bandwidth is actually achievable, what one Metal dispatch costs, and where the current 66 tok/s decode spends its per-token budget. Analysis says the model reads ~1.5 GB BF16 per token (roofline ~270 tok/s), so we run at ~24% efficiency and are dispatch-bound, not bandwidth-bound — this ticket verifies that claim with measurements.

## Acceptance

- Add a `roofline` subcommand that measures: achievable device bandwidth (large tensor copy and large matvec sweeps, GB/s vs size), per-dispatch overhead (timed chains of tiny ops), and matvec throughput at the model's actual shapes (h=1024, inter=3584, vocab=248094).
- Estimate dispatches per decode forward (profiled component scopes plus known ops per scope) and derive a per-token time budget: dispatch overhead vs weight reads vs compute.
- Publish `docs/research/metal-roofline-and-dispatch-overhead.md` with the budget table and a JSON artifact; state the measured efficiency ceiling that Stage 2 (fusion) must close.
