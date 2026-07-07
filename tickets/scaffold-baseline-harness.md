---
id: scaffold-baseline-harness
title: Scaffold baseline inference benchmark harness
status: done
priority: p1
dependencies: [audit-candle-support]
related: []
scopes: [runtime/candle, evals]
shared_scopes: []
paths: []
tags: [implementation, benchmark]
---
## Goal

Create the first Rust/Candle executable shape for deterministic local inference experiments and performance measurement.

## Work

- Define the crate layout and CLI entrypoint.
- Record prompt, model, device, dtype, tokens/sec, time to first token, memory notes, generated length, and output snapshots.
- Keep the harness simple enough to run before full MiniCPM support exists.

## Acceptance

- A local command can run a tiny supported model or dry-run model metadata and emit structured benchmark output.
- The harness shape is ready to host MiniCPM/Qwen decoder work.
