---
id: prototype-dflash-block-drafter
title: Prototype DFlash block drafter
status: todo
priority: p2
dependencies: [design-dflash-block-drafter, record-hidden-state-traces]
related: []
scopes: [inference/speculative, runtime/candle]
shared_scopes: []
paths: [src/main.rs, docs/research/dflash-block-drafter-prototype.md]
tags: [speculative, dflash, prototype]
---
## Goal

Prototype a DFlash-style block drafter using existing hidden-state traces and verifier accounting.

## Acceptance

- Implement an offline block proposal probe over captured traces.
- Report block acceptance, wasted tokens, and expected verifier calls.
- Compare the block probe with the EAGLE chain probe on the same prompts.
- Identify the minimum trained component needed for an online runner.
