---
id: scale-dspark-training-corpus-modal
title: Scale DSpark training corpus on Modal
status: todo
priority: p1
dependencies: [train-dspark-semi-autoregressive-drafter]
related: [benchmark-full-dspark-speedup]
scopes: [evals]
shared_scopes: [docs/research]
paths: [evals/dspark/**, docs/research/dspark-corpus-scaling.md]
tags: [dspark, training, modal, campaign-1000]
---
## Goal

Iteratively scale the DSpark training corpus on Modal CUDA credits, driven by held-out accepted-length curves — the cheapest tau improvement available anywhere in the campaign. The paper trained on 1.3M samples x 10 epochs; round one starts far smaller.

## Notes

- Smoke-run observation (2026-07-10): the target forward in prepare_target_cache fell back to the slow torch DeltaNet path — transformers requires BOTH flash-linear-attention and causal-conv1d for the fast path, and causal-conv1d was skipped (needs nvcc at image build). Fine at 500 samples (~4 min); at full-corpus scale, switch the Modal image to a CUDA-devel base and install causal-conv1d before the big cache runs.

## Acceptance

- A tau-vs-corpus-size curve (held-out, per prompt domain) with at least three corpus sizes, produced by re-running DeepSpec data-gen + training on Modal.
- Domain mix tuned toward the local benchmark classes (math/code-heavy) with the mix documented.
- Stop/continue recommendation per round based on the marginal tau per training dollar; artifacts and manifests versioned per round.
