---
id: define-experiment-pivot-gates
title: Define pivot gates for experimental inference work
status: done
priority: p1
dependencies: [build-transformers-parity-oracle, benchmark-runner-token-rate]
related: [design-dynamic-quantization-lab, design-speculative-decoding-lab]
scopes: [docs/research, coordination]
shared_scopes: []
paths: [docs/research/experiment-pivot-plan.md]
tags: [planning, experiments, performance]
---
## Goal
Make the pivot from baseline runner work to experimental changes explicit and evidence-driven.

## Acceptance
- Define correctness gates for text and multimodal paths.
- Define measurement gates for repeatable token-rate comparisons.
- Define profiling gates for choosing between DeltaNet, attention, quantization, and speculative decoding work.
- Produce a short roadmap that says what we do before and after crossing the gates.
