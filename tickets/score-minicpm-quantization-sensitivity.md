---
id: score-minicpm-quantization-sensitivity
title: Score MiniCPM quantization sensitivity
status: done
priority: p1
dependencies: [build-minicpm-quantization-calibration-set]
related: []
scopes: [quantization, runtime/candle]
shared_scopes: [evals, docs/research]
paths: [src/main.rs, src/lib.rs, src/quant_sensitivity.rs, docs/research/quantization-sensitivity-scoring.md]
tags: [quantization, calibration, implementation]
---
Add a sensitivity-scoring command for MiniCPM-V-4.6 mixed-precision quantization, following `docs/research/dynamic-quantization-lab.md`.

Acceptance:

- Score candidate modules with weight error, activation error, logit KL/drift, top-1 flip rate, and latency deltas where practical.
- Emit a reusable JSON sensitivity artifact keyed by module name and candidate quantization format.
- Keep text, vision, merger, embeddings, norms, and DeltaNet state families separable in the report.
- Run against the repo calibration set and baseline BF16 model.
