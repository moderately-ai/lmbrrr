---
id: compare-candle-transformers-text-logits
title: Compare Candle text logits against Transformers oracle
status: done
priority: p1
dependencies: [build-transformers-parity-oracle]
related: [optimize-generation-loop-overhead, profile-metal-decode-hot-path]
scopes: [runtime/candle, evals, docs/research]
shared_scopes: []
paths: [src/main.rs, evals/fixtures/minicpm_v46_transformers_text_logits.json, docs/research/minicpm-v46-transformers-parity-oracle.md, docs/research/minicpm-v46-text-logits-parity.md]
tags: [parity, minicpm, logits]
---
## Goal
Close the text logits correctness gate by comparing Candle next-token logits against the committed Transformers oracle fixture.

## Acceptance
- Add a Candle-side command or test helper that runs the three text-only oracle prompts and dumps top-k next-token ids/logits before sampling.
- Compare against evals/fixtures/minicpm_v46_transformers_text_logits.json with documented tolerance.
- Require top-1 token agreement for each prompt and record top-10 overlap/logit deltas.
- Keep the path scriptable for CPU and Metal, with exact model revision, dtype, and device in the output.
- Document any mismatch before using token-rate results to justify optimization work.
