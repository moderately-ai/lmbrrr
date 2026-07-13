---
id: eval-sampled-decode-scope
title: "EVAL (scope decision needed): sampled decode — speculative-sampling correctness + sampling-path perf"
status: todo
priority: p3
dependencies: []
related: []
scopes: [evals]
shared_scopes: []
paths: []
tags: [eval-wave, scope-question]
---
SCOPE QUESTION FOR THE USER (do not execute the eval before answering it): the ENTIRE campaign measures greedy/argmax decode. If the product ever runs temperature/top-p sampling, two things are unmeasured: (1) CORRECTNESS — speculative decoding with sampling must use the Leviathan/Chen rejection-sampling rule to preserve the target distribution exactly; if the dspark verify path is greedy-only (accept iff draft==argmax), enabling naive sampling on top would be WRONG, not just slow. (2) PERF — the host sampling path (remap_head_id_host + softmax/top-p over 248094 logits on host) has never been benched; it plausibly costs multiple ms/token and would erase kernel wins.

IF IN SCOPE, THE EVAL: (a) code audit: read src/dspark.rs + src/generate.rs sampling branches and state which acceptance rule is implemented — post the answer here with line refs; (b) correctness test: fixed 128-token context, temperature 0.8, generate 20k single tokens with the spec path vs 20k with plain target sampling, chi-square on the token histogram restricted to the top-100 tokens (distributions must match within test power) — this requires seeded RNG plumbing, in scope; (c) perf: bench --sampling arm vs greedy arm, host census (LMBRRR_HOST_TIMERS) to isolate the sampling bucket.

IF OUT OF SCOPE (greedy-only product): comment 'greedy-only confirmed' here, close, and add one guard: a startup warning or hard error when sampling flags are combined with --drafter unless the acceptance rule is distribution-preserving.
