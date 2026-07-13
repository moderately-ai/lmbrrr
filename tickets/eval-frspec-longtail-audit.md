---
id: eval-frspec-longtail-audit
title: "EVAL p1: FR-Spec 32k draft-vocab long-tail damage — per-class accepted-length audit incl. multilingual"
status: todo
priority: p1
dependencies: []
related: []
scopes: [evals]
shared_scopes: []
paths: []
tags: [eval-wave, dspark, quality]
---
WHY: our drafter drafts from an FR-Spec 32k reduced vocabulary. NVIDIA SPEED-Bench's headline finding (huggingface.co/blog/nvidia/speed-bench, Figure 4): FR-Spec-style vocab pruning is near-free on coding/math but SUBSTANTIALLY degrades accepted length on multilingual, RAG, and summarization — and the damage is 'largely invisible in low-diversity benchmarks'. Our 6-class validation suite has no multilingual class and modest per-class diversity, so we may be flying blind on exactly the classes FR-Spec hurts. NOTE this is a SPEED audit, not correctness — verification is full-vocab so output text is unaffected; the cost is drafts that can never match rare tokens -> lower acceptance -> lower tok/s on affected content.

PROCEDURE:
1. PROMPT ADDITIONS: add to evals/prompts/ two new classes: multilingual (>= 6 prompts spanning non-Latin scripts — zh, ja, ru, ar at minimum; generation-heavy tasks like 'translate and continue') and rare-token/RAG-style (quote-heavy extraction from pasted context with proper nouns). Keep them OUT of the standings mean until blessed; tag as audit classes.
2. MEASURE: same binary, two arms A/B via the drafter's draft-vocab setting (the 32k FR-Spec ranking vs full-vocab drafting — check how the round-4 drafter bundle encodes its head: if the drafter head is STRUCTURALLY 32k (trained that way), a full-vocab arm does not exist and the eval becomes: measure per-class acceptance on the new classes and compare against the code/math classes' acceptance to quantify the differential; state clearly which form applies after reading artifacts/dspark-drafter-round4's config).
3. METRICS (per spec-decode reporting standards): mean accepted length tau per class AND the acceptance-length DISTRIBUTION (per-round histogram — the mean masks heavy tails), fraction of drafted rounds, tok/s per class. Suite: python3 evals/run_spec_suite.py with the audit classes, --per-class 3 --reps 3, plus per-round data from the dspark report JSONs.
4. DECISION: if multilingual/RAG acceptance is <= 60% of code/math acceptance -> file a corpus item for round-5 drafter training (add multilingual + rare-token text to the training mix — cheap to include, needs user sign-off with the rest of round-5) and/or evaluate a larger draft vocab (48-64k) as an arm; if roughly uniform -> record and close (and cite as evidence our draft vocab is adequately sized).
RECORD: per-class table + histograms as a comment here + campaign log; cross-link drafter-checkpoint-ab-protocol (the round-5 verdict must include these classes).
