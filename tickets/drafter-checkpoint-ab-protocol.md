---
id: drafter-checkpoint-ab-protocol
title: "EVAL INFRA: drafter-checkpoint A/B protocol (pre-round-5 prep, no training needed)"
status: todo
priority: p3
dependencies: []
related: []
scopes: [evals]
shared_scopes: []
paths: []
tags: [eval-wave, dspark]
---
WHY: round-5 drafter training is user-gated (see round2-prep directive + dspark-cache-redesign-beyond-400k), but the moment ANY new checkpoint exists, someone must judge it — and the comparison protocol currently lives only in past-session muscle memory. Codify it now so a checkpoint drop is a same-day verdict.

PROTOCOL (write into evals/README-drafter-ab.md + validate once by A/B-ing round-4 against itself — the null test must come out flat):
1. SAME BINARY for both arms; drafters differ only via --drafter path. Never compare across binaries or pins.
2. Suite: `python3 evals/run_spec_suite.py --arm cand=target/<candidate> --arm base=artifacts/dspark-drafter-round4 --split validation --per-class 3 --reps 3 --gamma 6 --tag drafter-ab-<name>` (arms rotate automatically; run controls first per the measurement-protocol memory; quiet machine; DIVERGENT-CONTENT questions discounted from means — with different drafters MOST content will diverge, which is fine and expected here: tok/s comparison is via per-class MEANS across the 3 questions, and the null test establishes the noise band for exactly this situation).
3. METRICS, in decision order: (a) per-class + overall mean tokens/s; (b) acceptance: mean accepted length per drafted round, fraction of rounds drafted (from report fields; the scheduler adapts width, so acceptance shifts show up partly as width shifts — report both); (c) tau (mean accepted per proposed) per class; (d) STS calibration check — if the candidate's confidence head differs, the Platt calibration MUST be refit (evals/fit_sts.py) before the suite run, else the scheduler runs miscalibrated and the A/B is unfair to the candidate.
4. COST MODEL: if candidate propose cost differs structurally (layers/width), run the in-loop refit for the candidate arm before judging (calibration split, --reps 1), else the scheduler prices rounds wrong.
5. VERDICT TEMPLATE: ship iff overall mean >= +2% with no class regressing > 2%, acceptance gains explainable (tau up or width up, not content luck), null-test band respected. Record table + verdict on the training ticket.
DONE-WHEN: doc committed + the round4-vs-round4 null run posted here showing per-class deltas inside the noise band (this null band number is itself a standing asset — cite it in every future drafter verdict).
