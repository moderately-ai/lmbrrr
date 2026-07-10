---
id: add-minicpm-dspark-evaluator
title: Add MiniCPM DSpark evaluator to DeepSpec
status: todo
priority: p1
dependencies: []
related: [scale-dspark-training-corpus-modal, calibrate-dspark-confidence-head, train-dspark-semi-autoregressive-drafter]
scopes: [evals]
shared_scopes: [docs/research]
paths: [evals/dspark/**, docs/research/dspark-semi-autoregressive-training.md]
tags: [dspark, training, modal, evals]
---
## Goal

A MiniCPM-V-4.6 evaluator in DeepSpec (branch minicpm-v46) so tau and confidence-calibration metrics run on Modal right after each training round, instead of waiting for the Rust runner.

## Context (from the full DeepSpec read)

eval.py routes on draft config `architectures[0]` and Qwen3DSparkEvaluator.build_models loads the target with AutoModelForCausalLM, which cannot load a MiniCPM-V checkpoint. The confidence-head recorder (ECE/AUROC/Brier + reliability diagrams, collected only at confidence-threshold 0.0) lives in this evaluator — it is the data source for the STS fit in calibrate-dspark-confidence-head, and the tau-vs-corpus curves in scale-dspark-training-corpus-modal need per-round eval. The generation loop itself (base_evaluator) is target-agnostic once the model loads and returns rank-3 logits with output_hidden_states.

## Acceptance

- MiniCPMDSparkEvaluator subclassing Qwen3DSparkEvaluator with build_models loading the target via AutoModelForImageTextToText (bf16, sdpa) and asserting no final capture layer; registered in eval.py EVALUATORS under a distinct architectures key or config dispatch.
- An `evaluate` Modal function in evals/dspark/modal_app.py running eval.py against a checkpoint on the volume with a small task list (gsm8k + mt-bench subsets first; eval_datasets shipped with the repo).
- One successful eval run on the smoke checkpoint reporting tau, accept_rate@k, and the confidence reliability artifacts to the volume.
