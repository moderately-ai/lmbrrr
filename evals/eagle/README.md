# EAGLE Eval Utilities

This directory contains uv-managed utilities for EAGLE-style draft-head
experiments.

Current utility:

- `train_eagle_draft_head.py` consumes one or more `lmbrrr trace` JSON reports,
  trains a small observed-vocabulary MLP over fused hidden-state features, and
  exports a `manifest.json` plus `weights.safetensors`.

The first trainer is intentionally a plumbing and measurement artifact. Its
output vocabulary is limited to token ids observed in the trace set, so it is
not yet a production online drafter.
