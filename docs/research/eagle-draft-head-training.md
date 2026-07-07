# EAGLE Draft Head Training

Date: 2026-07-07

Ticket: `train-eagle-draft-head-from-traces`

## Scope

This ticket adds the first train/export path for an EAGLE-style direct-token
draft head. It is a training-pipeline smoke, not a speedup claim.

The trainer consumes `lmbrrr trace` JSON, concatenates the selected hidden-state
vectors in layer order, standardizes the fused feature vector, and trains a
small MLP classifier over the token ids observed in the trace set.

## Commands

Generate traces:

```sh
cargo run --release --features metal -- trace \
  --prompt "Answer in one sentence: what is 17 * 23?" \
  --capture-layer 0,11,23 \
  --max-new-tokens 8 \
  --top-k-logits 5 \
  --output target/eagle/traces/math.json
```

The local smoke used four traces:

- `math.json`: `Answer in one sentence: what is 17 * 23?`
- `capital.json`: `Answer in one sentence: what is the capital of France?`
- `colors.json`: `Name three colors in a traffic light.`
- `balls.json`: `A box has 3 red balls and 2 blue balls. If one red ball is
  added, how many balls are there?`

Train and export:

```sh
uv run python evals/eagle/train_eagle_draft_head.py \
  --trace target/eagle/traces/math.json \
  --trace target/eagle/traces/capital.json \
  --trace target/eagle/traces/colors.json \
  --trace target/eagle/traces/balls.json \
  --output-dir target/eagle/draft-head-smoke \
  --epochs 300 \
  --hidden-dim 128 \
  --draft-width 4
```

The output directory contains:

- `manifest.json`: architecture, capture layers, token vocabulary, dataset
  metadata, and offline metrics;
- `weights.safetensors`: feature mean/std and MLP weights.

## Smoke Result

The local smoke artifact was written to `target/eagle/draft-head-smoke/`:

- `manifest.json`: `7.5 KB`
- `weights.safetensors`: `1.5 MB`

Dataset:

- traces: `4`
- samples: `31`
- capture layers: `[0, 11, 23]`
- input dimension: `3072`
- hidden dimension: `128`
- output token ids: `26`

Metrics with the deterministic random split:

| Split | Samples | Top-1 | Top-5 | Mean Accepted Prefix |
| --- | ---: | ---: | ---: | ---: |
| Train | `25` | `1.00` | `1.00` | `2.64` |
| Eval | `6` | `0.00` | `0.33` | `0.00` |

## Interpretation

The exported artifact is runner-loadable by design, but its output vocabulary is
restricted to tokens seen in the traces. This makes it useful for validating the
training/export/integration plumbing before spending time on a larger draft
model. It is not expected to generalize well yet.

The poor held-out result is expected for this tiny observed-vocabulary
classifier: many eval contexts have too little lexical overlap with the training
split, and unseen target behavior cannot be represented. The useful result is
that the trace-to-artifact pipeline works and the overfit train split can
produce multi-token accepted prefixes.

The next EAGLE ticket should load this artifact inside the runner, verify that
its direct-token proposals reconstruct greedy output on the training traces, and
then measure whether the online overhead is small enough to justify expanding
the training set and output head.
