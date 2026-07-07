# Hidden-State Trace Export

Date: 2026-07-07

Ticket: `record-hidden-state-traces`

## Purpose

`lmbrrr trace` records the target-model features needed to train or evaluate an
EAGLE-style drafter. It runs greedy text-only generation and exports prompt ids,
generated ids, selected Qwen3.5 layer hidden states, top-k logits, and per-step
timing.

This is not a speedup path by itself. It is the data collection path that lets us
measure which layers are useful draft features before adding a trained drafter.

## Command Shape

Default low/mid/high layers:

```sh
cargo run --release --features metal -- trace \
  --prompt "Answer in one sentence: what is 17 * 23?" \
  --max-new-tokens 4 \
  --top-k-logits 5 \
  --output target/hidden-state-trace.json
```

Explicit layers:

```sh
cargo run --release --features metal -- trace \
  --prompt "Answer in one sentence: what is 17 * 23?" \
  --capture-layer 0,11,23 \
  --max-new-tokens 8 \
  --top-k-logits 8 \
  --output target/hidden-state-trace.json
```

If no `--capture-layer` is supplied, the runner records layers `0`, middle, and
last. For MiniCPM-V-4.6's 24-layer Qwen3.5 text backbone, that defaults to
`[0, 11, 23]`.

## Trace Semantics

Each step represents the target model state used to predict one generated token:

- Step 0 runs prompt prefill. Hidden states are captured from the last prompt
  position and the target token is the first generated token.
- Later steps run one-token cached decode. Hidden states are captured from the
  generated token just consumed, and the target token is the next generated
  token.
- Hidden states are last-position vectors only, exported as F32 arrays.
- Top-k logits are exported for diagnostics and acceptance analysis, not as a
  full training target.

This aligns the feature vector with a direct-token drafter: selected hidden
states at context position `t` should predict token `t + 1`.

## Report Fields

The JSON report includes:

- `prompt_token_ids`, `generated_token_ids`, and decoded `generated_text`.
- `capture_layers` and `top_k_logits`.
- Aggregate timing for model forward, argmax, and top-k extraction.
- `steps[]`, each with `context_position`, `target_token_id`, `top_logits`, and
  `hidden_states[]`.
- Each hidden-state record includes `layer_index`, `layer_kind`, `offset`,
  `seq_len`, `position`, `hidden_size`, original tensor `dtype`, and F32
  `values`.

## Current Limits

- Text only. Vision/video traces should wait until the text drafter pipeline is
  useful.
- Greedy only. Sampling traces need additional RNG and acceptance metadata.
- JSON is intentionally simple but not storage-efficient. If we generate large
  datasets, move values to `.safetensors` or Arrow and keep JSON as metadata.

## Local Smoke Result

The local MiniCPM-V-4.6 Metal smoke run used `--capture-layer 0,11,23`,
`--max-new-tokens 3`, and `--top-k-logits 5`. The exported trace had:

- 3 generated tokens and 3 trace steps.
- 3 hidden-state records per step.
- Hidden size 1024 for every captured vector.
- Context positions 26, 27, and 28.
- 5 top logits per step.

The report was written to `target/hidden-state-trace-smoke.json`.
