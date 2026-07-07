# Benchmark Runner

Date: 2026-07-07

The runner now has two output modes:

- `lmbrrr run` uses a Ratatui terminal view when stdout is interactive.
- `lmbrrr bench` writes machine-readable JSONL rows for repeatable token-rate
  comparisons.

## Interactive Run

Use `run` when inspecting model behavior:

```sh
cargo run --release --features metal -- run \
  --prompt "Solve 17 * 23. Think carefully." \
  --max-new-tokens 128
```

When stdout is a terminal, the command opens a temporary TUI with:

- a metrics panel showing prefill tokens/sec, output tokens/sec, prompt tokens,
  output tokens versus `--max-new-tokens`, and total tokens versus the run's
  max total token count;
- a `Reasoning` pane populated from `<think>...</think>` text;
- an `Answer` pane for non-reasoning output.

After generation finishes, the TUI exits and prints the final reasoning/answer
transcript normally. Use `--no-progress` for plain streamed text, or when
capturing output in scripts.

MiniCPM-V-4.6's chat template defaults to `enable_thinking=false`, which inserts
an empty closed `<think></think>` block before generation. Use
`--enable-thinking` to leave the `<think>` block open and let the model emit
visible reasoning text:

```sh
cargo run --release --features metal -- run \
  --enable-thinking \
  --prompt "Solve 17 * 23. Think carefully." \
  --max-new-tokens 256
```

## Benchmark Command

Use `bench` when comparing throughput:

```sh
cargo run --release --features metal -- bench \
  --max-new-tokens 128 \
  --warmup 1 \
  --iterations 3 \
  --output target/lmbrrr-bench.jsonl
```

By default this runs three text profiles: `short`, `medium`, and `long`.
Profiles can be selected explicitly:

```sh
cargo run --release --features metal -- bench \
  --profile long \
  --max-new-tokens 256 \
  --output target/lmbrrr-long.jsonl
```

Custom prompts can be supplied with repeated `--prompt` flags. If any custom
prompt is supplied without `--profile`, only the custom prompts run.

## JSONL Fields

Each measured iteration writes one JSON object. Warmup iterations execute but do
not write rows.

Important fields:

- `prompt_tokens`: prompt/prefill token count.
- `generated_tokens`: output tokens generated before EOS or the configured
  limit.
- `max_generated_tokens`: configured output-token cap.
- `max_total_tokens`: `prompt_tokens + max_generated_tokens`.
- `prefill_seconds`: synchronized wall time for the prefill forward pass.
- `prefill_tokens_per_second`: prompt tokens divided by prefill seconds.
- `time_to_first_token_seconds`: prefill plus first decode step.
- `decode_seconds`: output-token generation time.
- `output_tokens_per_second`: generated output tokens divided by decode seconds.
- `steady_state_tokens_per_second`: output rate excluding the first decoded
  token when available.
- `decode_model_input_tokens`: number of one-token decode forwards. This is
  usually `generated_tokens - 1` because the first output token is sampled from
  prefill logits.
- `decode_model_seconds`: synchronized wall time spent in one-token decode
  model forwards.
- `decode_model_tokens_per_second`: `decode_model_input_tokens` divided by
  `decode_model_seconds`.
- `decode_non_model_seconds`: decode wall time outside synchronized model
  forwards.
- `decode_non_model_share`: non-model decode time divided by `decode_seconds`.
- `sampling_seconds`: time spent applying repeat penalty and selecting next
  tokens.
- `next_input_seconds`: time spent creating one-token input tensors.
- `callback_seconds`: time spent in the per-token output callback.
- `decode_bookkeeping_seconds`: residual measured loop overhead not covered by
  the other timing buckets.
- `text.raw`: decoded generated text.
- `text.reasoning`: text found inside `<think>...</think>` spans.
- `text.answer`: text outside reasoning spans.
- `generation.enable_thinking`: whether the generation prompt left the thinking
  span open.

`decode_tokens_per_second` is kept as an alias for `output_tokens_per_second`.

## Comparing Runs

Keep generation settings identical between runs and compare the same profile:

```sh
jq -r '[.profile, .iteration, .prefill_tokens_per_second, .output_tokens_per_second, .steady_state_tokens_per_second] | @tsv' \
  target/lmbrrr-bench.jsonl
```

For hardware-oriented performance claims, prefer
`decode_model_tokens_per_second` and `prefill_tokens_per_second`. For
user-visible throughput, prefer `output_tokens_per_second` or
`steady_state_tokens_per_second`. Interactive TUI updates are throttled, but
any live display still adds some overhead that benchmark mode intentionally
avoids.

## Decode Profiling

Use `profile` when deciding which model path to optimize:

```sh
cargo run --release --features metal -- profile \
  --profile long \
  --max-new-tokens 32 \
  --output target/minicpm-v46-metal-decode-profile.json
```

This command runs the same benchmark profile prompts as `bench`, then profiles
single-token decode forwards with synchronized component timing. It is slower
than normal generation by design, but separates DeltaNet, full-attention, MLP,
norm, KV-cache, and argmax/scalar-transfer costs.

## Current Scope

The benchmark harness is text-only today. Multimodal benchmark profiles should
wait for the MiniCPM image parity ticket so image preprocessing and visual-token
replacement are known to match upstream behavior.
