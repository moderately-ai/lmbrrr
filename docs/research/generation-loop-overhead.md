# Generation Loop Overhead

Date: 2026-07-07

Ticket: `optimize-generation-loop-overhead`

## Objective

Separate model throughput from generation-loop overhead so token-rate
measurements can distinguish hardware/model speed from sampling, input tensor
creation, callbacks, and bookkeeping.

The previous benchmark measured `decode_seconds` as end-to-end loop time, but
prefill and decode model forwards were not explicitly synchronized. On Metal,
deferred execution can make model forward timing appear in the later argmax or
scalar-transfer step.

## Implementation

The shared generation loop now:

- synchronizes after prefill and one-token decode forwards;
- reports synchronized `decode_model_seconds` and
  `decode_model_tokens_per_second`;
- reports non-model decode timing buckets for sampling, one-token input tensor
  creation, callbacks, and residual bookkeeping;
- uses a greedy argmax fast path when `temperature <= 0.0`, avoiding Candle's
  generic `LogitsProcessor` F32 cast for deterministic generation;
- preserves Candle `LogitsProcessor` behavior for non-greedy sampling.

The first generated token is sampled from prefill logits, so
`decode_model_input_tokens` is normally `generated_tokens - 1`.

## Measurement

Command:

```sh
cargo run --release --features metal -- bench --profile long --max-new-tokens 64 --warmup 2 --iterations 5 --output artifacts/minicpm-v46-metal-long-bench-generation-loop-timing-64x5.jsonl
```

Comparison artifact from the prior DeltaNet run:

- `artifacts/minicpm-v46-metal-long-bench-after-deltanet-no-conv-shortcut-64x5.jsonl`

New artifact:

- `artifacts/minicpm-v46-metal-long-bench-generation-loop-timing-64x5.jsonl`

Median comparison:

| Metric | Prior | New | Notes |
| --- | ---: | ---: | --- |
| Output rate | `62.97 tok/s` | `66.10 tok/s` | Greedy fast path avoids generic F32 sampling work. |
| Steady-state rate | `65.23 tok/s` | `65.09 tok/s` | Essentially unchanged; this is the best user-visible decode-rate comparator. |
| Prefill rate | `171.31 tok/s` | `158.48 tok/s` | New value is synchronized and more honest. |
| Time to first token | `0.646s` | `0.644s` | No material change. |
| Decode model rate | unavailable | `66.00 tok/s` | New synchronized model-forward metric. |
| Decode non-model share | unavailable | `1.44%` | Sampling/input/callback/bookkeeping are small in greedy benchmark mode. |

Median non-model decode buckets:

| Bucket | Median |
| --- | ---: |
| Sampling | `0.0135s` |
| One-token input tensors | `0.00044s` |
| Callback | `0.000003s` |
| Residual bookkeeping | `0.000030s` |

## Interpretation

The previous prefill token rate was optimistic because some Metal work could be
deferred past the prefill timer. The new prefill metric is lower but better for
hardware comparisons.

For decode, the useful distinction is:

- `output_tokens_per_second`: user-visible end-to-end generation rate.
- `steady_state_tokens_per_second`: user-visible rate excluding first-token
  latency.
- `decode_model_tokens_per_second`: synchronized one-token model-forward rate.
- `decode_non_model_share`: how much room remains in Rust loop/sampling code.

After this change, the greedy benchmark shows only about `1.4%` non-model decode
overhead. That means further large speedups should come from model kernels,
cache layout, quantization, or speculative decoding rather than Rust loop
bookkeeping.

