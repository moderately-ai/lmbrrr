# Mixed Q4/Q8 Quantization Policy

Date: 2026-07-07

Ticket: `benchmark-mixed-q4-q8-quant-policy`

## Scope

This ticket adds and evaluates `q4k-mlp-q8-text`, a mixed MiniCPM-V-4.6 text
policy:

- text MLP weights use `q4k_block64_symmetric`;
- text full-attention and DeltaNet weights use `q8_symmetric`;
- protected tensors and non-text weights preserve source precision.

The intent was to keep more of the sensitive recurrent/attention path at q8
while preserving the q4 MLP speed/memory gain.

## Commands

Generate the artifact:

```sh
cargo run --release --features metal -- quant-convert \
  --policy q4k-mlp-q8-text \
  --output-dir artifacts/minicpm-v46-q4k-mlp-q8-text-full
```

Benchmark with the same shape as the prior dense/q8/q4 runs:

```sh
cargo run --release --features metal -- bench \
  --quantized-manifest artifacts/minicpm-v46-q4k-mlp-q8-text-full/manifest.json \
  --profile short \
  --profile medium \
  --profile long \
  --max-new-tokens 32 \
  --warmup 1 \
  --iterations 2 \
  --output artifacts/minicpm-v46-real-quant-q4k-mlp-q8-text-bench.jsonl
```

Run generation quality gates with the mixed policy included:

```sh
cargo run --release --features metal -- quant-quality \
  --max-new-tokens 64 \
  --output artifacts/minicpm-v46-q4-quality-with-mixed.json
```

## Artifact

| Metric | Value |
| --- | ---: |
| Quantized tensors | `186` |
| Q4K tensors | `72` |
| Q8 tensors | `114` |
| Preserved tensors | `593` |
| Quantized data bytes | `382,009,800` |
| Dense-equivalent bytes | `995,229,696` |
| Approx dense bytes avoided | `613,219,896` |

## Speed

Average decode tokens/sec:

| Profile | Dense | Q8 Text | Q4K MLP | Q4K Text-Safe | Mixed Q4/Q8 | Mixed Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | `65.68` | `66.63` | `68.65` | `67.98` | `67.44` | `1.027x` |
| Medium | `67.29` | `68.72` | `69.89` | `70.16` | `69.55` | `1.034x` |
| Long | `66.83` | `68.02` | `69.26` | `69.87` | `69.32` | `1.037x` |

Prefill ratios versus dense:

| Profile | Mixed Q4/Q8 |
| --- | ---: |
| Short | `1.013x` |
| Medium | `1.018x` |
| Long | `0.998x` |

The mixed policy is faster than dense and q8, but it is slightly slower than
the q4 MLP-only and q4 text-safe policies on decode.

## Quality Gate

`q4k-mlp-q8-text` did not pass the generation gate:

| Policy | Cases | Exact Matches | Passed Cases | Failed Cases | Mean Prefix | Mean Token Jaccard | Mean Lexical Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `q4k-mlp-only` | `7` | `2` | `4` | `3` | `0.504` | `0.770` | `0.736` |
| `q4k-mlp-q8-text` | `7` | `2` | `3` | `4` | `0.353` | `0.623` | `0.600` |
| `q4k-text-safe` | `7` | `2` | `4` | `3` | `0.446` | `0.706` | `0.691` |
| `q8-text-linears` | `7` | `2` | `3` | `4` | `0.415` | `0.825` | `0.814` |

Mixed-policy failures:

| Case | Pass | Divergence | Prefix | Token Jaccard | Lexical Jaccard |
| --- | --- | ---: | ---: | ---: | ---: |
| `text_short_factual_closed` | yes | n/a | `1.000` | `1.000` | `1.000` |
| `text_arithmetic_open_thinking` | no | `2` | `0.031` | `0.561` | `0.444` |
| `text_long_reasoning_closed` | no | `8` | `0.125` | `0.455` | `0.519` |
| `text_code_completion_closed` | no | `0` | `0.000` | `0.432` | `0.400` |
| `text_tool_style_closed` | no | `0` | `0.000` | `0.271` | `0.264` |
| `text_thinking_toggle_closed` | yes | n/a | `1.000` | `1.000` | `1.000` |
| `text_thinking_toggle_open` | yes | `20` | `0.312` | `0.641` | `0.571` |

## Decision

Do not make `q4k-mlp-q8-text` the default performance policy.

The policy saves more dense-equivalent bytes than q4 MLP-only because it also
compresses attention and DeltaNet weights to q8. However, it does not improve
generation quality versus q4 MLP-only, and it gives up some of the q4 decode
speed advantage. The best current interpretation is that generation drift is
not only caused by q4 attention/DeltaNet coverage; q4 MLP drift and q8 drift
both matter enough to require better calibration.

Next useful quantization work:

- add a stricter calibration policy that excludes MLP tensors involved in early
  divergence cases;
- test per-layer q4 MLP selection instead of all MLP tensors;
- keep q8 as a memory policy, but do not treat q8 text quantization as
  generation-equivalent to dense without a quality gate.

