# Real Quantized Inference Benchmark

Date: 2026-07-07

Ticket: `benchmark-real-quantized-inference`

## Setup

The benchmark compares dense BF16 MiniCPM-V-4.6 text inference against full
runtime `QTensor` replacement policies on Metal.

Common benchmark shape:

```sh
cargo run --release --features metal -- bench \
  --profile short \
  --profile medium \
  --profile long \
  --max-new-tokens 32 \
  --warmup 1 \
  --iterations 2 \
  --output target/minicpm-v46-real-quant-<policy>-bench.jsonl
```

Artifacts:

```sh
cargo run --release --features metal -- quant-convert \
  --sensitivity target/minicpm-v46-quant-sensitivity.json \
  --policy q8-text-linears \
  --output-dir target/minicpm-v46-q8-full

cargo run --release --features metal -- quant-convert \
  --sensitivity target/minicpm-v46-quant-sensitivity.json \
  --policy q4k-mlp-only \
  --output-dir target/minicpm-v46-q4k-mlp-full

cargo run --release --features metal -- quant-convert \
  --sensitivity target/minicpm-v46-quant-sensitivity.json \
  --policy q4k-text-safe \
  --output-dir target/minicpm-v46-q4k-text-safe-full
```

## Decode Rates

Average decode tokens/sec:

| Profile | Dense | Q8 Text | Q8 Ratio | Q4K MLP | Q4K MLP Ratio | Q4K Text-Safe | Q4K Text-Safe Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | `65.68` | `66.63` | `1.014x` | `68.65` | `1.045x` | `67.98` | `1.035x` |
| Medium | `67.29` | `68.72` | `1.021x` | `69.89` | `1.039x` | `70.16` | `1.043x` |
| Long | `66.83` | `68.02` | `1.018x` | `69.26` | `1.036x` | `69.87` | `1.045x` |

Prefill was effectively flat:

| Profile | Q8 Ratio | Q4K MLP Ratio | Q4K Text-Safe Ratio |
| --- | ---: | ---: | ---: |
| Short | `0.993x` | `0.999x` | `1.006x` |
| Medium | `0.990x` | `1.006x` | `1.004x` |
| Long | `0.998x` | `0.997x` | `0.996x` |

## Memory Impact

Runtime quantized-load summaries:

| Policy | Replaced Tensors | Quantized Data | Dense Equivalent | Approx Dense Bytes Avoided |
| --- | ---: | ---: | ---: | ---: |
| Q8 Text | `186` | `497,615,592` | `995,229,696` | `497,614,104` |
| Q4K MLP | `72` | `148,635,648` | `528,482,304` | `379,846,656` |
| Q4K Text-Safe | `96` | `173,408,256` | `616,562,688` | `443,154,432` |

All three policies used `backend: candle_qtensor_requantized`.

## Output Sanity

Q8 text linears passed the existing text logits parity fixture.

Q4K MLP-only and Q4K text-safe preserved top-1 logits on all three text fixtures
but failed the stricter shared-logit-delta threshold on two short/medium cases:

| Policy | Top-1 Fixture Matches | Strict Fixture Passes |
| --- | ---: | ---: |
| Q8 Text | `3/3` | `3/3` |
| Q4K MLP | `3/3` | `1/3` |
| Q4K Text-Safe | `3/3` | `1/3` |

## Decision

Q8 text quantization is the safer memory policy, but it only improves decode by
about `1.4-2.1%` in this end-to-end benchmark. It is not the main speed lever.

Q4K MLP-only and Q4K text-safe produce a more meaningful decode gain, roughly
`3.5-4.5%`, while keeping prefill flat. The quality risk is visible in logit
drift, so the next quantization work should not broaden q4 blindly. The better
path is:

- add more output-quality checks beyond top-1 fixture parity;
- test mixed policies that keep attention/deltanet at q8 while using q4 for
  selected MLP tensors;
- profile whether the F32 activation cast around quantized `QMatMul` is now a
  measurable bottleneck in the full model.

