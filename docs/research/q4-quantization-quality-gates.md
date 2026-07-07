# Q4 Quantization Quality Gates

Date: 2026-07-07

Ticket: `calibrate-q4-quantization-quality-gates`

## Scope

`lmbrrr quant-quality` compares greedy dense MiniCPM-V-4.6 generation against
runtime quantized policies on the text calibration set. This is a generation
gate, not a semantic judge. It catches drift that single-step logits parity can
miss.

The command runs these policies over the same tokenized prompts:

- dense BF16 baseline;
- `q8-text-linears`;
- `q4k-mlp-only`;
- `q4k-text-safe`.

For each candidate, the report includes:

- exact generated-token match;
- first divergence index;
- common-prefix ratio;
- generated-token multiset Jaccard;
- lexical multiset Jaccard over decoded text;
- length ratio delta;
- per-case and per-policy gate result.

The gate passes a case when the candidate exactly matches dense tokens, or all
configured coarse thresholds pass. The default thresholds are intentionally
conservative enough to catch early structural drift:

| Metric | Default |
| --- | ---: |
| Minimum prefix ratio | `0.25` |
| Minimum token Jaccard | `0.50` |
| Minimum lexical Jaccard | `0.50` |
| Maximum length ratio delta | `0.50` |

Use `--fail-on-gate` when this should fail a script or CI job.

## Commands

One-case smoke:

```sh
cargo run --release --features metal -- quant-quality \
  --max-cases 1 \
  --max-new-tokens 32 \
  --output target/minicpm-v46-q4-quality-smoke.json
```

Full text calibration matrix with a 64-token cap:

```sh
cargo run --release --features metal -- quant-quality \
  --max-new-tokens 64 \
  --output target/minicpm-v46-q4-quality-full.json
```

The default manifest paths are:

- `target/minicpm-v46-q8-full/manifest.json`
- `target/minicpm-v46-q4k-mlp-full/manifest.json`
- `target/minicpm-v46-q4k-text-safe-full/manifest.json`

## Local Result

The one-case smoke passed for every policy on `text_short_factual_closed`; all
three quantized policies exactly reproduced dense output:

```text
The capital of France is Paris.
```

The full text matrix did not pass:

| Policy | Cases | Exact Matches | Passed Cases | Failed Cases | Mean Prefix | Mean Token Jaccard | Mean Lexical Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `q8-text-linears` | `7` | `2` | `3` | `4` | `0.415` | `0.825` | `0.814` |
| `q4k-mlp-only` | `7` | `2` | `4` | `3` | `0.504` | `0.770` | `0.736` |
| `q4k-text-safe` | `7` | `2` | `4` | `3` | `0.446` | `0.706` | `0.691` |

Notable failures:

- Open-thinking arithmetic diverged early for every quantized policy.
- Long closed reasoning stayed lexically close but diverged before the default
  prefix threshold.
- `q4k-text-safe` regressed the code-completion case badly, returning a prose
  explanation instead of code.
- Tool-style output remains sensitive: q8 and q4 MLP-only diverged immediately,
  while q4 text-safe stayed closer to dense.

## Decision

Do not promote q4 as a default policy yet.

The earlier benchmark showed q4 decode gains of roughly `3.5-4.5%`, but this
gate shows that the speedup comes with visible generation drift. The next
quantization ticket should test a mixed policy rather than broadening q4:

- keep DeltaNet and attention-sensitive tensors at q8;
- keep only selected MLP tensors at q4;
- rerun this generation gate beside decode/pre-fill benchmarks;
- require all short factual and thinking-control cases to pass before treating
  the policy as usable for interactive runs.

