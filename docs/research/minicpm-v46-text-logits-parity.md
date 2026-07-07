# MiniCPM-V-4.6 Text Logits Parity

Date: 2026-07-07

This note records the first Candle-vs-Transformers next-token logits check for
the text-only MiniCPM-V-4.6 path.

## Command

```sh
cargo run --features metal -- logits \
  --top-k 10 \
  --fail-on-mismatch \
  --output target/minicpm-v46-candle-logits-parity-strict.json
```

The command compares Candle output against
`evals/fixtures/minicpm_v46_transformers_text_logits.json`.

## Result

The strict Metal run passed.

| Case | Top-1 | Top-10 Overlap | Max Shared Logit Delta |
| --- | --- | --- | --- |
| `text_closed_thinking_short` | match | 9/10 | 0.25 |
| `text_open_thinking_math` | match | 9/10 | 0.25 |
| `text_closed_thinking_long_reasoning` | match | 10/10 | 0.25 |

The observed differences are small BF16-scale rank-neighbor differences in the
tail of the top-10 list. The top-1 token is stable on all three prompts.

## Interpretation

This closes the initial text logits correctness gate for text-only performance
work. It does not validate multimodal logits, image-conditioned hidden states,
or image processor pixel parity.

The command is intentionally report-oriented:

- default mode writes a JSON report and exits successfully even when a case
  fails, so mismatches can be inspected;
- `--fail-on-mismatch` turns the same comparison into a gating command;
- output records model id, revision, fixture path, dtype, device, top-k values,
  overlap counts, shared-token logit deltas, and case-level pass/fail status.

## Next Use

Use this command after changes to Qwen3.5 text decode, DeltaNet, caches,
matmuls, dtype handling, quantization, or speculative verification. Text
throughput improvements should not be treated as valid unless this gate still
passes or the mismatch is explicitly explained.
