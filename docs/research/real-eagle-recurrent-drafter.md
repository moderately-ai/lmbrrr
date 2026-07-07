# Real EAGLE Recurrent Drafter

Date: 2026-07-07

Ticket: `implement-real-eagle-recurrent-drafter`

## Scope

This ticket moves beyond the earlier `eagle-live-probe`. The live probe scored a
draft head after target hidden states had already been computed. The new
`eagle-recurrent-draft` path drafts before target verification:

1. Run the target model once on the prompt and capture the configured anchor
   hidden-state layers.
2. Use a small recurrent MLP drafter to propose a block from:
   - anchor hidden feature;
   - previous drafted token as one-hot state;
   - normalized draft position.
3. Verify the proposed block in one target chunk.
4. Compare the reconstructed speculative output against greedy target output
   for the same accepted-length window.

This is a real speculative runner shape because future draft tokens are proposed
without first computing target hidden states for those future positions.

## Artifact

Trainer:

```sh
uv run python evals/eagle/train_eagle_recurrent_drafter.py \
  --trace target/eagle/traces/math.json \
  --trace target/eagle/traces/capital.json \
  --trace target/eagle/traces/colors.json \
  --trace target/eagle/traces/balls.json \
  --output-dir target/eagle/recurrent-drafter-overfit-smoke \
  --epochs 500 \
  --hidden-dim 192 \
  --draft-width 4 \
  --eval-fraction 0
```

Training smoke:

| Metric | Value |
| --- | ---: |
| Samples | `100` |
| Capture layers | `[0, 11, 23]` |
| Feature dim | `3072` |
| Input dim | `3097` |
| Hidden dim | `192` |
| Target vocab | `26` |
| Previous-token vocab | `24` |
| Train top-1 | `1.00` |
| Train top-5 | `1.00` |
| Mean anchor acceptance | `3.23 / 4` |

The output vocabulary is intentionally restricted to observed target tokens in
the trace set. Previous-token state uses the observed previous-token vocabulary;
the Rust runner falls back to an all-zero previous-token state when a prompt
ends in an unseen token.

## Runner Commands

Math smoke:

```sh
cargo run --release --features metal -- eagle-recurrent-draft \
  --drafter-manifest target/eagle/recurrent-drafter-overfit-smoke/manifest.json \
  --prompt "Answer in one sentence: what is 17 * 23?" \
  --draft-width 4 \
  --output target/eagle/recurrent-draft-math.json
```

Capital smoke:

```sh
cargo run --release --features metal -- eagle-recurrent-draft \
  --drafter-manifest target/eagle/recurrent-drafter-overfit-smoke/manifest.json \
  --prompt "Answer in one sentence: what is the capital of France?" \
  --draft-width 4 \
  --output target/eagle/recurrent-draft-capital.json
```

## Smoke Results

| Prompt | Accepted Tokens | Accepted Length | Calls Saved Estimate | Exact Greedy Prefix | Draft Text | Reconstructed Text | Model-Time Speedup Estimate |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| Math | `4` | `5` | `4` | `true` | `17 * ` | `17 * 2` | `1.32x` |
| Capital | `4` | `5` | `4` | `true` | `The capital of France` | `The capital of France is` | `1.02x` |

The drafter overhead was small in both runs:

| Prompt | Draft Seconds | Verify Seconds | Speculative Model Seconds | Baseline Model Seconds |
| --- | ---: | ---: | ---: | ---: |
| Math | `0.0044` | `0.0719` | `0.6306` | `0.8312` |
| Capital | `0.0062` | `0.0550` | `0.7387` | `0.7572` |

## Interpretation

This is the first implementation in the repo where an EAGLE-style component
proposes multiple future tokens before target verification. It validates the
runner contract we need for actual acceleration:

- one target anchor forward;
- multiple drafter predictions;
- one target verification chunk;
- exact greedy reconstruction check;
- target-call savings and speed estimate in the report.

The result is still a smoke. It overfits four short traces and uses an
observed-vocabulary output head. The speed estimate is also a short-window
measurement, so it is useful for direction but not a production claim.

## Next Work

The next EAGLE depth should be training quality, not more runner plumbing:

- expand traces beyond the four smoke prompts;
- add a held-out eval split that preserves full prompt sequences;
- replace observed-token output with a larger candidate vocabulary, likely
  top-k target tokens collected from traces;
- measure multi-cycle generation after accepting a block and using the verifier
  bonus token as the next anchor;
- compare against the DSpark confidence scheduler once drafter confidence is
  calibrated.

