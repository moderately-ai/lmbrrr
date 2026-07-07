# EAGLE Draft Runner Probe

Date: 2026-07-07

Ticket: `integrate-eagle-draft-runner`

## Scope

This adds a Rust live probe for the exported EAGLE draft-head artifact. The
probe runs the target MiniCPM/Qwen model greedily, captures the draft head's
configured hidden-state layers, calls the draft head at each generation step,
and reports accepted-prefix accounting against the target greedy tokens.

This is still a probe, not an accelerated EAGLE decode. The draft head consumes
target hidden states that have already been computed, so it measures proposal
quality and draft-head overhead before we add a true recurrent drafter path.

## Command

Train an all-samples overfit artifact for reconstruction smoke testing:

```sh
uv run python evals/eagle/train_eagle_draft_head.py \
  --trace target/eagle/traces/math.json \
  --trace target/eagle/traces/capital.json \
  --trace target/eagle/traces/colors.json \
  --trace target/eagle/traces/balls.json \
  --output-dir target/eagle/draft-head-overfit-smoke \
  --epochs 300 \
  --hidden-dim 128 \
  --draft-width 4 \
  --eval-fraction 0
```

Run the live probe:

```sh
cargo run --release --features metal -- eagle-live-probe \
  --draft-head-manifest target/eagle/draft-head-overfit-smoke/manifest.json \
  --prompt "Answer in one sentence: what is 17 * 23?" \
  --max-new-tokens 8 \
  --draft-width 4 \
  --output target/eagle/live-probe-math.json
```

## Smoke Result

For the local MiniCPM-V-4.6 Metal smoke:

- generated text: `17 * 23 is `
- scheduled draft width: `4`
- accepted tokens: `4`
- accepted length with bonus token: `5`
- exact greedy prefix match: `true`
- draft head time: `0.001901084s`
- target forward time: `0.416607833s`
- draft-head overhead share versus target forward: `0.00456`

All 8 live draft-head proposals matched the target tokens for the overfit math
prompt. The first scheduled chain reconstructed the target prefix `17 * 2`.

## Interpretation

The useful result is that Rust can now load the Python-exported draft-head
manifest and safetensors weights, run the MLP over live captured hidden states,
and produce verifier-compatible accounting.

The missing piece is still the actual speedup path: the current head predicts
from target hidden states after the target model has already paid for them. To
turn this into acceleration, the next implementation needs a drafter recurrence
or feature-prediction path that can propose multiple future tokens without
running the full target layer stack for each token.
