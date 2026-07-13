# DSpark Confidence Scheduler Prototype

Date: 2026-07-07

Ticket: `prototype-dspark-confidence-scheduler`

## Scope

This is the verifier-side scheduler harness, not a trained DSpark drafter. The
runner can now accept per-position draft confidences, select a scheduled prefix
before target verification, and report the resulting accepted length and verifier
waste.

The useful command path is:

```sh
cargo run --release --features metal -- spec-verify \
  --prompt "Answer in one sentence: what is 17 * 23?" \
  --baseline-draft-tokens 8 \
  --draft-confidence 0.98,0.96,0.93,0.88,0.72,0.65,0.60,0.55 \
  --schedule-confidence-threshold 0.70 \
  --output artifacts/dspark-confidence-scheduler-smoke.json
```

The scheduler multiplies confidences left to right and keeps the longest prefix
whose cumulative confidence remains above the threshold. The first token that
would drop below the threshold, and all suffix tokens after it, are not sent to
the verifier.

## Smoke Result

For the command above:

- original draft tokens: `8`
- scheduled draft tokens: `4`
- dropped draft tokens: `4`
- scheduled cumulative confidence: `0.7699507200000001`
- next rejected cumulative confidence: `0.5543645184`
- accepted tokens: `4`
- accepted length including target bonus token: `5`
- verifier waste tokens: `0`
- baseline greedy prefix match: `true`

## Why This Matters

This gives us the accounting we need before training a DSpark-style confidence
head:

- scheduled draft length;
- confidence threshold and cumulative confidence;
- accepted length;
- verifier waste;
- exact greedy preservation.

Once an EAGLE or DFlash drafter emits real per-position confidence scores, the
same verifier path can measure whether confidence scheduling reduces verifier
waste without changing greedy output.

## Current Limits

The confidences are supplied explicitly. They are not calibrated model outputs
yet. This means the harness can validate scheduler math and reporting, but it
cannot prove DSpark quality or speedup until a drafter produces confidence
scores alongside draft tokens.
