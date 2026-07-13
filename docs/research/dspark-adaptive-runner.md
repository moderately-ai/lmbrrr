# DSpark Adaptive Runner

Date: 2026-07-07

Ticket: `integrate-dspark-adaptive-scheduler`

## Scope

`eagle-live-probe` now accepts a DSpark-style confidence threshold:

```sh
--schedule-confidence-threshold <0..1>
```

The live EAGLE probe gets per-token confidence from the draft head, applies the
same cumulative-confidence scheduler used by `spec-verify`, truncates the draft
prefix before verification accounting, and reports the schedule alongside
accepted length and estimated target-call savings.

This is still a live probe, not accelerated speculative decoding. The draft head
runs on target hidden states that have already been computed. The useful result
is that the online runner path now carries confidence values through scheduling
and verifier-compatible accounting.

## Commands

Fixed width:

```sh
cargo run --release --features metal -- eagle-live-probe \
  --draft-head-manifest target/eagle/draft-head-overfit-smoke/manifest.json \
  --prompt "Answer in one sentence: what is 17 * 23?" \
  --max-new-tokens 8 \
  --draft-width 4 \
  --output artifacts/dspark-live-fixed-width.json
```

Adaptive schedule:

```sh
cargo run --release --features metal -- eagle-live-probe \
  --draft-head-manifest target/eagle/draft-head-overfit-smoke/manifest.json \
  --prompt "Answer in one sentence: what is 17 * 23?" \
  --max-new-tokens 8 \
  --draft-width 4 \
  --schedule-confidence-threshold 0.9999999999 \
  --output artifacts/dspark-live-adaptive.json
```

The threshold is intentionally high because the overfit smoke head is extremely
confident on its training prompt. This demonstrates truncation behavior; it is
not a calibrated DSpark policy.

## Smoke Result

| Mode | Candidate Draft | Scheduled Draft | Accepted Tokens | Accepted Length | Estimated Target Calls Saved | Exact Prefix |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fixed | `4` | `4` | `4` | `5` | `4` | `true` |
| Adaptive | `4` | `3` | `3` | `4` | `3` | `true` |

Adaptive schedule details:

- threshold: `0.9999999999`
- scheduled cumulative confidence: `0.999999999993211`
- next rejected cumulative confidence: `0.9999999994444884`
- dropped draft tokens: `1`

## Interpretation

The runner now has the plumbing required for DSpark-style dynamic draft lengths:

- the drafter emits token-level confidence;
- the runner schedules a prefix before verifier accounting;
- reports include scheduled length, dropped tokens, accepted length, waste, and
  estimated target calls saved.

The next useful work is calibration. The current overfit MLP emits saturated
confidence on seen prompts, so threshold values are not meaningful. A real
DSpark policy needs confidence calibration over held-out traces and a true
multi-token drafter that can propose future tokens without target-model hidden
states.
