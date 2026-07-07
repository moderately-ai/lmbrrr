# EAGLE Chain Drafter Prototype

Date: 2026-07-07

Ticket: `prototype-eagle-chain-drafter`

## Scope

This is an offline EAGLE-style chain probe over exported target traces. It is
not a trained drafter and it is not a speedup claim.

The command consumes `lmbrrr trace` JSON, treats each step's target top-1 token
as an oracle direct-token chain draft, and verifies accepted-prefix accounting
against the trace's greedy generated token ids. It also reports hidden-feature
metadata for the captured layers so we can validate the feature alignment that a
real EAGLE drafter will train on.

## Commands

Generate a trace:

```sh
cargo run --release --features metal -- trace \
  --prompt "Answer in one sentence: what is 17 * 23?" \
  --capture-layer 0,11,23 \
  --max-new-tokens 6 \
  --top-k-logits 5 \
  --output target/eagle-chain-trace-smoke.json
```

Run the chain probe:

```sh
cargo run --release --features metal -- eagle-chain-draft \
  --trace target/eagle-chain-trace-smoke.json \
  --draft-width 4 \
  --output target/eagle-chain-draft-smoke.json
```

## Smoke Result

For the local MiniCPM-V-4.6 Metal smoke:

- capture layers: `[0, 11, 23]`
- draft width: `4`
- accepted tokens: `4`
- accepted length with bonus token: `5`
- exact greedy prefix match: `true`
- first feature step had `3` hidden states, hidden size `1024`, context position
  `26`, and fused feature L2 norm about `8.44`.

## Interpretation

This validates:

- trace feature alignment: feature at context position `t` predicts token
  `t + 1`;
- chain accepted-prefix accounting;
- exact greedy reconstruction for an oracle direct-token drafter;
- the JSON shape needed by a future trainer or small Candle draft head.

It does not validate draft-model quality or latency. A real EAGLE implementation
still needs a trained direct-token head over fused low/mid/high hidden states.
