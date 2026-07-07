# Greedy Speculative Verifier Harness

Date: 2026-07-07

Ticket: `implement-greedy-spec-verifier`

## Purpose

`lmbrrr spec-verify` is the first runtime tool for speculative decoding work. It
does not implement a speedup by itself. It verifies a proposed text-only draft
token block against the MiniCPM-V-4.6 target model in greedy mode and reports the
accepted prefix, bonus token, verifier waste, and synchronized timing.

## Command Shape

Explicit draft tokens:

```sh
cargo run --release --features metal -- spec-verify \
  --prompt "Answer in one sentence: what is 17 * 23?" \
  --draft-token 123,456,789 \
  --output target/spec-verify-explicit.json
```

Self-generated baseline draft:

```sh
cargo run --release --features metal -- spec-verify \
  --prompt "Answer in one sentence: what is 17 * 23?" \
  --baseline-draft-tokens 4 \
  --output target/spec-verify-baseline.json
```

Intentional rejection:

```sh
cargo run --release --features metal -- spec-verify \
  --prompt "Answer in one sentence: what is 17 * 23?" \
  --baseline-draft-tokens 4 \
  --corrupt-draft-at 2 \
  --output target/spec-verify-corrupt.json
```

`--baseline-draft-tokens N` first generates `N + 1` greedy target tokens, uses
the first `N` as the proposed draft, then checks that the verifier reconstruction
matches the baseline prefix. `--corrupt-draft-at I` mutates one proposed token so
we can prove first-rejection accounting without needing an external draft model.

## Verification Flow

1. Tokenize the MiniCPM chat prompt.
2. Prefill the target model on the prompt and take the argmax as the target for
   draft position 0.
3. Run one cached `forward_all_logits` call over the whole draft block.
4. Use chunk logits `0..N-2` as targets for draft positions `1..N-1`.
5. Use the final chunk logit as the bonus token when every draft token is
   accepted.
6. If a draft token mismatches, emit the target token at that position as the
   bonus token and treat later verified positions as verifier waste.

This is intentionally closer to real speculative verification than a purely
sequential checker: after prompt prefill, the draft block is verified in one
target chunk, so the report can measure wasted verifier work after a rejected
position.

## Report Fields

The JSON report includes:

- `draft_token_ids`, `target_token_ids`, `accepted_token_ids`, and
  `reconstructed_token_ids`.
- `accepted_tokens`, `bonus_tokens`, `accepted_length`, `acceptance_rate`.
- `first_rejected_index`, `verifier_waste_tokens`, `verifier_waste_share`.
- `prefill_seconds`, `verify_seconds`, `argmax_seconds`, and
  `verify_tokens_per_second`.
- Per-position token ids, decoded token strings, match status, accepted status,
  and first-rejection marker.

## Current Limits

- Text only. Image and video cache verification can come later once text
  speculative mechanics are stable.
- Greedy only. Sampling acceptance rules are deliberately excluded from this
  first verifier.
- The baseline-draft mode is a correctness harness, not a performance result.
  It spends target-model work to create the draft and must not be counted as a
  speculative speedup.

## Local Smoke Results

On the local MiniCPM-V-4.6 Metal path:

- Baseline draft with `--baseline-draft-tokens 4` accepted all 4 proposed tokens,
  emitted a 1-token bonus, reconstructed 5 baseline tokens, and reported
  `baseline_prefix_match: true`.
- Corrupted draft with `--baseline-draft-tokens 4 --corrupt-draft-at 2` accepted
  2 proposed tokens, rejected at index 2, emitted the target replacement token,
  reported 1 wasted verified suffix token, and matched the expected rejection
  index.

The reports were written to `target/spec-verify-baseline.json` and
`target/spec-verify-corrupt.json`.
