# DFlash Block Drafter Design

Date: 2026-07-07

Ticket: `design-dflash-block-drafter`

## Conclusion

DFlash should not be the next implementation after the verifier and trace
export. The next implementation should still be the EAGLE-style chain drafter,
because it exercises the same hidden-state features with a much smaller model
and gives us an accepted-length baseline.

DFlash becomes actionable once we have:

- a trace dataset from `lmbrrr trace`;
- a trained-drafter path, likely Python first;
- verifier measurements for block widths `4`, `8`, and `16`;
- an EAGLE chain baseline measured at the same target verification budget.

## Paper Details To Preserve

DFlash uses a lightweight block-diffusion drafter for speculative decoding. The
important implementation details for this repo are:

- A draft block is generated in one forward pass rather than token by token.
- The target model provides hidden context features.
- Selected target hidden states are concatenated and projected into a target
  context feature:

```text
Ht = RMSNorm(Wc [H_l1 ; ... ; H_lk])
```

- Target context is injected into every draft layer as KV entries. Draft tokens
  produce queries; both target context and draft tokens produce keys and values:

```text
Qi = WiQ Hd
Ki = [WiK Ht ; WiK Hd]_seq
Vi = [WiV Ht ; WiV Hd]_seq
```

- The target context bypasses the draft layer's query projection, output
  projection, and FFN. It only conditions attention through K/V.
- Training samples random clean anchor tokens and masks the following block
  positions. The model predicts the masked future positions in parallel.
- Earlier block positions get higher loss weight because the first wrong token
  rejects the remaining suffix.

Sources: `docs/research/papers/dflash-2602.06036.pdf`, especially Sections 4.1,
4.2, 5.5, and Appendix A.3.

## Naming For This Repo

Use `draft_width` to mean the number of proposed future tokens we send to the
verifier. Training examples contain one clean anchor token plus `draft_width`
future labels. This avoids confusion with the paper's anchor-inclusive block
construction.

Initial widths:

| `draft_width` | Purpose |
| --- | --- |
| 4 | Plumbing, fast iteration, low verifier waste risk. |
| 8 | First serious comparison against EAGLE chain drafting. |
| 16 | Larger-block target once width 8 has acceptable latency and acceptance. |

Train larger-width models before smaller-width inference if we want dynamic
width later. The paper reports that larger block-size training generalizes down
to smaller inference block sizes better than the reverse.

## Target Features

Start with the trace exporter's default low/mid/high layers:

```text
[0, 11, 23] for MiniCPM-V-4.6's 24-layer Qwen3.5 text stack
```

Then test five layers, matching DFlash's stronger setting:

```text
[0, 5, 11, 17, 23]
```

MiniCPM-V-4.6 text hidden size is `1024`. Keep the first DFlash drafter hidden
size at `1024` so the draft output head can be aligned with the target token
embedding/LM-head shape. A smaller hidden size can be tested later, but it adds
another projection and makes quality regressions harder to interpret.

## Draft Model Shape

Minimum useful model:

- token embedding for anchor and mask/input tokens;
- learned position embeddings for block positions `0..draft_width`;
- feature projection `Wc: k * 1024 -> 1024` plus RMSNorm;
- `1`, then `3`, then `5` draft transformer layers;
- each layer has KV injection from `Ht` plus bidirectional attention among draft
  block positions;
- LM head over every masked future position.

Do not start with input-only feature fusion as the main path. It is useful as an
ablation, but DFlash's key claim is that KV injection scales better as draft
depth increases.

## Training Data Layout

Generate target traces with `lmbrrr trace` over the prompt matrix from
`docs/research/speculative-decoding-lab.md`.

For each generated sequence and sampled anchor position `a`, store:

- `sample_id`
- `prompt_token_ids`
- `generated_token_ids`
- `anchor_position`
- `anchor_token_id = token[a]`
- `labels = token[a + 1 .. a + 1 + draft_width]`
- `feature_layers`
- `features`: `[feature_layers, 1024]` F32 or BF16
- `draft_width`
- loss weights `exp(-(k - 1) / gamma)` for label position `k`

Start with JSON metadata plus `.safetensors` feature blocks once the dataset is
larger than smoke tests. Do not keep large feature datasets as JSON.

Training should happen outside Candle first. A Python/PyTorch trainer can read
the trace exports, train the drafter, then export a safetensors checkpoint for
Candle inference.

## Verifier Interface

The current `spec-verify` command already verifies explicit draft token ids in a
target chunk. A real DFlash runner should factor that logic into a reusable
runtime function:

```text
verify_block(target_state, draft_token_ids) -> VerificationReport
```

The report must include:

- `draft_width`
- `accepted_tokens`
- `bonus_token_id`
- `accepted_length`
- `verify_seconds`
- `verifier_waste_tokens`
- exact reconstructed token ids

Correctness gate: speculative output ids must match baseline greedy output ids
for the same prompt, token cap, dtype, and device.

## Metal Measurements Required

Before implementing a DFlash drafter in Candle, measure these on Apple Metal:

- target one-token decode latency from `bench`;
- target chunk verification latency for widths `4`, `8`, and `16`;
- feature-fusion projection latency for 3 and 5 target layers;
- draft forward latency for `1`, `3`, and `5` draft layers at each width;
- end-to-end cycle time:

```text
cycle_seconds = draft_seconds + verify_seconds + sampling_and_bookkeeping_seconds
```

Break-even condition:

```text
cycle_seconds < accepted_length * baseline_decode_seconds_per_token
```

Do not claim a DFlash speedup from accepted length alone. On local Apple Silicon,
block parallelism can lose to memory traffic or small-kernel overhead even if the
paper's B200/FA4 results look strong.

## Implementation Sequence

1. Use trace labels as an oracle block drafter to validate block verifier metrics.
2. Train a tiny width-4 KV-injected drafter in Python on a small trace dataset.
3. Export drafter weights to safetensors.
4. Implement Candle inference for the draft module.
5. Measure width-4 end-to-end speedup against greedy baseline and EAGLE chain.
6. Scale to width 8, then width 16 only after width 4 is correct and measured.
7. Add input-fusion and no-target-feature ablations only after the KV path works.

## Pivot Gates

Proceed from design to implementation only when:

- `prototype-eagle-chain-drafter` has produced accepted-length and draft-latency
  numbers;
- target chunk verification timings are available for widths `4`, `8`, and
  `16`;
- a trace dataset can be generated reproducibly;
- we have a concrete training script location and output checkpoint format.

Stop or defer DFlash if:

- width-4 draft latency is greater than one target decode step;
- target chunk verification is not faster than verifying the same tokens
  sequentially;
- accepted length is below the EAGLE chain drafter at equal verification budget;
- JSON trace storage becomes the bottleneck before moving feature values to a
  binary format.
