# Speculative Decoding Lab

Date: 2026-07-07

Ticket: `design-speculative-decoding-lab`

## Goal

Design a text-first speculative decoding lab for MiniCPM-V-4.6 on Candle/Metal.
The lab should let us measure accepted length, verifier waste, draft latency, and
true output speed before attempting DSpark- or DFlash-style model changes.

The current runner is ready for this design work because:

- text prompt/token parity is covered;
- text next-logit parity passes against Transformers;
- image processor parity is now covered for representative synthetic shapes;
- benchmark timing separates synchronized model decode from sampling and loop
  overhead;
- Metal decode profiling shows model work, not Rust bookkeeping, is the real
  bottleneck.

## Metrics

Every speculative run should report:

- `draft_tokens`: number of proposed draft tokens.
- `verified_tokens`: number of draft tokens sent through target verification.
- `accepted_tokens`: accepted draft tokens before mismatch or rejection.
- `bonus_tokens`: target-produced bonus tokens appended after verification.
- `accepted_length`: `accepted_tokens + bonus_tokens`.
- `acceptance_rate`: `accepted_tokens / verified_tokens`.
- `draft_seconds`: wall time spent producing drafts.
- `verify_seconds`: synchronized target verification time.
- `round_seconds`: draft plus verify plus sampling/scheduler overhead.
- `speedup_vs_baseline`: baseline decode seconds divided by speculative decode
  seconds for the same output token cap, prompt, dtype, and device.
- `verifier_waste_tokens`: verified suffix tokens after the first rejected
  position.
- `verifier_waste_share`: wasted verified tokens divided by `verified_tokens`.

For greedy decoding, the first correctness gate is simple: speculative output
must exactly match baseline greedy output token ids. For non-greedy sampling,
use the standard speculative acceptance rule later; do not relax correctness for
early speed claims.

## Stage 0: Verifier Harness

Build the verifier before building a drafter.

Implementation shape:

1. Add an eval command that accepts a prompt and a proposed draft token sequence.
2. Run the target model on `prompt + draft`.
3. For greedy mode, compare each draft token against the target argmax at that
   position.
4. Accept the longest matching prefix and append the target bonus token.
5. Emit JSON with accepted length, per-position target token ids, and timing.

This gives us a deterministic way to test tree/block/chain verification without
training a draft model first.

Acceptance gate:

- Verifier exactly reproduces baseline greedy generation when drafts are copied
  from a baseline run.
- Verifier rejects an intentionally corrupted suffix at the first bad token.
- Verification timing is recorded separately from generation-loop overhead.

## Stage 1: Built-In MTP Or Self-Draft Prototype

MiniCPM-V-4.6 config exposes `mtp_num_hidden_layers = 1`. If the checkpoint
contains usable MTP weights, this is the lowest-friction draft source.

Update from `audit-minicpm-mtp-weights`: the local MiniCPM-V-4.6 safetensors
header has no `mtp.*` or draft-like tensors, so this checkpoint does not provide
a built-in MTP drafter. Keep the MTP branch documented for future checkpoints,
but use the replay drafter path for the current experiment.

Plan:

1. Audit safetensor names for MTP modules and document whether weights are
   present.
2. If present, wire a one-token MTP draft head behind the text decoder.
3. Verify one-token drafts with the Stage 0 verifier.
4. Measure accepted length and speedup against baseline greedy decode.

If MTP weights are absent or not straightforward, use this stage to implement a
temporary "replay drafter" that consumes known baseline token ids. Replay is not
a speedup mechanism, but it validates verifier batching, acceptance accounting,
and output reconstruction.

Exit criteria:

- `accepted_length` and verifier timing are measurable.
- The speculative loop can produce identical greedy output.
- Speedup claims are blocked until a real drafter exists.

## Stage 2: EAGLE-Style Chain Drafter

EAGLE-3 is the first trained-drafter target because it is closer to the current
autoregressive runner than DFlash or DSpark.

Required runtime changes:

- Add hidden-state capture hooks for selected Qwen3.5 layers.
- Export target traces: prompt ids, generated ids, low/mid/high hidden states,
  logits, and accepted/rejected positions.
- Add a small draft module that predicts tokens directly from fused target
  features and previous draft tokens.
- Start with chain verification before dynamic trees.

Training sketch:

1. Generate trace data from the target model in greedy mode for the short,
   medium, long, code, and reasoning prompts.
2. Train direct token prediction from fused low/mid/high features.
3. Simulate train-time test by feeding the draft model its own previous sampled
   tokens for later positions.
4. Evaluate average accepted length by prompt class.

Exit criteria:

- Chain accepted length is above `1.5` on at least one stable prompt class.
- Draft latency is below one target decode step.
- End-to-end speedup exceeds `10%` before adding trees.

## Stage 3: DFlash-Style Block Drafter

DFlash becomes interesting after we can capture target features and run a
verifier. It is a larger jump because it needs a trained block-diffusion drafter.

Design constraints for this repo:

- Start with text-only MiniCPM/Qwen3.5 features.
- Use block sizes `4`, `8`, then `16`; do not start with long blocks.
- Treat KV injection as the primary design, not input-only feature fusion.
- Measure draft latency on Metal, not just accepted length.
- Compare against the EAGLE-style chain at equal target verification budget.

Exit criteria:

- Block drafter produces multiple draft tokens in one forward pass.
- Acceptance length improves over the chain drafter at equal or lower draft
  latency.
- End-to-end speedup exceeds `20%` on at least one prompt profile.

## Stage 4: DSpark-Style Scheduler

DSpark should be implemented only after a block drafter exists. Its main ideas
map to two local experiments:

1. Semi-autoregressive draft head:
   - Add a lightweight Markov head over parallel draft logits.
   - Feed the previous sampled draft token into the next position's adjustment.
   - Measure whether conditional acceptance decays less across block positions.
2. Confidence-scheduled verification:
   - Add a confidence head predicting per-position prefix survival.
   - Start with a single-request threshold scheduler.
   - Add hardware-aware batch scheduling only after the runner supports batched
     verification.

Local scheduler objective:

- For one request, select the longest prefix whose cumulative confidence has
  positive expected value after accounting for measured target verification
  latency.
- For future batches, maximize expected accepted tokens times measured
  target-model SPS at the verification batch size.

Exit criteria:

- Confidence scores are calibrated enough that higher scheduled confidence
  raises empirical acceptance rate.
- Scheduler reduces verifier waste without lowering output correctness.
- DSpark-style Markov head improves accepted length over DFlash-style parallel
  logits at the same block size.

## Prompt Matrix

Use text-only prompts until multimodal hidden-state parity exists:

- short factual prompt;
- arithmetic prompt;
- long reasoning prompt;
- code completion prompt;
- reasoning enabled with visible `<think>` output;
- reasoning disabled with closed thinking block.

Track prompt class separately. The papers show strong domain effects: code and
structured outputs often accept longer drafts than open chat.

## Pivot Gates

Do not claim speculative speedup unless:

- baseline greedy output token ids match speculative output token ids;
- comparison uses the same prompt, max token cap, dtype, device, and model
  revision;
- at least five measured iterations are reported;
- median speedup is above `10%` for prototype work or above `20%` before
  changing architecture direction;
- accepted length, draft latency, verify latency, and verifier waste are all in
  the JSON output.

## Proposed Follow-Up Tickets

1. `audit-minicpm-mtp-weights`
   - Inspect safetensor keys for MiniCPM/Qwen MTP modules.
   - Document whether a built-in draft head is available.
2. `implement-greedy-spec-verifier`
   - Add a text-only verifier command for proposed draft token sequences.
   - Validate exact greedy reconstruction and intentional suffix rejection.
3. `record-hidden-state-traces`
   - Add optional hidden-state capture for selected Qwen3.5 layers.
   - Export trace JSON/NPZ artifacts for draft training.
4. `prototype-eagle-chain-drafter`
   - Train or stub a direct token chain drafter over fused target features.
   - Measure accepted length and draft latency.
5. `design-dflash-block-drafter`
   - Specify block-diffusion architecture, feature injection, and training data
     layout after hidden-state traces exist.
6. `prototype-dspark-confidence-scheduler`
   - Add confidence-head outputs and single-request verification-length
     scheduling after a block drafter exists.
