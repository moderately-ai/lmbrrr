# MiniCPM-V-4.6 Transformers Parity Oracle

This ticket establishes a repeatable parity path between the Candle runner and
the upstream MiniCPM-V-4.6 Transformers implementation.

## What Is Committed

- `evals/fixtures/minicpm_v46_text_prompts.json` records prompt strings and
  token ids for four cases:
  - short text, closed thinking
  - short math, open thinking
  - longer reasoning-style prompt, closed thinking
  - single-image chat-template marker, closed thinking
- `tests/minicpm_v46_text_parity.rs` checks the fixture against
  `lmbrrr::prompt::chat_prompt` and the vendored MiniCPM tokenizer at
  `docs/research/models/minicpm-v-4.6/hf-model/tokenizer.json`.
- The same test file also checks the Rust image placeholder expansion shape
  for a synthetic processed-image layout with image id tags and slice tags.
- `evals/minicpm_v46_oracle.py` is the optional Transformers oracle used to
  regenerate comparable JSON from `AutoTokenizer` and `AutoProcessor`.
- `evals/fixtures/minicpm_v46_transformers_text_logits.json` records top-10
  next-token logits for the three text-only prompts from the live Transformers
  model.
- `evals/fixtures/minicpm_v46_transformers_image_expansion.json` records the
  live Transformers processor expansion for the model-card refract image. The
  single-image prompt expands from 19 chat-template tokens to 211 processor
  tokens at `downsample_mode=16x`.

## Local Verification

Run the CI-cheap parity test:

```sh
cargo test --test minicpm_v46_text_parity
```

Regenerate the Rust-side fixture for inspection:

```sh
cargo test --test minicpm_v46_text_parity regenerate_minicpm_v46_text_fixture -- --ignored --nocapture
```

Inspect the external oracle CLI without importing optional dependencies:

```sh
python3 evals/minicpm_v46_oracle.py --help
```

Generate a live Transformers prompt fixture once a MiniCPM-capable Transformers
environment is installed:

```sh
pip install "transformers[torch]>=5.7.0" torchvision
python3 evals/minicpm_v46_oracle.py \
  --output /tmp/minicpm_v46_transformers_prompts.json
```

Capture processor-expanded image token ids by supplying an image:

```sh
python3 evals/minicpm_v46_oracle.py \
  --image https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/refract.png \
  --output evals/fixtures/minicpm_v46_transformers_image_expansion.json
```

Capture selected next-token logits for text-only prompts after model weights are
available:

```sh
python3 evals/minicpm_v46_oracle.py \
  --model-dir docs/research/models/minicpm-v-4.6/hf-model \
  --weights-dir /Users/tsanterre/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4.6/snapshots/8169864629825dc1d755a5aa1cd8b5935dcbc83f \
  --with-next-token \
  --top-k-logits 10 \
  --output evals/fixtures/minicpm_v46_transformers_text_logits.json
```

The run used `transformers==5.13.0`, `torch==2.12.1`,
`torchvision==0.27.1`, and cached snapshot
`8169864629825dc1d755a5aa1cd8b5935dcbc83f`.

## Parity Boundaries

The committed fixture is generated from the Rust prompt renderer and the
vendored tokenizer. It is designed to fail quickly if our runner drifts from the
mirrored MiniCPM chat-template behavior.

The Python oracle is the source of truth for live Transformers comparison. The
repo tests still keep those imports outside normal `cargo test`, so CI does not
need to install Torch or load model weights.

The single-image committed fixture covers the chat-template stage, where one
`<|image_pad|>` marker is inserted. Full image expansion depends on
`AutoProcessor` image resizing, slicing, `downsample_mode`, and per-image patch
grid metadata. The Rust test covers the string expansion algorithm with
synthetic metadata; the Python oracle can record the real expanded token ids
when an image path is supplied.

Top-10 next-token logits are committed for the text-only prompts. The next step
is now implemented by `lmbrrr logits`:

```sh
cargo run --features metal -- logits \
  --top-k 10 \
  --fail-on-mismatch \
  --output target/minicpm-v46-candle-logits-parity-strict.json
```

On July 7, 2026 this passed on Metal with top-1 agreement for all three
text-only prompts, top-10 overlaps of 9/10, 9/10, and 10/10, and max shared
logit delta of 0.25. The image fixture currently captures processor expanded
token ids, not image-conditioned logits.
