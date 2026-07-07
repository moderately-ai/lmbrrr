# MiniCPM-V-4.6 Image Parity

Date: 2026-07-07

Ticket: `validate-minicpm-image-parity`

## Objective

Add an executable parity check for the MiniCPM-V-4.6 image preprocessing path.
This is the gate before treating multimodal generation results as meaningful.

## Oracle

Fixture generator:

```sh
/tmp/lmbrrr-minicpm-oracle-venv/bin/python evals/minicpm_v46_image_oracle.py \
  --output evals/fixtures/minicpm_v46_transformers_image_processor.json
```

Fixture:

- `evals/fixtures/minicpm_v46_transformers_image_processor.json`

The oracle uses `AutoProcessor.from_pretrained(...).image_processor` from the
vendored MiniCPM-V-4.6 metadata and deterministic synthetic RGB images. No
binary test images are checked into the repository.

Cases:

| Case | Source size | Grid | Patch count | Pixel shape |
| --- | ---: | ---: | ---: | --- |
| `synthetic_small_unsliced` | `64x96` | `[0, 0]` | `1` | `[1, 3, 14, 15680]` |
| `synthetic_large_sliced` | `900x1200` | `[2, 3]` | `7` | `[1, 3, 14, 100128]` |
| `synthetic_tall_sliced` | `1200x600` | `[3, 1]` | `4` | `[1, 3, 14, 61824]` |

## Rust Check

Test:

```sh
cargo test --test minicpm_v46_text_parity image_processor_matches_transformers_oracle
```

The Rust test regenerates the same synthetic pixels in memory and checks:

- `pixel_values` shape;
- `target_sizes`;
- per-image grid;
- patch count;
- pixel sum and mean with small aggregate tolerance;
- 64 sampled flattened pixel values per case with `0.04` absolute tolerance.

The sampled-value tolerance accounts for interpolation differences between the
Rust `image` crate resize implementation and the Transformers backend. The
structural values are exact.

## Implementation Note

`preprocess_paths` now delegates to `preprocess_rgb_images`, which accepts
in-memory `RgbImage` inputs. The CLI behavior remains path-based, while tests
can avoid checked-in binary images.

## End-to-End Smoke

Ticket: `port-minicpm-v46-full-path`

An end-to-end Metal BF16 image-conditioned run now succeeds with a deterministic
temporary image generated under `target/multimodal-smoke/`:

```sh
magick -size 128x96 gradient:'#1b5e8f-#f5d76e' \
  -fill '#ffffff' -draw 'rectangle 20,24 108,72' \
  -fill '#1b5e8f' -draw 'circle 64,48 64,24' \
  target/multimodal-smoke/simple-shapes.png

cargo run --release --features metal -- run \
  --prompt "Describe the image in one short sentence." \
  --image target/multimodal-smoke/simple-shapes.png \
  --max-new-tokens 16 \
  --no-progress
```

Observed answer:

```text
The image shows a simple design with a blue circle centered within a white rectangle,
```

Observed metrics:

- device: `Metal(MetalDevice(DeviceId(1)))`
- dtype: `BF16`
- prompt tokens: `89`
- generated tokens: `16`
- prefill: `0.727834333s`, `122.28 tok/s`
- decode: `0.231624833s`, `69.08 tok/s`

Two runtime fixes were required for this smoke:

- image preprocessor tensors remain F32, then cast to the text embedding dtype
  before the vision tower so BF16/F16 model loads match conv weights;
- chunked vision attention makes narrowed `q`, `k`, `v`, and `k^T` tensors
  contiguous before Metal matmuls.

## Remaining Parity Work

This validates image preprocessing, placeholder expansion, the compiled vision
tower path, image embedding insertion, and one plausible image-conditioned
decode. It is still not full numeric parity for the vision tower. The next gate
should compare image feature shapes and selected hidden-state values after
`get_image_features` against Transformers.
