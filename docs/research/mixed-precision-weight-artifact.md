# Mixed-Precision Weight Artifact

Date: 2026-07-07

Ticket: `convert-minicpm-mixed-precision-weights`

## Command

The first converter writes a custom lmbrrr artifact:

```sh
cargo run --release --features metal -- quant-convert \
  --sensitivity artifacts/minicpm-v46-quant-sensitivity.json \
  --policy q8-text-linears \
  --output-dir artifacts/minicpm-v46-q8-text-linears
```

Fast smoke conversion:

```sh
cargo run --release --features metal -- quant-convert \
  --sensitivity artifacts/minicpm-v46-quant-sensitivity.json \
  --policy q8-text-linears \
  --max-tensors 2 \
  --output-dir artifacts/minicpm-v46-q8-smoke
```

The converter currently supports:

- `q8-text-linears`: quantize unprotected text MLP, full-attention, and DeltaNet
  linear weights to per-tensor symmetric q8.
- `q4k-mlp-only`: quantize unprotected text MLP weights to block-64 symmetric
  packed q4.
- `q4k-text-safe`: quantize text MLP and full-attention q/k/v/o projection
  weights to block-64 symmetric packed q4, while leaving DeltaNet projections
  in source precision.

## Files

Each artifact directory contains:

```text
manifest.json
weights.lmbq
```

`manifest.json` records:

- source model id and revision;
- source safetensor file paths, sizes, and SHA-256 hashes;
- sensitivity artifact path, hash, kind, schema version, and candidate formats;
- one row per tensor with family, shape, source dtype, source bytes, chosen
  format, protection reason, and expected bytes;
- offsets and lengths into `weights.lmbq` for quantized tensors.

Protected tensors are not rewritten. Their manifest rows use `format: "source"`
and point back to the exact source safetensor file.

`weights.lmbq` stores only quantized tensors:

- q8 tensors: one little-endian f32 scale followed by one signed byte per value;
- q4k tensors: repeated 64-value blocks, each with one little-endian f32 scale
  followed by packed signed 4-bit values encoded as `value + 8`.

This is intentionally a small custom format for the next loader ticket. It is
not GGUF and does not yet use Candle's `QTensor` serialization.

## Limits

The converter is deterministic but conservative. It relies on the sensitivity
artifact to identify scored unprotected text tensors, but q4 policies are still
policy experiments rather than quality-approved recommendations. The current
sensitivity pass showed per-tensor q4/q5 error is high, so q4k artifacts should
be treated as loader/kernel experiments until activation and logit drift scoring
exists.
