# Quantized Linear Loader

Date: 2026-07-07

Ticket: `load-minicpm-quantized-linear-weights`

## Command Usage

Any model-loading command can now take a mixed-precision manifest:

```sh
cargo run --release --features metal -- logits \
  --quantized-manifest target/minicpm-v46-q8-smoke/manifest.json \
  --output target/minicpm-v46-q8-smoke-logits.json
```

Benchmark smoke:

```sh
cargo run --release --features metal -- bench \
  --quantized-manifest target/minicpm-v46-q8-smoke/manifest.json \
  --profile short \
  --max-new-tokens 8 \
  --warmup 0 \
  --iterations 1 \
  --output target/minicpm-v46-q8-smoke-bench.jsonl
```

## Loader Behavior

The runner loads the normal MiniCPM safetensors first, then applies the manifest
as a text-linear replacement pass. Only tensor rows with packed data in the
manifest are replaced.

Current replaced families:

- text MLP projections;
- full-attention q/k/v/o projections;
- DeltaNet projection weights.

Still protected on the source path:

- token embeddings and LM head;
- norms;
- DeltaNet `A_log`, `dt_bias`, conv weights, and recurrent/conv state;
- vision tower;
- multimodal merger.

The abstraction is `MixedLinear`: dense Candle `Linear` or Candle `QMatMul`.
For this first loader, packed q8/q4k artifact values are dequantized back into a
normal tensor and wrapped as `QMatMul::Tensor`. This validates manifest loading,
replacement, and logits/bench integration, but it is not yet a speed path.

JSON reports include:

```json
"quantized_load": {
  "manifest": ".../manifest.json",
  "quantized_tensors": 2,
  "replaced_text_linears": 2,
  "backend": "dequantized_qmatmul_tensor"
}
```

## Verification

Q8 smoke artifact:

- `logits` with `target/minicpm-v46-q8-smoke/manifest.json` replaced 2 text
  linears and passed the existing text logits parity fixture.
- `bench --profile short --max-new-tokens 8` ran generation with the same
  artifact and recorded token-rate metrics plus quantized-load metadata.

The next step is benchmarking real quantized matmul kernels instead of this
dequantized correctness fallback.
