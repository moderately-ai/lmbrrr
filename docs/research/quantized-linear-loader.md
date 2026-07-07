# Quantized Linear Loader

Date: 2026-07-07

Tickets: `load-minicpm-quantized-linear-weights`,
`store-quantized-weights-as-qtensor`

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
The first loader used a dequantized correctness fallback. The current loader
keeps replaced linears as Candle `QTensor` weights:

1. read the custom `weights.lmbq` tensor bytes;
2. dequantize transiently on CPU into F32 values;
3. re-quantize into Candle's native GGML layout on the target device with
   `QTensor::quantize_onto`;
4. store `QMatMul::from_qtensor` in the model.

The transient CPU tensor is dropped after replacement. This is not zero-copy
from the custom artifact, because the artifact's simple symmetric q8/q4k bytes
are not Candle/GGML block layouts. It does avoid keeping the replaced model
weights as dense runtime tensors.

Activation inputs are cast to F32 for quantized `QMatMul` and cast back to the
original activation dtype afterward. This matches the Metal quantized matmul
benchmark finding that Candle's current quantized path expects F32 activations.

JSON reports include:

```json
"quantized_load": {
  "manifest": ".../manifest.json",
  "quantized_tensors": 2,
  "replaced_text_linears": 2,
  "backend": "candle_qtensor_requantized",
  "quantized_data_bytes": 32776,
  "dense_equivalent_bytes": 65536,
  "approx_dense_bytes_avoided": 32760
}
```

## Verification

Q8 smoke artifact:

- `logits` with `target/minicpm-v46-q8-smoke/manifest.json` replaced 2 text
  linears with `candle_qtensor_requantized` and passed the existing text logits
  parity fixture.
- `bench --profile short --max-new-tokens 8` ran generation with the same
  artifact and recorded token-rate metrics plus quantized-load metadata:
  prefill `130.46 tok/s`, decode `10.57 tok/s`.

The next step is a full quantized inference benchmark, not a two-tensor smoke,
so we can see whether native QTensor storage and matmul speedups survive
end-to-end model overhead.
