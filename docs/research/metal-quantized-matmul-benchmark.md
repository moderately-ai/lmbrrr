# Metal Quantized Matmul Benchmark

Date: 2026-07-07

Ticket: `benchmark-metal-quantized-matmul-kernels`

## Command

Run the benchmark with:

```sh
cargo run --release --features metal -- quant-matmul-bench \
  --chunk-tokens 128 \
  --warmup 2 \
  --iterations 5 \
  --output target/minicpm-v46-metal-quant-matmul-bench-chunk128.json
```

The command generates deterministic MiniCPM/Qwen3.5-shaped weights and
activations, then measures:

- dense Candle matmul for F32/F16/BF16 activations and weights;
- Candle `QTensor::quantize_onto` plus `QMatMul::forward` for Q8_0, Q4K, Q5K,
  and Q6K weights;
- decode MV shape (`[1, 1, in]`) and prefill/chunk MM shape
  (`[1, chunk_tokens, in]`) separately;
- activation cast behavior for quantized rows.

Representative shapes:

- DeltaNet `in_proj_qkv` and `out_proj`;
- MLP up/gate and down projections;
- full-attention q and o projections;
- optional LM head with `--include-lm-head`.

## Result Summary

Run:

```text
chunk_tokens=128, warmup=2, iterations=5, include_lm_head=false
```

The report produced 180 rows and no failed rows. Quantized F16/BF16 activation
rows use `activation_cast: "to_f32"` because Candle's current Metal quantized
matmul path requires F32 activations.

Median speedup versus dense F32 by mode:

| Mode | Q8_0 | Q4K | Q5K | Q6K |
| --- | ---: | ---: | ---: | ---: |
| decode MV | 1.78x | 1.76x | 1.49x | 1.18x |
| prefill MM | 1.51x | 1.16x | 1.00x | 1.02x |

Q8_0 is the strongest first target. Q4K helps decode but is less consistently
useful for prefill in this generated-shape benchmark. Q5K/Q6K are not compelling
enough to prioritize before we have real activation/logit sensitivity scores.

Cast overhead for Q8_0 was visible but not catastrophic in this short run:

- decode F16-to-F32 median ratio versus direct F32: 0.88x;
- decode BF16-to-F32 median ratio versus direct F32: 0.91x;
- prefill F16-to-F32 median ratio versus direct F32: 0.97x;
- prefill BF16-to-F32 median ratio versus direct F32: 1.03x.

## Decision

Do not write custom kernels immediately.

Use Candle's existing Q8_0 Metal path first for a real q8 text-linear loader.
It has plausible decode and chunk-prefill upside, and the current bottleneck is
still integrating true quantized storage into the model path. The loader ticket
currently dequantizes packed weights back to `QMatMul::Tensor`; replacing that
with real `QTensor` storage is the next useful step.

Custom BF16/F16 activation x quantized-weight Metal kernels become justified
only if a real model run shows the F32 activation cast dominates after the
loader uses true quantized storage.
