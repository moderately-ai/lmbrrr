# q4k-full-text policy — full text-decoder quantization

Formalized 2026-07-11. The `q4k-full-text` conversion policy covers every text linear — MLP, full-attention, and DeltaNet projections — minus the two per-layer decay gates (`in_proj_a`/`in_proj_b`: 32 KB each, high sensitivity, zero bandwidth value). The lm_head (tied embedding, not a checkpoint tensor) pairs with the runtime `--quantize-lm-head q4k` flag. Protections are advisory under this policy: the sensitivity-set gate is skipped by design (campaign quality bar).

## from_source formats

New manifest formats `q4k_from_source` / `q6k_from_source` / `q8_0_from_source` reference the original safetensors instead of writing an lmbq data blob. The loader mmaps the source file and quantizes straight to the GGML QTensor — eliminating both the ~143 MB artifact duplicate and the lmbq→f32→GGML double quantization the older policies pay. Conversion is manifest-only in effect (seconds, no data file content). A per-tensor fallback ladder is available via repeatable `--fallback "<name-suffix>=<rung>"` (rungs q4k/q6k/q8-0), to be chosen by the quality harness on collapse — **not needed in practice** (see below). The manifest's `expected_weight_bytes` uses true GGML block sizes (q4_K 144/256, q6_K 210/256, q8_0 34/32) so roofline accounting is exact: 150 quantized tensors, text-linear reads drop from ~1.0 GB to ~0.28 GB per token.

## Measured (same session, math prompt, controls first)

| lane | q4k-mlp-q8-text (+q4k head) | q4k-full-text (+q4k head) |
| --- | --- | --- |
| greedy decode | 200.0 tok/s | **221.5–223.8 tok/s (+11%)** |
| verify chunk l=1/2/4/8 (ms) | 4.80 / 8.51 / 10.61 / 15.16 | **4.22 / 7.50 / 9.46 / 13.39** |
| scheduled spec, math | 146.9 tok/s @ τ 2.13 | 145.8 tok/s @ **τ 1.69** |

The greedy and verify lanes win outright. The spec lane is flat-to-negative: the round-1 drafter was trained against BF16-target hiddens, and heavier target quantization degrades draft acceptance (τ 2.13 → 1.69) — the same drafter–target mismatch recorded when the first quantized policies landed. Same-model greedy divergence also rises (advisory prefix 15 vs 128 on math): more quantization noise near ties. **Operating points per lane:** greedy/aggregate → `q4k-full-text`; speculative → `q4k-mlp-q8-text` until the drafter is retrained against quantized-target traces (round-2 scope).

## Quality (advisory, quant-quality ladder, 7 text cases)

| policy | gates passed | mean prefix ratio |
| --- | --- | --- |
| q8-text-linears | 4/7 | 0.538 |
| q4k-mlp-only | 3/7 | 0.462 |
| q4k-text-safe | 2/7 | 0.337 |
| q4k-mlp-q8-text | 4/7 | 0.572 |
| **q4k-full-text** | **4/7** | **0.521** |

No collapse (empty/looping output) on any case or class; spot checks on math/expository/code are fully coherent. Per campaign policy this is reported, not gated; the fallback ladder stays unused.

## Reproduce

```
lmbrrr quant-convert --policy q4k-full-text --output-dir target/minicpm-v46-q4k-full-text
lmbrrr run --quantized-manifest target/minicpm-v46-q4k-full-text/manifest.json --quantize-lm-head q4k ...
lmbrrr quant-quality --quantize-lm-head q4k   # ladder picks up the manifest automatically
```
