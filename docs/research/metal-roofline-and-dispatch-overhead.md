# Metal Roofline and Dispatch Overhead

Date: 2026-07-10

Ticket: `measure-metal-roofline-and-dispatch-overhead`

Host: MacBook Pro, binned Apple M4 Max (32 GPU cores, 36 GB unified, ~410 GB/s spec bandwidth). Command: `cargo run --release --features metal -- roofline --output target/metal-roofline.json`. All device measurements BF16 unless noted.

## Measured primitives

| Primitive | Result |
| --- | ---: |
| Streaming elementwise read+write (affine over 64 MB–1 GB) | 137–179 GB/s |
| Weight-read bandwidth, lm_head matvec (248094×1024, 508 MB) | 351 GB/s (1.45 ms/op) |
| Weight-read bandwidth, 8192×8192 matvec (134 MB) | 335 GB/s (0.40 ms/op) |
| Dependent 1024×1024 matvec chain | 6.46 µs/op ≈ 325 GB/s |
| MLP-shaped GEMV 3584×1024 (up/gate) | **82.6 GB/s** (88.8 µs/op) |
| MLP-shaped GEMV 1024×3584 (down) | **88.5 GB/s** (82.9 µs/op) |
| DeltaNet qkv GEMV 6144×1024 | 179.5 GB/s (70.1 µs/op) |
| Small projections (512–2048 out) | 41–250 GB/s (25–34 µs/op) |
| Bare dispatch (dependent tiny-affine chain) | 2.2 µs/dispatch |

The initial copy-bandwidth test read 10 PB/s because a bare `Tensor::zeros` buffer is elided by the backend; the command now materializes buffers first. The weight-read ceiling for decode purposes is **~350 GB/s (85% of spec)**, demonstrated by the lm_head and square matvecs.

## Interpretation: where the 16.5 ms token goes

Measured single-token forward (verify-table γ=1): 16.3–18.2 ms ≈ 60 tok/s. Summing the measured per-shape matvec costs over the model (18 DeltaNet layers ≈ 385 µs each, 6 full-attention layers ≈ 350 µs each, lm_head 1.45 ms) accounts for ~10.5 ms; the remaining ~6 ms is the DeltaNet recurrent-rule op chain, norms, RoPE, and mask/bookkeeping.

Three corrections to the campaign's earlier "dispatch-bound" hypothesis, now grounded in data:

1. **Raw dispatch launches are secondary.** ~550 dispatches × 2.2 µs ≈ 1.2 ms of the 16.5 ms token (~7%). Worth reclaiming via command-buffer discipline, but not the main lever.
2. **Small-GEMV underutilization is the primary dense inefficiency.** The MLP GEMVs run at ~83–89 GB/s — 4× below the achievable 350 GB/s — while a plain 1024×1024 dependent chain reaches ~325 GB/s. Candle's Metal GEMV tiling for the 3584-wide MLP shapes is the concrete suspect; fixing MLP GEMV utilization alone is worth ~4.7 ms/token (`fix-runner-hot-path-naive-ops` / `bf16-activation-quantized-matmul-metal`).
3. **The lm_head is honestly bandwidth-bound** at 1.45 ms/token (8.8% of the token at BF16, ~30% of a fully-optimized BF16 token). Only quantization shrinks it (`quantize-full-text-decoder-q4-incl-lm-head`).

## Projections (revised with measured numbers)

| Configuration | weight bytes | at 350 GB/s + overhead | projected tok/s |
| --- | ---: | ---: | ---: |
| BF16 today (measured) | 1.5 GB | — | 60 |
| BF16, GEMV fixed + DeltaNet fused | 1.5 GB | ~4.3 ms + ~1 ms | 180–220 |
| Q4K full (incl. lm_head), fused | ~0.45 GB | ~1.3 ms + ~1 ms | 300–500 |

These bracket the Stage 2 (≥150 BF16) and Stage 3 (≥250 quantized forwards/s) gates comfortably, provided the GEMV utilization and DeltaNet fusion work lands. The JSON artifact is `target/metal-roofline.json`; the projections embedded there still use the assumed-dispatch model and are superseded by this note's budget.
