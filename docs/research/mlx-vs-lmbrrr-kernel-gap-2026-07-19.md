# MLX vs lmbrrr kernel gap (2026-07-19, M3 Pro)

## Headline
On the large ternary FFN shape (17408×5120, m=1), **lmbrrr Q2_0 GEMV is ~1.7× faster than MLX quantized_matmul (bits=2, gs=128)**.

The previously reported e2e MLX decode lead (~16.8 vs ~14.5 tok/s) is therefore **not** explained by a faster 2-bit matvec. Look at fusion, dispatch count, host path, and quant scheme differences.

## Isolated numbers

### lmbrrr `gguf bench-gemv`
| kernel | shape | m | ms | eff GB/s |
|---|---|---:|---:|---:|
| Q2_0 GEMV | 17408×5120 | 1 | 0.223 | 106.1 |
| Q2_0 GEMV | 5120×5120 | 1 | 0.071 | 98.3 |
| mm2d t64_k128 | 17408×5120 | 1..8 | ~0.55 | ~43 (flat) |
| BITPLANE int4-act | 17408×5120 | 1 | 0.191 | 123.9 |

### MLX `evals/profiling/mlx_qmm_bench.py` + focused 17408 run
| path | shape | m | ms | eff GB/s |
|---|---|---:|---:|---:|
| qmv bits2/gs128 | 17408×5120 | 1 | 0.380 | 62.4 |
| qmv | 17408×5120 | 5 | 1.114 | 21.3 |
| qmv | 17408×5120 | 8 | 1.650 | 14.3 |
| qmv | 5120×5120 | 1 | 0.244 | 28.5 |
| qmv gate_up 34816×5120 | m=1 | 0.555 | 85.3 |

MLX switches qmv→qmm near m=10; at verify width m≤8 it stays on qmv and **does not amortize**.

## Implications
- Do **not** spend campaign time porting MLX qmv for decode m=1.
- e2e catch-up if desired: profile dispatch counts / fusion / host vs MLX full model (needs MLX Bonsai weights).
- Spec path already prefers mm2d which MLX does not match at m≤8.

## Receipts
M3 `/tmp/mlx_bench.log`; lmbrrr `gguf bench-gemv` stdout.
