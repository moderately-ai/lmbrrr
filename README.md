# lmbrrr

A fun project to push smaller language models to their limit with the latest and greatest in advances for inference.

The general approach to this will comprise of using the candle rust library, likely some forked changes needed... but as we progress we'll upstream improvements and examples.

## hardware scope

Metal is the sole supported backend — the crate is designed and optimized for Apple Silicon (M4 Max), and the Metal dependencies are unconditional:

```sh
cargo run --release -- run ...
```

The one build toggle is `metal-debug-labels`, which forwards Candle's Metal debug-label instrumentation for kernel-debugging sessions.

## models

Starting with (MiniCPM-V-4.6)[https://huggingface.co/openbmb/MiniCPM-V-4.6] as the base model to test out; the current target is the ternary **Ternary-Bonsai-27B** (Q2_0, 2.125 bpw) with a DSpark drafter for speculative decode.

## performance

`gguf spec` runs the proven operating point by default (planar mm2d verify + quality-free margin-1.0 acceptance + fused prefill). Measured tok/s (Q4_1 drafter, 96-token generation, warm):

| | M3 Pro | M4 Max |
|---|---|---|
| spec (default) | 17.4 | ~33 |
| plain decode | 14.5 | 33.1 |
| prefill / TTFT | 44 | 105 |

Note: speculative decode is a ~1.2× win on the (bandwidth-limited) M3 but roughly breaks even on the M4 Max, whose plain decode is already ~2.3× faster — see [docs/performance.md](docs/performance.md) for the full table. Full acceleration program (every spike P0–P10): [docs/research/full-acceleration-program-2026-07-19.md](docs/research/full-acceleration-program-2026-07-19.md), acceptance modes (`--fast` / `--exact` / `--no-mm2d`), and the engine's bottleneck analysis.

Versus the reference engines at the same ~2-bit point: lmbrrr leads the true-ternary-Q2_0 field, beating prism-ml's own llama.cpp fork on both hosts (14.5 vs 13.3 M3; 33.1 vs 28.2 M4). But prism-ml's MLX fork (affine 2-bit, ~2.19 bpw) is the fastest **raw** decode overall — 16.8 M3 / 40.3 M4 — and lmbrrr only edges it via speculation, and only on the M3 (17.4 vs 16.8). Full comparison, with the quant-scheme caveats, in [docs/performance.md](docs/performance.md#reference-engine-comparison).
