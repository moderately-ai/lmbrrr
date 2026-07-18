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

Note: speculative decode is a ~1.2× win on the (bandwidth-limited) M3 but roughly breaks even on the M4 Max, whose plain decode is already ~2.3× faster — see [docs/performance.md](docs/performance.md) for the full table, acceptance modes (`--fast` / `--exact` / `--no-mm2d`), and the engine's bottleneck analysis.
