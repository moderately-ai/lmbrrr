# lmbrrr

A fun project to push smaller language models to their limit with the latest and greatest in advances for inference.

The general approach to this will comprise of using the candle rust library, likely some forked changes needed... but as we progress we'll upstream improvements and examples.

## hardware scope

Currently targeting metal as the primary target for this project at the moment.

Useful backend feature sets:

```sh
cargo run --release --features metal -- run ...
cargo run --release --features apple-optimized -- run ...
```

`metal` enables Candle's Metal backend. `apple-optimized` enables both Metal and
Accelerate. Other forwarded Candle backend features are available for future
experiments: `accelerate`, `mkl`, `cuda`, `cudnn`, `flash-attn`, and
`metal-debug-labels`.

## models

Starting with (MiniCPM-V-4.6)[https://huggingface.co/openbmb/MiniCPM-V-4.6] as the base model to test out.
