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

Starting with (MiniCPM-V-4.6)[https://huggingface.co/openbmb/MiniCPM-V-4.6] as the base model to test out.
