# DeltaNet Decode Optimization

Date: 2026-07-07

Ticket: `optimize-deltanet-metal-decode`

## Objective

Improve MiniCPM-V-4.6 text decode throughput on Metal by reducing per-token overhead in the Qwen3.5 `GatedDeltaNet` layers while preserving the existing parity gates.

The hot-path profile showed DeltaNet recurrent decode as the largest synchronized component during one-token decode, so this pass stayed inside `src/qwen35.rs` and avoided custom Metal kernels.

## Kept Changes

`GatedDeltaNet` now caches two layer constants at load time:

- `dt_bias_f32`: `dt_bias` converted to F32 and reshaped for gate broadcasting.
- `a_log_exp_f32`: `exp(A_log)` converted to F32 and reshaped for gate broadcasting.

This removes repeated per-token dtype conversion, reshape, and exponentiation from `deltanet_gates_and_repeat`.

`GatedDeltaNet::recurrent_delta_rule` now dispatches `seq_len == 1` into a decode-specific helper. The helper performs the same recurrent update without allocating and iterating through a sequence-output vector. The recurrent cache is still stored back in the model dtype after each step so the deterministic decode sample remains unchanged from the prior runner.

## Rejected Variant

I tested a one-token depthwise-convolution shortcut that replaced the kernel-position loop with a single broadcast multiply plus reduction over the convolution window.

That variant improved synchronized component timing, but it changed the generated token sequence. The likely cause is BF16 reduction-order drift in the convolution sum. It also did not improve the unsynchronized 64-token benchmark:

- F32 recurrent state plus conv shortcut: median `61.67 tok/s`
- BF16 recurrent state plus conv shortcut: median `61.58 tok/s`
- Baseline: median `62.65 tok/s`

The final code keeps the original convolution arithmetic order.

## Correctness

Strict text-logit parity still passes against the Transformers oracle:

Command:

```sh
cargo run --features metal -- logits --top-k 10 --fail-on-mismatch --output target/minicpm-v46-candle-logits-parity-after-deltanet-no-conv-shortcut.json
```

Result:

- Passed: `true`
- Top-1 match: `3/3`
- Top-10 overlap: `9/10`, `9/10`, `10/10`
- Max shared logit delta: `0.25`

The retained variant also preserves the benchmark sample answer prefix from the baseline run.

## Performance

Baseline artifact:

- `target/minicpm-v46-metal-long-bench.jsonl`

Final artifact:

- `target/minicpm-v46-metal-long-bench-after-deltanet-no-conv-shortcut-64x5.jsonl`

Command:

```sh
cargo run --release --features metal -- bench --profile long --max-new-tokens 64 --warmup 2 --iterations 5 --output target/minicpm-v46-metal-long-bench-after-deltanet-no-conv-shortcut-64x5.jsonl
```

Median benchmark comparison:

| Metric | Baseline | Final | Delta |
| --- | ---: | ---: | ---: |
| Output rate | `62.65 tok/s` | `62.97 tok/s` | `+0.5%` |
| Steady-state rate | `64.90 tok/s` | `65.23 tok/s` | `+0.5%` |
| Prefill rate | `169.31 tok/s` | `171.31 tok/s` | `+1.2%` |
| Time to first token | `0.653s` | `0.646s` | `-1.1%` |

The end-to-end win is modest because the optimized work is one component of a larger decode step and because Candle/Metal queues work asynchronously in normal generation.

Synchronized profile comparison:

Baseline artifact:

- `target/minicpm-v46-metal-decode-profile-32.json`

Final artifact:

- `target/minicpm-v46-metal-decode-profile-after-deltanet-no-conv-shortcut.json`

Command:

```sh
cargo run --release --features metal -- profile --profile long --max-new-tokens 32 --output target/minicpm-v46-metal-decode-profile-after-deltanet-no-conv-shortcut.json
```

Profile deltas:

| Metric | Baseline | Final | Delta |
| --- | ---: | ---: | ---: |
| Profiled model forward | `10.14 tok/s` | `17.99 tok/s` | `+77.5%` |
| DeltaNet recurrent rule avg | `1.663 ms` | `0.639 ms` | `-61.6%` |
| DeltaNet gates/repeat avg | `0.301 ms` | `0.178 ms` | `-40.9%` |
| Argmax share | `0.48%` | `0.39%` | effectively unchanged |

The synchronized profile exaggerates wall-clock impact because every profiled component forces a device synchronization. It is still useful for identifying launch-heavy or scalar-heavy code paths, and it confirms that the recurrent helper and cached constants reduce component overhead.

## Next Implications

This ticket shows that small Candle-op cleanup helps, but it is not enough to materially push hardware utilization. The next meaningful speed work should target either:

- generation-loop overhead and sampling/logit transfer, which still runs every token, or
- a faithful custom Metal DeltaNet recurrent kernel, gated by a decode-level oracle so we can safely change arithmetic order.

