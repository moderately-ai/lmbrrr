# Metal Decode Hot Path Profile

Date: 2026-07-07

This note records the first instrumented text-only decode profile for the
MiniCPM-V-4.6 Candle runner on Metal.

## Commands

Normal release benchmark:

```sh
cargo run --release --features metal -- bench \
  --profile long \
  --max-new-tokens 64 \
  --warmup 1 \
  --iterations 3 \
  --output target/minicpm-v46-metal-long-bench.jsonl
```

Synchronized component profile:

```sh
cargo run --release --features metal -- profile \
  --profile long \
  --max-new-tokens 32 \
  --output target/minicpm-v46-metal-decode-profile-32.json
```

The profile command runs the same long benchmark prompt, performs prefill, then
profiles 32 true single-token decode forwards. It synchronizes the Metal device
around each profiled component. That makes the profile intrusive and slower
than normal generation, but it gives useful attribution between code-path
families.

## Baseline Throughput

The normal release benchmark used a 102-token prompt and generated 64 tokens per
iteration.

| Metric | Values |
| --- | --- |
| Prefill tok/s | 168.54, 170.71, 169.31 |
| Output tok/s | 62.73, 62.65, 62.20 |
| Steady-state tok/s | 64.98, 64.90, 64.42 |
| Decode seconds | 1.020, 1.021, 1.029 |

Median output rate is about 62.65 tok/s. Median steady-state rate is about
64.90 tok/s.

## Decode Component Breakdown

The synchronized 32-step profile reported 10.14 profiled decode forwards/sec.
This is not the user-visible token rate; it includes synchronization after each
component scope.

| Component | Share | Total Seconds | Avg per Event |
| --- | ---: | ---: | ---: |
| DeltaNet recurrent rule | 34.6% | 0.958 | 1.663 ms |
| DeltaNet depthwise conv | 9.1% | 0.252 | 0.437 ms |
| MLP | 8.7% | 0.242 | 0.315 ms |
| DeltaNet output gate/norm/projection | 8.2% | 0.228 | 0.397 ms |
| Input RMSNorm | 7.7% | 0.214 | 0.279 ms |
| Post-attention RMSNorm | 7.6% | 0.210 | 0.273 ms |
| DeltaNet gates/repeat | 6.3% | 0.174 | 0.301 ms |
| DeltaNet QKV projection | 5.1% | 0.140 | 0.244 ms |
| Full-attention rotary/KV cache/repeat | 4.6% | 0.128 | 0.666 ms |
| Full-attention KV projection/norm | 3.1% | 0.086 | 0.450 ms |
| Full-attention matmul/softmax | 1.7% | 0.048 | 0.252 ms |
| Full-attention output projection | 1.5% | 0.043 | 0.222 ms |
| Full-attention Q/gate projection | 1.4% | 0.038 | 0.195 ms |

Grouped by layer kind:

| Group | Share |
| --- | ---: |
| Linear-attention / DeltaNet layers | 81.3% |
| Full-attention layers | 18.4% |
| Final norm | 0.3% |

Argmax plus scalar transfer took 0.015 seconds total across 32 decode forwards,
about 0.48% of model-forward-plus-argmax time in the synchronized profile. It
is not the first bottleneck.

## Interpretation

The decode hot path is dominated by the Qwen3.5 linear-attention path, not the
full-attention matmul/softmax path. The single biggest target is the DeltaNet
recurrent update. The depthwise causal convolution and output gate/norm/proj
blocks are secondary DeltaNet targets.

Full-attention layers are measurable but not dominant. The full-attention
matmul/softmax component is especially small in this long-prompt decode profile
because only six of twenty-four layers are full-attention layers and single-token
decode uses cached keys/values.

The profiler does not count exact Metal kernel launches. It records synchronized
component scopes, with 193 profiled events per decode forward. Exact launch
counts still require Xcode/Metal capture or lower-level Candle/Metal tracing.

## Ranked Backlog

1. Optimize DeltaNet recurrent decode.
   Preserve the existing text logits gate and start by reducing the number of
   Candle ops in the recurrent update. A grouped Candle implementation is worth
   trying before a custom Metal kernel; if launch count or intermediate tensors
   remain high, a fused Metal kernel becomes justified.

2. Optimize DeltaNet depthwise causal convolution.
   It is the second-largest named component and currently performs explicit
   per-token/per-kernel-loop tensor work.

3. Reduce DeltaNet output gate/norm/projection overhead.
   This block is roughly the same size as the depthwise conv and may benefit
   from avoiding repeated dtype conversions or reshapes.

4. Investigate RMSNorm overhead.
   Combined input and post-attention RMSNorm scopes are about 15% of profiled
   decode component time. This may partly reflect synchronization granularity,
   but it is large enough to watch.

5. Defer full-attention/DFlash-style work for now.
   Full-attention matmul/softmax is only about 1.7% of the synchronized decode
   component profile. Optimizing it first is not justified by this evidence.

6. Defer sampling/CPU-transfer optimization as a primary lane.
   Argmax plus scalar transfer is below 1% in this profile. Generation-loop
   cleanup may still matter for measurement hygiene, but it is not the dominant
   model decode cost.

## Recommended First Optimization

Start with DeltaNet recurrent decode. The target is to preserve top-k logits
parity while reducing op count and intermediate tensors in
`GatedDeltaNet::recurrent_delta_rule`.

## Tempting But Not Justified Yet

DFlash/full-attention kernel work is not justified as the next step. It is
important research context, but this MiniCPM-V-4.6 text decode profile points at
DeltaNet first.
