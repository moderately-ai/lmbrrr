# DSpark Block Runner

Date: 2026-07-10

Ticket: `integrate-dspark-block-runner`

## What landed

- `src/dspark.rs`: Candle port of DeepSpec's Qwen3DSparkModel inference — fused-context injection (fc + hidden_norm, per-layer projected ctx K/V cached with absolute-position RoPE), bidirectional block attention with per-head QK-RMSNorm, full 256-dim rotary at theta 1e7 (training's Qwen3 rope path ignores partial_rotary_factor — verified empirically), left-to-right greedy Markov sampling, confidence head. Loads the DeepSpec checkpoint directly (HF safetensors + config.json).
- **Parity oracle** (`dspark-drafter-parity`): a fixture generated in the pinned training environment on Modal (deterministic ctx + anchor -> block hidden, base/corrected logits, sampled tokens, confidence). Candle matches 8/8 sampled tokens with max diffs: block_hidden 0.031, base logits 0.0625, confidence 0.016 (BF16 Metal vs BF16 CPU torch).
- On-device capture in the target (`set_device_capture`) feeds the drafter's fused context with no CPU copies; verify-chunk captures for the anchor + accepted prefix are appended each round (valid regardless of rollback, since they were computed under correct state).
- `dspark-run --drafter <checkpoint>`: the trained drafter inside the rollback-verified multi-round loop.
- Enabled by a fork addition: candle had no Metal i32 casts at all; the `lmbrrr` branch of tomsanbear/candle (rev 7b6d1981) adds the full i32 cast surface, and lmbrrr's candle deps now point at that rev. Upstream PR planned (`upstream-fork-kernels`).

## First real-drafter measurement (smoke checkpoint: 491 conversations, 24 steps)

| Metric | Value |
| --- | ---: |
| Exact-greedy agreement | **128/128 tokens** through 126 rollbacks |
| Mean accepted length tau | 1.02 (accepted histogram: 125 rounds at 0, 1 at 1) |
| Position-0 acceptance | 0.8% |
| Speculative wall rate | 17.4 tok/s vs 65.2 greedy (0.27x — far below the tau 4-5 break-even, as expected) |
| Draft / verify / re-advance | 11 ms / 31 ms / 15 ms per round |

Interpretation: the machinery is complete and correct; the smoke drafter is (deliberately) useless. Drafter quality is now the campaign's sole blocking variable — `scale-dspark-training-corpus-modal` owns it, with the tau-vs-corpus curve deciding investment. Known runner optimization for when tau is real: the Markov sampling loop performs one argmax readback + one confidence readback per position (8 device syncs per proposal); batch or defer them.

Comparisons against the recurrent-EAGLE smoke and the speedup gates run in `benchmark-full-dspark-speedup` once a real drafter exists.
