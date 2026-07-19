# Ternary-Bonsai-27B inference — performance & defaults

On-device speculative decoding for the ternary **Ternary-Bonsai-27B** (Q2_0, 2.125 bpw) target on Apple Silicon, Metal-only. This documents the measured throughput and the default operating point.

## Usage

```sh
# Speculative decode — runs the default operating point (planar mm2d verify,
# margin-1.0 acceptance, fused prefill). No env flags needed.
lmbrrr gguf spec \
  --gguf   Ternary-Bonsai-27B-Q2_0.gguf \
  --drafter Ternary-Bonsai-27B-dspark-Q8_0.gguf   # Q8_0 recommended (see below)

# Plain greedy decode (throughput floor; packed GEMV path)
lmbrrr gguf decode --gguf Ternary-Bonsai-27B-Q2_0.gguf --warmup
```

### Acceptance modes (`gguf spec`)

| flag | acceptance | quality | when |
|---|---|---|---|
| *(default)* | `--accept-margin 1.0` | **quality-free** (teacher-forced PPL == greedy; gate passed) | the default |
| `--fast` | `--accept-margin 3.0` | PPL +8–16% vs greedy (coherent, a real tradeoff) | max speed, quality-tolerant |
| `--exact` / `--accept-margin 0` | exact argmax | **byte-identical to greedy** (verified) | reproducibility / lossless |
| `--no-mm2d` | (any margin) | — | disable the planar verify path (packed GEMV) |

The mm2d and prefill routes are also overridable by env (`LMBRRR_MM2D=0`, `LMBRRR_DELTANET_PREFILL_FUSED=0`, …); env wins over the CLI default, so bench/suite scripts that set env are unaffected. Acceptance is CLI-only (the flags above).

## Throughput

Measured on the deployment target (M3 Pro) and the dev machine (M4 Max), same Q4_1 drafter, same prompt, 96-token generation, warm, default operating point unless noted.

| | **M3 Pro** (~150 GB/s) | **M4 Max** (~410 GB/s) | M4 speedup |
|---|---|---|---|
| **spec** (default, margin 1.0) | **18.2 tok/s** | **~33 tok/s** | 1.9× |
| spec `--fast` (margin 3.0) | **20.1** | 34.6 | 1.9× |
| plain decode | **14.5** | 33.1 | 2.3× |
| prefill / TTFT | 44 tok/s | 105 tok/s | 2.4× |

Acceptance is identical across machines (margin-1.0 → mean 3.0 tokens accepted / round). The Q8_0 drafter adds a few percent on top (~25% faster propose, ~13% of the round) with unchanged acceptance.

### Speculative decoding is regime-dependent

The headline: **spec is a clear win on the M3 but ~breaks even on the M4.**

- **M3 Pro** — spec 17.4 vs decode 14.5 = **1.2×** (1.25× at `--fast`). The M3's plain decode is memory-bandwidth-limited, so replacing sequential decode steps with an amortized batched verify pays off.
- **M4 Max** — spec ~33 vs decode 33.1 = **~1.0×** (1.05× at `--fast`). The M4 Max's plain decode is already ~2.3× the M3's, so the drafter-propose + verify overhead roughly cancels the acceptance gain.

Practical guidance:
- On the **M3**, run `gguf spec` — the default gives the 1.2× for free.
- On the **M4 Max**, plain `gguf decode` is essentially as fast as spec and simpler (no drafter). The speculative machinery is tuned for the bandwidth-starved regime; pushing the M4 further would mean re-profiling the verify/propose balance on that hardware, not more M3-tuned spec.

## Reference-engine comparison

How does lmbrrr stack up against the other engines that run this model? All three run the same Ternary-Bonsai-27B base at a ~2-bit operating point; decode is greedy, 96-token generation, warm, same prompt. The two reference engines are **prism-ml's own forks** (the ones shipped alongside the model): the [llama.cpp `prism` fork](https://github.com/PrismML-Eng) (Q2_0 type-42, same ternary quant as lmbrrr) and the [MLX `prism` fork](https://github.com/PrismML-Eng) driving `prism-ml/Ternary-Bonsai-27B-mlx-2bit` (their affine-2-bit port, with the fork's `qmv_wide` / ternary kernels).

| engine | quant | **M3 Pro** | **M4 Max** |
|---|---|---|---|
| **prism MLX** (fork) | affine 2-bit (~2.19 bpw) | **16.8** | **40.3** |
| **lmbrrr** (this repo) | Q2_0 ternary (2.125 bpw) | 14.5 | 33.1 |
| prism llama.cpp (fork) | Q2_0 ternary (2.125 bpw) | 13.3 | 28.2 |
| *lmbrrr spec* (margin 1.0, Q8_0 drafter) | Q2_0 + drafter | *17.4* | *~33* |

Read honestly:

- **prism MLX is the fastest raw decode on both hosts** — ~16% over lmbrrr on the M3 (16.8 vs 14.5), ~22% on the M4 (40.3 vs 33.1). MLX's affine-2-bit `qmv` kernels beat our Q2_0 GEMV. This is consistent with lmbrrr's own wall: every hot lmbrrr kernel is instruction-issue-bound at 35–74% of peak bandwidth (see Notes), and MLX's more mature kernels capture more of the remaining headroom.
- **Among true-ternary Q2_0 engines, lmbrrr is the fastest** — it beats the reference llama.cpp fork (the one prism-ml shipped) on both hosts: +9% on the M3 (14.5 vs 13.3), +17% on the M4 (33.1 vs 28.2).
- **lmbrrr's speculative decode is the only path that edges MLX, and only on the M3** — spec 17.4 vs MLX decode 16.8 (~4%; 18.1 at `--fast`). On the M4, MLX decode (40.3) is well ahead of lmbrrr spec (~33). So lmbrrr is competitive at the top of the M3 (bandwidth-starved) regime and behind on the M4.

Caveats — this is **not** a clean engine-vs-engine verdict:

- **Different quant schemes.** MLX runs affine 2-bit (4 levels, ~2.19 bpw); lmbrrr and llama.cpp run true ternary Q2_0 (3 levels, 2.125 bpw). MLX carries ~3% more bits and a more expressive codebook, so part of its speed advantage is a different operating point, not purely a faster engine. The apples-to-apples row is lmbrrr-vs-llama.cpp (identical Q2_0), where lmbrrr wins.
- **Quality was not cross-compared.** All three produced coherent output on the probe, but no cross-engine PPL was run. lmbrrr's margin-1.0 acceptance is PPL-equivalent to *its own* greedy — that is not a claim about MLX-vs-lmbrrr output quality.
- **MLX spec is unmeasured.** The prism MLX fork ships `spec_decode_verify`; a working MLX speculative path would likely extend its lead. lmbrrr's spec advantage here is only over MLX *decode*, not MLX spec.

Bottom line: lmbrrr is a from-scratch candle/Rust engine that leads the ternary-Q2_0 field and reaches MLX-decode parity on the M3 via speculation — but Apple's MLX, running a slightly richer 2-bit scheme through very mature Metal kernels, is the raw-throughput leader on both machines.

## Notes

- **Engine wall (M3, proven):** every hot kernel is instruction-issue-bound (mm2d verify 54 GB/s, decode-mv 111 GB/s, v2_decode 52 GB/s — all 35–74% of peak DRAM bandwidth), not bandwidth-bound. Bandwidth tricks don't pay; the verify sits at the matmul2d op's instruction-issue ceiling for the pre-M5 M3. See `tickets/verify-spec-acceleration-routemap.md`.
- **Planar plane cache:** the planar verify path builds a repacked plane artifact once per target, cached at `~/.cache/lmbrrr/mm2d` (content-addressed). The first `gguf spec` run on a cold cache builds it (~1 min for the 27B); subsequent runs hit the cache. Prebuild with `gguf repack --gguf <Q2_0> --out ~/.cache/lmbrrr/mm2d`.
- **Memory budget:** planar-only keeps ~9.2 GB resident (planes + drafter), under the M3 Pro's ~14.3 GB Metal working-set budget. Plain `LMBRRR_MM2D=1` *without* `--planar` loads both packed + planar copies (~15.5 GB) and silently corrupts on the M3 — the default (planar) avoids this.
- **Drafter:** Q8_0 is recommended (`gguf requant --gguf <bf16> --dtype q8_0`) — lossless acceptance, ~25% faster propose than Q4_1.

*Measurements: M3 Pro (macOS 27) / M4 Max (36 GB). Rotated where noted; M4 spec showed ~9% run-to-run variance vs the M3's tight repeats (dev-box thermal/background). Absolute tok/s is prompt- and length-sensitive — treat these as the ~100-token / warm regime.*

*v3 blessed baseline 2026-07-19 (Q8_0 drafter, N=128, 3 rotated reps, prose): plain 14.47 / exact 15.35 byte-match / m1 18.19 / m3 20.09. See ticket `blessed-v3-standings-re-baseline-post-f1-defaults-q8-0`.*

### Conf-adaptive margin (experimental)

`--adapt-margin 1.0,2.0` picks exact / margin-1 / margin-3 per round from mean draft confidence.
On a long prose run it can match `--fast` throughput with better teacher-forced PPL than global margin-3;
multi-prompt results are mixed — keep opt-in until the suite gate passes.
