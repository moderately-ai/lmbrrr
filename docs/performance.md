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
| **spec** (default, margin 1.0) | **17.4 tok/s** | **~33 tok/s** | 1.9× |
| spec `--fast` (margin 3.0) | 18.1 | 34.6 | 1.9× |
| plain decode | 14.5 | 33.1 | 2.3× |
| prefill / TTFT | 44 tok/s | 105 tok/s | 2.4× |

Acceptance is identical across machines (margin-1.0 → mean 3.0 tokens accepted / round). The Q8_0 drafter adds a few percent on top (~25% faster propose, ~13% of the round) with unchanged acceptance.

### Speculative decoding is regime-dependent

The headline: **spec is a clear win on the M3 but ~breaks even on the M4.**

- **M3 Pro** — spec 17.4 vs decode 14.5 = **1.2×** (1.25× at `--fast`). The M3's plain decode is memory-bandwidth-limited, so replacing sequential decode steps with an amortized batched verify pays off.
- **M4 Max** — spec ~33 vs decode 33.1 = **~1.0×** (1.05× at `--fast`). The M4 Max's plain decode is already ~2.3× the M3's, so the drafter-propose + verify overhead roughly cancels the acceptance gain.

Practical guidance:
- On the **M3**, run `gguf spec` — the default gives the 1.2× for free.
- On the **M4 Max**, plain `gguf decode` is essentially as fast as spec and simpler (no drafter). The speculative machinery is tuned for the bandwidth-starved regime; pushing the M4 further would mean re-profiling the verify/propose balance on that hardware, not more M3-tuned spec.

## Notes

- **Engine wall (M3, proven):** every hot kernel is instruction-issue-bound (mm2d verify 54 GB/s, decode-mv 111 GB/s, v2_decode 52 GB/s — all 35–74% of peak DRAM bandwidth), not bandwidth-bound. Bandwidth tricks don't pay; the verify sits at the matmul2d op's instruction-issue ceiling for the pre-M5 M3. See `tickets/verify-spec-acceleration-routemap.md`.
- **Planar plane cache:** the planar verify path builds a repacked plane artifact once per target, cached at `~/.cache/lmbrrr/mm2d` (content-addressed). The first `gguf spec` run on a cold cache builds it (~1 min for the 27B); subsequent runs hit the cache. Prebuild with `gguf repack --gguf <Q2_0> --out ~/.cache/lmbrrr/mm2d`.
- **Memory budget:** planar-only keeps ~9.2 GB resident (planes + drafter), under the M3 Pro's ~14.3 GB Metal working-set budget. Plain `LMBRRR_MM2D=1` *without* `--planar` loads both packed + planar copies (~15.5 GB) and silently corrupts on the M3 — the default (planar) avoids this.
- **Drafter:** Q8_0 is recommended (`gguf requant --gguf <bf16> --dtype q8_0`) — lossless acceptance, ~25% faster propose than Q4_1.

*Measurements: M3 Pro (macOS 27) / M4 Max (36 GB). Rotated where noted; M4 spec showed ~9% run-to-run variance vs the M3's tight repeats (dev-box thermal/background). Absolute tok/s is prompt- and length-sensitive — treat these as the ~100-token / warm regime.*
