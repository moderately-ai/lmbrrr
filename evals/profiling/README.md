# evals/profiling — on-device GPU-trace profiling harness (M3, macOS 27)

Reusable tooling for the measurement/rigor loop: black-box wall-time sweeps and
white-box `gpucapture`/`gpudebug` GPU-trace profiling of Metal kernels (ours and
MLX's, for cross-checks). The full command reference is `../../metal_notes.md`;
this is the runnable harness + the one-time setup. Governed by
`../../docs/research/rigor-protocol.md` (triangulate wall time + counters + source
before any conclusion).

All of this runs on the **M3 referee** (macOS 27, `gpucapture`/`gpudebug` 1.0). The
M4 dev box is macOS 15 and has neither the tools nor Metal 4.1 — edit/commit here,
run there (`ssh m3`, repo at `~/lmbrrr-work/lmbrrr`).

## One-time setup: a debuggable Python (no sudo)

`gpucapture` only attaches to a **debuggable** process (`com.apple.security.get-task-allow`).
A `cargo` binary has it for free; stock/Xcode/Homebrew Python does not, and MLX's own
`mx.metal.start_capture()` writes a bundle `gpudebug` cannot open. So MLX must be run
under a re-signed uv interpreter. uv's cpython is adhoc/linker-signed with no hardened
runtime, so re-signing is clean and does not break MLX's dylibs:

```sh
# on the M3:
uv add mlx                    # into the workspace venv (~/lmbrrr-work/lmbrrr/.venv)
P=~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3.12
codesign -f -s - --entitlements evals/profiling/get-task-allow.plist "$P"
codesign -d --entitlements - "$P"    # verify get-task-allow=true
"$P" -c 'import mlx.core as mx; mx.eval(mx.zeros((8,8))@mx.zeros((8,8)))'  # still imports
```

## Files

- `mlx_qmm_bench.py` — black-box wall-time sweep; crosses the qmv/qmm batch boundary
  (m=1..32) so you actually measure the path you think you do. Reports eff/raw GB/s + GFLOP/s.
- `mlx_prep.py` — phase 1 (uncaptured): synthesize+quantize+save inputs. Edit shape/m here.
- `mlx_qmm_only.py` — phase 2 (captured): load + loop the kernel only.
- `mlx_capture.sh` — orchestrates the two-phase `gpucapture` flow. `PY=<debuggable-python> ./mlx_capture.sh`.
- `get-task-allow.plist` — the entitlement for the re-sign step.

## Two hard-won traps (both in metal_notes, repeated here)

- **Counters are timeline-global** — the captured region must contain ONLY the kernel.
  A naive capture is ~80% `mx.random.normal` RNG; the tell is `gpu_write_bandwidth ≈
  gpu_read_bandwidth`. That is why capture is two-phase (prep uncaptured → kernel-only captured).
- **Never `profile run --embed`** on a large trace — it deadlocks the session (client at
  0% CPU, a separate `status` client also blocks). Recovery: `kill -9` the clients AND
  `GPUToolsReplayService.xpc`. Use plain `profile run`, then navigate
  `performance/timeline/counters/*` directly (external captures have no embedded session,
  so `profile load 0` failing with "no profiling sessions in trace" is expected).

## First result this harness produced (2026-07-17)

MLX 2-bit `qmm` at m=32 is `f32_limiter 91.55%` (dense bf16 GEMM after dequant-to-threadgroup;
bf16 runs on the f32 pipe) — a fundamental floor 2.8× our mm2d at m≤8, and MLX uses the
slower `qmv` below m=10 anyway. Independently confirms the verify kernel is settled.
Full write-up: `metal_notes.md` §15.E. Ticket: `dequant-bf16-dense-gemm-verify` (closed wontdo).
