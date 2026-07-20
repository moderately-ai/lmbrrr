# AGENTS.md — cold-start orientation for any agent/session

lmbrrr is a from-scratch, throughput-maxed **on-device inference engine** (Rust + a
Metal candle fork) for `prism-ml/Ternary-Bonsai-27B` — a 27B ternary-quantized (Q2_0,
2.125 bpw) hybrid Qwen3.5 VLM (gated-DeltaNet + every-4th full-attention), running its
text-decode path on an Apple **M3 Pro (18 GB)**, with a DSpark speculative-decode layer.

This file is the harness-agnostic source of truth. (Claude Code also reads `~/.claude`
memory + `metal_notes.md`; a different harness should rely on the repo: this file, the
tickets, `docs/research/`, and `metal_notes.md`.)

## Read order (cold start)

1. **This file** — orientation, constraints, where things live.
2. **The goal** (below) — the campaign you are driving.
3. **`docs/research/rigor-protocol.md`** — the gates every result must pass. Non-negotiable.
4. **`docs/research/full-acceleration-program-2026-07-19.md`** — the FULL living program
   (P0–P10, every spike, kill criteria). Dispatch hub ticket:
   `program-full-bonsai-acceleration-program-2026-07-19-canonical`.
5. **`ticketsplease show verify-spec-acceleration-routemap`** — measurement ledger +
   refutation archive (2026-07-17 research wave). Ranked-actionable text there is STALE;
   use the program doc above for what to run next.
6. **`metal_notes.md`** — the macOS-27 GPU capture/profile playbook, the Metal 4.1
   capability facts, and the settled kernel case studies (§14, §15).

## The goal (the campaign)

> Continuously push the entire on-device Bonsai/Ternary-27B inference engine — end to end,
> everything inside, outside, and around speculative decode: the target forward and its
> ternary matmuls, the DeltaNet/full-attention and recurrent/KV/conv state handling, the
> drafter and verify, the decode loop, sampling, quantization and weight layout, the memory
> envelope, host/dispatch overhead, and prefill/TTFT — toward the highest sustainable,
> quality-preserving throughput and latency achievable on the M3 (Metal-only; CUDA excluded),
> a moving frontier to maximize over the coming days rather than a fixed number, so passing
> any particular tok/s or latency mark is a checkpoint that funds the next experiment, not
> completion; drive it by working the tracked route-map backlog as a living thing where every
> result reshapes the board — refuted routes closed with their verdict so they never resurface,
> promising ones spawning deeper child tickets and new pathways across any part of the pipeline
> — and treat every candidate, from the literature or our own, as a hypothesis whose reported
> result counts only if its regime (hardware, batch/M, precision, memory- vs compute-bound)
> matches ours; hold the whole campaign to the rigor protocol in which a striking result is
> assumed a measurement artifact until its confounds are cleared, no number becomes a finding
> until it triangulates across black-box wall time, white-box GPU-trace counters, and the
> underlying source/algebra AND its mechanism is proven by intervening to move it, one variable
> changes per experiment with controls run first, and every result is logged with its regime and
> a measured-vs-inferred tag — because the standard is never merely observing that something is
> faster but understanding why, and shipping only what is correct, proven-in-our-regime, and
> mechanistically understood.

## TASK-0 — DONE (2026-07-17): the ruler is fixed and the standings survived it

The harness fixes are landed and the fresh quiet/rotated v2 baseline is set (ticket
`eval-harness-validity-fixes`, closed — full ledger in its comments). Key reconciliation:
the ~+6% chain inflation belonged to the **MiniCPM lane** (generate.rs device chain), never
to the gguf lane the Bonsai standings ride on; the gguf ruler's own (small, mostly anti-spec)
biases are fixed in 02e12c8. **v2 baseline (2026-07-17): plain 14.42 / exact 14.67 / m3 19.18.**
**v3 baseline (2026-07-19, post-F1+Q8_0+defaults): plain 14.47 / exact 15.35 / m1 18.19 / m3 20.09.**
Trap: never set the spec mm2d env on the plain arm — LMBRRR_MM2D_PLANAR=1 at m=1 craters
decode to 5.9 tok/s (planar kernel vs the mv GEMV optimum).

## Hard constraints & operating rules

- **Metal-only.** CUDA offload is out of scope for this campaign (training on Modal/CUDA is
  fine; the inference engine is on-device Metal).
- **Ship-local, run-remote.** Edit/commit on the M4 dev box (macOS 15 — no Metal 4.1, no
  macOS-27 GPU tools). Run ALL benches/captures/experiments on the **M3 referee**: `ssh m3`,
  repo at `~/lmbrrr-work/lmbrrr`, `git pull` to sync. Inference benches run **foreground**;
  training/Modal/remote jobs run **backgrounded** (never block a turn on them).
- **The candle fork** lives on branch `lmbrrr` of `huggingface/candle` (remote `tomsanbear`);
  lmbrrr pins it by `rev` in `Cargo.toml`. Metal kernels go there, then bump the pin.
- **Measurement hygiene** (see rigor protocol): interleave A/B arms (DVFS droop), wait for a
  quiet machine (CPU load + GPU wallpaper contamination), first runs after a kernel change are
  shader-compile transients, `nextest` strips the metal feature — rebuild the release binary.
- **GPU profiling**: `evals/profiling/` (needs a debuggable uv python — see its README);
  `metal_notes.md` for the gpudebug flow. Never `profile run --embed` (deadlocks).
- **Tickets**: `ticketsplease` / `tkt` CLI (`/ticketsplease` skill). Work is git-versioned
  markdown tickets with scopes for conflict-free parallel dispatch. The ticket **comments are
  the results ledger** (append-only, per-route); the epic `rollup` is the dashboard.
- **Environment gotchas**: the M3's non-interactive ssh shell lacks `/opt/homebrew` on PATH
  (use full paths); use `uv run modal` and always `modal run --detach` for long runs; never
  pipe build/test output into grep/head (run raw or `>file`/`tee`); read files before editing.

## Where durable state lives

| What | Where |
|---|---|
| The ranked route map (all acceleration routes + verdicts) | ticket `verify-spec-acceleration-routemap` (EPIC) |
| Rigor gates (the method) | `docs/research/rigor-protocol.md` |
| Metal capture/profile playbook + capability walls + kernel case studies | `metal_notes.md` |
| Measurement/profiling harness (runnable) | `evals/profiling/` |
| Research surveys | `docs/research/` (e.g. `acceleration-frontier-survey.md`, `dspark-verify-weightbound-gemm.md`) |
| Current frontier ops-log (Claude-specific, non-canonical) | `~/.claude-work-2/.../memory/bonsai-acceptance-drive-plan.md` |
| Bring-up plan (architecture, config derivations) | ticket `ternary-bonsai-27b-support` (EPIC), `dspark-bonsai-integration` |

## Current frontier snapshot (as of 2026-07-19; **v3** blessed baseline)

- **v3 baseline (M3, Q8_0 drafter, planar defaults, prose prompt, N=128, 3 rotated reps, spread <0.3%):**
  plain **14.47**, spec exact **15.35** (ids byte-match plain every rep), margin-1.0 **18.19**
  (accept 2.969), margin-3.0 / `--fast` **20.09** (accept 3.448). Ledger:
  `blessed-v3-standings-re-baseline-post-f1-defaults-q8-0`.
- **Default OP (2026-07-19 suite):** soft `--adapt-margin 0,1.5,1,3` → mean **~19.8 tok/s** (+5.3% vs fixed m1),
  mean PPL better than global `--fast`; no class PPL worse than m3. Hard adapt `1,2` REFUTED as default
  (fact/summarize collapse). Escape: `--no-adapt-margin` / `--exact` / `--fast`.
- Lift vs v2 (2026-07-17): plain 14.42→14.47; exact 14.67→15.35; m3 19.18→20.09. Acceptance
  unchanged at m3 (3.448) — pure stack speed (F1 occupancy + Q8_0 propose + defaults).
- Product default = **soft adapt** (`0,1.5,1,3`) → prose N=128 **20.16 tok/s** (≈`--fast` 20.10; suite mean +5.3% vs fixed m1). Fixed m1 via `--no-adapt-margin` = 18.20. Campaign speed OP = `--fast` ~20.1.
- PPL cost at margin-3.0 remains CLASS-DEPENDENT (prose +5.4% / code +4.4% / math +3.9% /
  **factual +21.4%**; peakedness gate REFUTED — trajectory-level; see relaxed-typical-acceptance).
- Round anatomy (m3 arm): wall ~218 ms flat; verify dominates;  accept cap still saturates often.
- Full program: `docs/research/full-acceleration-program-2026-07-19.md`.
- The verify **matmul is settled** — mm2d (`matmul2d` on the packed uint2b operand) is the best
  available at m≤8 on M3 (independently re-confirmed vs MLX's f32-bound qmm, `metal_notes` §15.E).
  The lever is NOT the verify kernel — it is acceptance + verify *structure* + the arch state handling.
- Ranked actionable order (in the epic): width-7 drafter retrain (Modal fused prep+train
  in flight) → Sequoia-DP'd tree with a confidence-placed branch → rollback-free GDN
  masked-solve + lossless wide tree → EAGLE-3 drafter upgrade. Escape hatch (bounded quality
  loss): more relaxed acceptance.

## Board caveat (run this first)

`tkt reconcile` reports ~8 tickets `in-progress` with no work branch (they predate the
tkt/branch convention or are stale). **Treat `in-progress` status skeptically** and reconcile
against the code/`git log` before assuming a ticket reflects reality.
