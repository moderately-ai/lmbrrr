# Metal decode utilization — trace evidence, machinery audit, and the scheduling change list

Date: 2026-07-12 (evening session). Instruments: bounded one-step `.gputrace` (`lmbrrr trace --gpu-capture-step`, tooling landed at 6e5b1b8), Xcode pipeline statistics + encoder counters export, dispatch-level fork bench (`metal_benchmarks`), five targeted code-audit passes over candle's Metal backend, upstream PR archaeology.

## 1. Per-kernel GPU-time budget for one decode step (Xcode pipeline statistics, q4k-mlp-q8 manifest, K5-era binary)

| kernel | % of GPU time | interpretation |
| --- | --- | --- |
| kernel_mul_mv_q4_K_bf16 (141 dispatches) | 36.2% | lm_head + MLP projections — healthy, and q4_K is the fastest possible head format (1.16ms vs q8_0 1.28 / bf16 1.42 measured) |
| kernel_mul_mv_q8_0_bf16 (55) | 27.6% | attention + DeltaNet projections still q8_0 in the deployed mixed manifest — 2.1× the bytes of q4_K for the same weights |
| gated_delta_v2_decode_bf16 | 14.3% | the K5 kernel, on target (~0.5ms) |
| copy_bf16_strided + copy2d + cast_f32 | 9.4% | pure data movement (attribution in §5) |
| rmsnorm_bf16 | 6.6% | norm soup |
| sdpa + epilogue + elementwise | ~5.9% | |

Total step bytes read: 518MB (encoder counters). Bandwidth peaks 370 GiB/s, average far lower.

## 2. The utilization diagnosis (encoder counters + timeline)

- **Active Cores comb**: the GPU drains to zero between dependency clusters *within* encoders, not just at encoder/buffer boundaries. Kernel occupancy during bursts: **8–17%**. Occupancy Manager target pinned high (the hardware wants more resident work); Shader Launch Limiter ≈ 0 (launch is not the constraint); no limiter above ~28% in the mid-stack.
- **One ~0.6–0.7ms hole** (~12% of the traced step) at a command-buffer boundary mid-step: GPU fully idle. Mechanism (code-verified): candle commits a new command buffer every 50 encoder acquisitions (`CANDLE_METAL_COMPUTE_PER_BUFFER`, default 50 → ~11 buffers/step); at each new encoder start the code injects waitForFence on **all** outstanding prior-encoder output fences; crossing a commit/schedule gap this parks the GPU.
- **The mid-stack is dependency-serialized, not machinery-serialized**: m=1 decode is a linear chain of ~500 kernels at 5–20μs each. Nearly every adjacent pair is a true hazard, so the auto-barrier (correctly) fires between almost every dispatch, and an intra-encoder barrier drains the pipeline. The rare stacked (overlapping) dispatches in the timeline are the genuinely independent pairs — the concurrent machinery works; the workload has almost no concurrency to give it.
- The final encoder (lm_head) is the only issue-bound region: IntComplex/InstrThroughput limiters ~59%, launch limiter 54.6%, 196 GB/s — consistent with the q4_K issue-rate diagnosis (SoA layout falsification, +3.7% vs a 1.5–2× gate; see campaign log).

## 3. Machinery audit: what candle already does right (verified against upstream HEAD)

The fork's `candle-metal-kernels/src/metal/commands.rs` is **byte-identical to upstream main** (includes #3595's readback-race fix). Verified properties:

- Encoders are created `computeCommandEncoderWithDispatchType(Concurrent)`; one encoder is reused for up to 50 dispatches.
- All device buffers are `HazardTrackingModeUntracked`; correctness comes from (a) an auto-barrier (`memoryBarrierWithScope(Buffers)`) inserted only when the per-dispatch read/write buffer-set tracking detects a true hazard, and (b) `MTLFence` wait/update across encoders via a global written-buffer→fence map.
- Commits are asynchronous (no host wait at swap); encode-ahead works — the host keeps encoding into fresh buffers while committed ones execute. The only thing that stalls encode-ahead is a readback or synchronize.
- Production greedy already exploits this: the device-chain keeps the sampled id on-GPU, encodes token N+1 while N runs, and drains once per `READBACK_EVERY = 8` tokens. Greedy has **no per-token host block**.
- The dspark round has exactly **two structural full drains** (draft-proposal readback, verify-targets readback); each full drain pays ~1–2ms OS wait-notification latency on top of GPU completion — this is the measured "bare host cost" the cost-model contract carries. The targets drain is fundamental to speculative decoding (round r+1 depends on r's acceptance); the proposal drain is engineerable (on-device chunk assembly + width selection).

## 4. Upstream lineage (the prior art, hunted)

- **#2037** (tomsanbear, Apr 2024, open) "always reuse command encoders/buffers" + companion **#2061**: the original attack. Recorded lessons: perf-neutral on the models of the day; stalled on a stable-diffusion corruption that smelled like a sync race — the characteristic failure mode of this machinery. Its concrete payoff then: dramatically smaller gputraces.
- **#3511** (ivarflakstad, May 2026, merged) "Concurrent dispatching" — intra-encoder parallelism; **#3532** (merged) "Improved inter-encoder sync and gemv" — untracked buffers + input/output dependency tracking + fences + residency sets; measured +20% quantized / 2–2.5× dense at landing. Our base carries both.
- **Gap between #3532's stated design and its code (upstream too)**: the PR says "wait for the minimum amount of required fences", but encoder-start waits on ALL outstanding fences; per-buffer-at-first-touch waiting is a legitimate upstream improvement (modest local value for a single chain).
- **#3467** (open, experimental) "Lazy backend" — op-graph capture aiming at automatic elementwise fusion (<5% capture overhead so far). Philosophically the same conclusion our trace forces: the mid-stack needs fewer, fatter kernels.
- **#3496** (open) "Backend driven sampling" — on-device sampling; lmbrrr's greedy device-chain already implements the private equivalent.

## 5-final. Dispatch census — resolved (wave-2 audit)

Corrections to §1/§5 from the full code-derived census:

- **The Xcode "count" column is NOT a per-step dispatch count.** Mismatches are bidirectional (deltanet core 18 expected vs 98 shown, but badd 48 expected vs 6 shown) — no window/aggregation explains both directions; only kernel PRESENCE and GPU-time percentages are trustworthy. Anchor that proves it: core:epilogue must be 1:1 by construction (one call emits both) and the shown 98:72 is impossible.
- **sdpa-vector provably runs at decode** (`use_sdpa` true for l=1/mask-None; head_dim 256 in the vector kernel's supported set); the softmax-fallback attribution in the first-pass analysis was wrong.
- **The real cast source, and the biggest hidden item of the day: every quantized GEMV pays one `cast_f32_bf16`.** candle's quantized matvec hardcodes an F32 output (`quantized/metal.rs:406`); `MixedLinear::forward` then `.to_dtype(BF16)` (`quantized_linear.rs:68`). ~100–130 casts/step ≈ 0.4–0.6ms of GPU+host+barrier. Fix needs bf16-dst KERNEL variants (the MSL kernels store through `device float*` — an allocation-only change is insufficient), mv+mc scope only (the m≥8 mm routing is head-only and the head must STAY F32 for argmax numerics, as must the drafter heads).
- **The DeltaNet cat emits 3 `copy2d_bf16` per layer (54/step)** — dst_s≠d2 defeats the blit shortcut — but does NOT break the compute encoder (no blits involved). Fix: pass b/a as separate kernel inputs (they are dense-protected and must NOT be row-concatenated into the quantized projection).
- Elementwise census per step: rmsnorm 61, badd 48, bmul 30, silu 24, sigmoid 6, rope 12 (+ partial-rotary split/cat ~24 movement dispatches in the 6 attention layers; fix = rope variant applying rotation over the first rotary_dim=64 lanes in place).

## 7. Instruments & procedures (how to reproduce every measurement here)

- **One-step GPU capture**: `METAL_CAPTURE_ENABLED=1 lmbrrr trace --quantized-manifest <M> --quantize-lm-head q4k --prompt "..." --max-new-tokens 8 --gpu-capture-step 5 --output /dev/null` → `decode-step-5.gputrace` (~6GB — snapshots all resident buffers; one step is the practical max). Open in Xcode, click Profile/Analyze to replay. Read: Summary insights; Counters view (per-track: Active Cores / Occupancy / Limiters / Bandwidth); pipeline statistics for GPU-time % per kernel (trust % + presence, NOT the count column, see §5-final); export encoder counters CSV for scripted analysis (per-encoder bytes/bandwidth/limiters — derive encoder GPU-time ≈ bytes/bandwidth).
- **Dispatch-level kernel benches** (fork `candle-metal-kernels`, `cargo build --release --example metal_benchmarks`): `qmv` (mv GB/s, n-sweep), `qmm` (mc vs mm at m 8/12), `qmv-soa` (layout experiment + bitwise check), `qmv-capture` (bounded .gputrace of the mv kernel + pipeline occupancy stats; needs METAL_CAPTURE_ENABLED=1), `barrier-probe` (independent-chain overlap ratio; 1.0 = barriers serialize). Many dispatches per command buffer to escape the ~1-3ms commit floor; discard first runs (shader compile).
- **In-loop cost refit** (after any kernel/config change touching round costs): run the calibration suite (`run_spec_suite.py --split calibration --per-class 3 --reps 1 --gamma 6 --tag <t>`), then per-l drafted-round residual medians + no-draft median from `round_residual_ms` correct the runtime table; `greedy_step_ms` = runtime `verify_ms[1]` + no-draft median (residuals are measured against the table in the bundle's cost_model.json AT RUN TIME — compound against that, not an older artifact). STS refit from the same reports via `evals/fit_sts.py`.
- **A/B protocol**: rotated arms, ≥3 reps, quiet machine for perf claims (±10% ambient floor; controls bound rig noise ±3%), DIVERGENT-CONTENT rows excluded for same-weights arms; different-weights arms are content-divergent by construction — judge on aggregate + τ + the quality ladder (`quant-quality`).
- **Correctness gates per change**: `cargo nextest run`; stub oracle (`dspark-run` without `--drafter`, corrupt 0/3/5 must be invariant, 0.75 logit-noise bound); `tree-check`; drafter smoke text-compare. Sync-machinery changes additionally re-run the barrier probe and never share a slice with numerics changes (the #2037 stable-diffusion-corruption lesson).
- **Agent dossiers** (full source-cited reports behind this doc): session task outputs for commit machinery, encoder/barrier semantics, copy provenance, quant policy, sync census, host-encode anatomy, dispatch census — key conclusions are inlined above; the upstream PR lineage is §4.

## 5. Copy/cast provenance (first pass — superseded by §5-final; kept for the audit trail)

Candle lowering rules (verified): `cast_f32_bf16` = contiguous F32→BF16 `to_dtype`; `copy_bf16_strided` = non-contiguous same-dtype copy not reducible to uniform 2-D blocks; `copy2d_bf16` = 2-D tile copy where the blit shortcut (`src_s==d2 && dst_s==d2`) fails; contiguous copies go to **blit encoders** (uncounted in compute stats, and each blit **force-ends the compute encoder** — a pipelining cost of its own).

Attributed sites (per step): DeltaNet `cat([qkvz|b|a])` — 3 cats × 18 layers, the single largest avoidable source (fix: pass the three buffers to the kernel separately; the kernel already assumes the packed layout); KV-cache `slice_set` appends (2 × 6 layers, halvable by packing K/V in one cache tensor); q-gate-split norm copies (6); rope `contiguous()` (≤12, fixable by fusing rope into the projection reshape); `offset0()` materializations are no-ops in steady state. **Open contradiction being resolved by wave 2**: the cat arithmetic predicts ~54 copy2d dispatches vs 16 observed, and 6 `cast_f32_bf16` co-exist with sdpa-vector dispatches (the softmax-fallback attribution can't be right as stated) — the corrected census lands in this doc when the agent returns.

## 5b. Wave-2 findings (scoped barriers, host anatomy, commit pacing)

**Scoped barriers: falsified by probe + vendor text + MLX precedent.** The Apple header for `memoryBarrierWithResources` says the barrier "ensures that ALL dispatches in the encoder have completed execution" — scope narrows only the cache-flush set, not the scheduling sync. MLX (whose CommandEncoder candle's is a near-verbatim port of) never uses resource-scoped barriers; its extra trick is `ConcurrentContext` — an RAII region that suppresses barriers *within* a group of known-independent dispatches, paying one global barrier at the boundary. Our empirical probe (fork `metal_benchmarks barrier-probe`: independent q4_K GEMV chains interleaved in one concurrent encoder) measured interleaved/sequential ratios of **0.85–1.00** against a full-overlap bound of 0.25–0.50 — global barriers serialize essentially everything, and there is no headroom for scoping to recover on a chain workload. Verdict: park scoped barriers; note ConcurrentContext for genuinely parallel future workloads (tree verify, batch).

**Host encode anatomy (per-dispatch, ~2.8μs):** dominant costs are 6–15 objc msg_sends (worst: gemv's ~15 set_bytes), ~7 lock acquisitions (the `pipelines` cache takes a WRITE lock on every hit; `EncoderState` mutex 4×/dispatch), and 2–3 heap Strings for kernel names (binary/gemv build names via `format!` every call). Ranked cheap fixes (R-list, est. −1.0–1.6μs/op combined ≈ −0.5–0.8ms/step): R1 static kernel names, R2 read-lock pipeline cache, R3 call-site pipeline handles, R5 collapse the 4 EncoderState locks into the dispatch. Backprop recording is already free in inference (no Var → BackpropOp(None)). These are upstream-worthy candle patches.

**Commit pacing, not buffer size.** `CANDLE_METAL_COMPUTE_PER_BUFFER=1000` measured WORSE than the default 50 (287.4±9.8 vs 305.1±1.6 steady, q4full manifest — high ambient load, re-verify quiet, but the sign is mechanistically explained): with a giant buffer nothing commits until the every-8-token readback, so the GPU starves behind the encoder. The default 50 paces commits. The boundary-hole fix is eager pacing (explicit flush per token, or `enqueue()` early — `enqueue` exists unused in `command_buffer.rs`), possibly a SMALLER cap; to be measured quiet.

**Manifest flip (change #1): SHIPPED.** 6-class validation: mean 161.0 → 179.3 (+11.4%), math 259.6 (τ 3.58), τ up in 4/6 classes; truthful refit on the new target: realized in-loop greedy 5.09 → **4.03ms** (+26% floor), STS pos-0 accept 0.723 → 0.862 (fakequant match confirmed); bundle sts.json + cost_model.json (spec-round-cost-model-r4q4f) + suite defaults updated. Bench-mode greedy at defaults reached ~305 tok/s (load-suspect, quiet re-check pending).

## 6. The change list (ranked, with decision experiments)

| # | change | layer | expected | status |
| --- | --- | --- | --- | --- |
| 1 | Spec-lane manifest swap to `q4k-full-text` (attention+DeltaNet q8→q4_K; drafter rounds 2–4 are fakequant-matched to exactly this policy — the q8-mix objection died with round-1) | config | ~−0.4ms/step + possible τ gain; first A/B: math 259.5±6.3 (record, τ 3.58) vs 228.2 shipped; coding wash | **SHIPPED** (6-class mean +11.4%) |
| 2 | Command-buffer cadence: `CANDLE_METAL_COMPUTE_PER_BUFFER` | env | kills the 0.6ms boundary hole | **MEASURED quiet: knee at CPB=100, +3–4% vs 50; ≥150 declines** — default flip = first wave-3 slice |
| 3 | DeltaNet cat elimination + rope-in-place + bf16-dst GEMV (wave 1; the census's cast fix folded in) | fork+lmbrrr | ~−0.3ms + comb teeth removed | **SHIPPED (fork 8ebbebd8, bit-preserving): package +12.5–15.2% steady greedy; in-loop greedy 4.03→3.65ms** |
| 4 | Resource-scoped barriers (`memoryBarrierWithResources`) in candle's auto_barrier | fork | unknown — hinges on Apple's scoped-barrier semantics | decision experiment: two independent GEMV chains interleaved, global vs scoped vs none (bench task to build) |
| 5 | On-device dspark chunk assembly (kill 1 of 2 per-round drains) | lmbrrr | ~−1–2ms per drafted round of OS wait latency | design after 1–3 land |
| 6 | Encoder-start per-buffer fence waits (upstream tidiness per #3532's own design) | fork | small | opportunistic, PR-worthy |
| 7 | Elementwise fusion of the norm/add/silu soup | fork+lmbrrr | the long-term comb fix (#3467 direction) | after 3 |

Risk discipline: every encoding/barrier change re-rolls tie-flips and risks silent race corruption (the #2037 lesson) — each gates through the trajectory-invariance oracle, tree-check, and text-compare before any perf reading, per the standing measurement protocol.

## 8. Drafter-propose attribution wave (2026-07-13) — two falsifications, the real mechanism, and the instrument fixes

The spec lane's propose bucket went through three attribution rounds in one day; the corrected mechanism and the instrument lessons supersede any earlier per-dispatch pricing.

- **Instrument fix 1 (verify-drain contamination):** the fenced propose ladder's first timer armed without draining in-flight work; under async readback the previous round's verify tail drained into the backbone fence, producing a DETERMINISTIC 6.7-vs-10.1ms bimodality (same round positions across 8 runs — greedy acceptance is deterministic). Survived two wrong hypotheses (ctx growth, CPB commits — CPB 100 vs 4096 A/B: identical lows, highs slightly WORSE at 4096). Fixed: synchronize before arming. Post-fix backbone reads a tight **6.20ms**, and fenced-sum ≈ in-loop (9.8–10.0ms): propose is fully serial, no hidden overlap.
- **Falsification 1 (dispatch diet, backbone):** width-fusing q/k/v and gate/up + add-norm/swiglu depth fusions moved e2e ZERO. Width siblings have no hazard barrier between them (already concurrent, same total bytes); the removed elementwise stages were cheap. The "~0.15ms/dispatch floor" was an average mistaken for a marginal cost — dossier §1's 5–20µs/kernel had already contradicted it.
- **Falsification 2 (dispatch boundaries, chain):** halving the chain's dispatches (single-dispatch step via packed 32-bit atomic argmax, fork 3d69e89a — bitwise-verified, goldens byte-identical across the pin bump) moved the fenced chain segment ZERO (3.43ms at both 12 and 6 dispatches). Boundary cost is ~free here too.
- **The real chain mechanism: lone-dispatch execution latency.** 573µs/step in-situ vs 78µs/step isolated = ~16GB/s effective on a ~150GB/s part; the kernel read q8 quants as 32 scalar byte loads per block at 2 latency-serialized rows/thread. The isolated bench hid it by pipelining 64 chains per command buffer — **lone-dispatch latency never shows in deep-pipelined micro benches; bench serially-dependent production dispatches at pipeline depth 1.** Fix shipped (fork f2c7fa82): packed_char4/bfloat4 vector loads (strict sequential addition order — the bitwise CPU-reference gate holds) + 128 TGs (one row/thread).
- **Pricing rule going forward:** per-stage dispatch costs are heterogeneous — width-parallel siblings ≈ 0, small elementwise ≈ cheap, occupancy-collapsed or latency-bound executions carry the wall. Attribution order before building anything: fenced ladder → per-kernel isolated-vs-in-situ ratio → build.
- **Parallel lane:** the transplanted Qwen3.5 MTP head (`--drafter-mtp`, untuned, 15 tensors from the base checkpoint) drafts at mean accepted length 2.03/round (depth 3, math) with the committed stream EXACTLY equal to greedy — draft cost is ~3 one-layer forwards/round vs the block drafter's ~10ms, making the steep verify slope (verify_ms[2]=9.8 vs greedy 5.5) the shared wall for every spec lane.
