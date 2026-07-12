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

## 5. Copy/cast provenance (first pass; wave-2 audit in flight)

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
| 1 | Spec-lane manifest swap to `q4k-full-text` (attention+DeltaNet q8→q4_K; drafter rounds 2–4 are fakequant-matched to exactly this policy — the q8-mix objection died with round-1) | config | ~−0.4ms/step + possible τ gain; first A/B: math 259.5±6.3 (record, τ 3.58) vs 228.2 shipped; coding wash | 6-class validation A/B RUNNING |
| 2 | Command-buffer cadence: raise `CANDLE_METAL_COMPUTE_PER_BUFFER` (50 → step-sized) | env | kills the 0.6ms boundary hole | queued behind (1) |
| 3 | DeltaNet cat elimination + KV pack + rope-in-place (fewer copies ⇒ fewer barriers ⇒ fewer drains) | lmbrrr | ~−0.3ms + comb teeth removed | after wave-2 census |
| 4 | Resource-scoped barriers (`memoryBarrierWithResources`) in candle's auto_barrier | fork | unknown — hinges on Apple's scoped-barrier semantics | decision experiment: two independent GEMV chains interleaved, global vs scoped vs none (bench task to build) |
| 5 | On-device dspark chunk assembly (kill 1 of 2 per-round drains) | lmbrrr | ~−1–2ms per drafted round of OS wait latency | design after 1–3 land |
| 6 | Encoder-start per-buffer fence waits (upstream tidiness per #3532's own design) | fork | small | opportunistic, PR-worthy |
| 7 | Elementwise fusion of the norm/add/silu soup | fork+lmbrrr | the long-term comb fix (#3467 direction) | after 3 |

Risk discipline: every encoding/barrier change re-rolls tie-flips and risks silent race corruption (the #2037 lesson) — each gates through the trajectory-invariance oracle, tree-check, and text-compare before any perf reading, per the standing measurement protocol.
