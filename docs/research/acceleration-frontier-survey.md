# Acceleration frontier survey — beyond the DSpark program (2026-07-11)

Four parallel research agents swept the space (output head, decoding methods, Metal utilization, quantization/sparsity), each briefed with our measured numbers. This synthesis ranks their findings into a program; full agent reports are preserved in the session transcript, sources cited inline on the tickets.

## Tier 1 — dominant levers

**1. Re-grid the fused DeltaNet decode kernel (new p1).** The agent read our fork's kernel and found it launches 16 threadgroups (2,048 threads) on a 32-core GPU — half the machine idle, state slab read twice, ~19 GB/s effective on a ~2 MB working set. MLX's kernel for the same op uses grid (32, dv, B·heads) = 512 threadgroups with simd-cooperative dk reduction and a single state read. Projection: the 2.8 ms DeltaNet block → 0.3–0.5 ms, i.e. **single-stream 235 → ~520–600 tok/s**, which is 65–72% of roofline — at or above MLX/llama.cpp-class utilization for this size. Calibration: best-in-class 1–2B models run at 50–70% of roofline (fixed per-token costs), so ~830 tok/s is a hard ceiling and ~550 is the honest target. Parity gates and the decode oracle de-risk the rewrite.

**2. Draft-free n-gram speculation: prompt-lookup + verify-logit token recycling (new p1).** Chain-mode drafts from (a) suffix matches against prompt+history and (b) an adjacency matrix of top-k candidates harvested from verify logits we already compute. Zero training, zero new kernels, CPU-side lookup, drafts fire only on match → strict greedy floor, no probe tax. Published: up to 2.8× on summarization/code/grounded tasks — exactly the classes where DSpark τ collapses (1.25–1.45). Composes with DSpark under the existing scheduler (RASD pattern: per-round mux between n-gram, drafter, greedy).

## Tier 2 — cheap, proven, composing

**3. FR-Spec drafter-vocabulary trimming (spec lane, ~1–2 days).** The drafter's 248k head is ~half its cost; drafting from the top-32k frequency tokens is provably output-identical (verify keeps the full head). llama.cpp has a byte-identical reference (−85% draft-head time).
**4. imatrix calibration pipeline + mixed 3-bit policy (~days).** The fork already ships the imatrix loader and `from_float_imatrix` for k-quants; only activation-accumulation hooks are missing. imatrix-q4k first — free quality at identical bytes, plausibly recovers part of the τ 2.13→1.69 drafter-acceptance loss — then mixed q3_K (+4–5%) and full q3-class (+~10%) as reported experiments. Floor: 3-bit class; q2/IQ2 collapses at 1B (measured in literature).
**5. MTP head drafting off the verify pass (decision after round-2 ablation).** ~$20–60 on the round-2 traces; τ≈1.8–2.2 uniformly at near-zero draft latency (DeepSeek-measured 85–90% +1 acceptance); head projections batch through our multi-column kernels. If round-2 DSpark τ disappoints, this becomes the drafter; if it succeeds, it still adds free +1 extension.
**6. GEMV width + split-K + barrier-level concurrency (extends kernel lane).** Concatenate gate+up and q/k/v/gate projections (own data: 6144-row GEMV runs 2.2× the bandwidth of 3584-row); MLX-style `qmv_split_k` for skinny shapes; llama.cpp-style hazard-tracked concurrent encoding (+8–12% and +5–8%; note our CB falsification tested commit granularity, not barrier granularity).
**7. BF16 recurrent-state storage (f32 accumulate) (~days).** 36 MB/token of f32 state traffic → ~+2–3%. Quamba shows even int8 SSM state survives; gate with a generation-length drift sweep (the 2026-07-07 null result predates the fused kernels — bytes now show).

## Tier 3 — conditional

**8. Certified-exact sub-vocabulary head (CSV-Decode class).** Bit-exact greedy via cluster upper bounds (<2% fallback, ~18% of vocab scored); our tied anisotropic embedding tightens the bounds. BUT: our lm_head GEMV already runs at ~85% of roofline, so the prize today is ~+7–16%, growing to ~1.3× only in the endgame. Gate: measure the isolated q4_K head GEMV first (10 minutes) before committing 1–2 weeks.
**9. KV q8_0 + Quest-style page selection — only if ≥16k contexts become a workload.** ~0% at 4k; +8–25% at 16k; the difference between unusable and ~200 tok/s at 262k. Parked until long context matters.

## Recorded negatives (do not revisit without new evidence)

- **Self-speculative/layer-skip/early-exit drafting: dead on this architecture.** Measured on Qwen3.5-0.8B (our exact 18+6 hybrid): acceptance 0.038–0.233, 0.000 for early-exit (arXiv 2605.01106). The sequential GDN/attention interleave is the cause; scale-invariant.
- Lookahead/Jacobi window decoding: structurally incompatible with the recurrent chunk scan. CLLM/diffusion-style: retrains the target, violates exact-greedy.
- Weight sparsity (all forms): no Metal sparse story; quality-per-byte dominated by sub-4-bit quant at 1B.
- ANE/AMX offload of the head: ANE effective bandwidth <100 GB/s and on the critical path; AMX ~3× slower than GPU for the shape. Power play, not latency play.
- Weight duplication / dequant caches / prefetch: unified memory — no bandwidth to gain, dequant caches trade scarce bandwidth for abundant capacity, backwards.
- 350 GB/s is final (85–90% of spec is the empirical M-series ceiling; llama.cpp's best matches it).

## Cross-cutting observations

- The small-m verify gap (l≤12 at 1.5–1.7× a decode step vs theoretical ~1.05×) remains the universal multiplier under every speculation method; llama.cpp's Metal MTP net-loss is the cautionary tale.
- Quantized aggregate (860) vs BF16 aggregate (1530): the multi-column kernels don't batch-scale — separate ticket-worthy finding for the aggregate lane.
- Composition estimate if Tier 1+2 land alongside a successful round-2 drafter: kernel re-grid (~2.3×) × n-gram/MTP/DSpark speculation on cheap verify (1.3–2×) puts **single-stream 4-digit tok/s within the roofline math** (830 × τ_eff) for structured domains — the campaign's stated end goal.
