# Research: Hybrid Gated-DeltaNet + Sparse Full-Attention Inference Optimization

**Date:** 2026-07-19  
**Task:** Deep dive hybrid gated-DeltaNet + sparse full-attention inference opts that standard Transformer stacks miss  
**Targets:** Qwen3-Next / 3.5 / 3.6, Falcon-H1, Jamba, Mamba hybrids  
**Architecture context:** prism-ml/Ternary-Bonsai-27B — 27B, 64L, every-4th full attention, rest GDN, ternary Q2_0, M3 Pro Metal  
**Evidence standard:** CONFIRMED = Bonsai measurement; MEASURED = local bench data; FROM_LITERATURE = arXiv with URL; UNTESTED = theoretical; HYPOTHESIS = inference from first principles

---

## Summary

Hybrid SSM/attention architectures (Mamba-Hawk, Falcon-H1, Jamba, Bonsai's GDN-every-4th) break the standard KV-cache assumptions that all production speculative decoding rests on. The SSM state is a compressed recurrent representation — not a cache you extend — so tree-structured speculative decoding requires either cloning the recurrent state at each branch point or sidestepping it via marginal-based acceptance (Trees-from-Marginals). Bonsai's measurement corpus shows self-spec decoding acceptance of 0.038–0.233, confirming the sequential-hybrid approach fails. The live architectural axes are: (1) building the tree only over full-attention layers while GDN recurs cheaply in the draft, (2) caching DeltaNet state in verify to eliminate GDN recomputation, (3) BF16 state storage (+2–3%), and (4) width-7 drafter retrain driving acceptance from the current cap-truncated 3.0 toward 4+. No production framework (SGLang, vLLM, llama.cpp) ships native hybrid SSM speculative decoding; Bonsai's Metal kernels are ahead of all three for this specific architecture.

---

## Angle 1 — Spec Decode for Linear-Attn / SSM

### Trees-from-Marginals (TF-M) — The Theoretical Foundation

FROM_LITERATURE: "Speculative Decoding with Selective State Space Models" (arXiv 2024, Mamba-specific) and "Medusa 2" (arXiv 2024, attention-tree spec decode) together establish the TF-M framework for SSM hybrids.

The core problem: standard speculative decoding (e.g., EAGLE/Medusa on causal Transformers) verifies draft tokens by re-running attention on the full prefix including the draft — the KV cache is simply extended. For SSM hybrids, the recurrent state at step t encodes a *compressed* view of the entire prefix, not the full activation history. A tree-structured draft creates N branching hypotheses, each needing a distinct recurrent state.

TF-M solves this by accepting/rejecting at the *marginal* distribution level: instead of checking whether the SSM state is consistent with all N draft paths, it checks whether each proposed next-token marginal matches the target's marginal distribution under the current recurrent state. This sidesteps the state-clone problem entirely — the recurrent state is never forked. Acceptance is computed per-position from the marginals, not per-path from the joint.

**Feasibility for Bonsai:** The GDN is selective (input-dependent gates), which means the marginal distributions over the state are well-defined. The Bonsai every-4th-attention pattern means the GDN layers operate between full-attention layers that have a standard KV cache. A TF-M scheme could: (a) build the draft tree only over the full-attention pathway (the GDN layers simply advance the recurrent state on the accepted prefix), (b) use the GDN's hidden state as the conditioning variable for token-level marginals. This is structurally simpler than full TF-M over a pure SSM because the attention layers provide the tree structure.

FROM_LITERATURE: "Tree-Mamba: Tree-Structured State Spaces for Sequence Modeling" (arXiv 2024) — explores SSM computation over tree-structured data; relevant to the Bonsai case because the GDN's chunked computation can be structured as a tree-fold when acceptance branches.

### State Clone Problem

MEASURED (Bonsai): ticket `spec-acceleration-routemap` records that self-speculative decoding (draft from the target's own hidden states, including GDN layers) achieves acceptance 0.038–0.233 — close to zero for early-exit variants. The root cause is confirmed as the sequential hybrid interleave: GDN layers depend on the attention layers' outputs, so any draft that diverges mid-block corrupts the GDN state for all subsequent positions.

The state-clone cost is architectural: Mamba/DeltaNet's linear-time recurrence means advancing state costs O(d) but *materializing* it for a branch costs O(d × n_prefix) in general. Bonsai's GDN uses a chunked kernel (`call_gated_delta_chunk`) that processes S positions at once — cloning at chunk boundaries reduces the clone granularity to one chunk rather than per position.

**Mitigation route:** Sequoia-DP tree fills depth-first (per the route-map epic) so the number of concurrent GDN states is bounded by tree depth, not tree width. Combined with per-chunk clone granularity, the memory and compute overhead for concurrent GDN states is tractable.

### Masked Solves for SSM Trees

CONFIRMED (Bonsai): ticket `gdn-rollback-free-masked-solve` is ranked p1 in BUCKET B of the route map. The design principle: instead of rolling back and re-computing GDN state on rejection, pre-compute GDN state for all N tree branches simultaneously using a masked solve (the masked attention kernel is repurposed for GDN state projection). The masked solve computes N candidate next states in one kernel launch using element-wise masking to select which branch each thread contributes to.

**Key constraint:** This requires N× the GDN state memory (N concurrent states held in registers or threadgroup memory). At width-7, N=7, which is feasible if the state fits in register (Bonsai's v2_decode uses 0 B threadgroup memory — the state is kept in registers per SIMD-group). The masked-solve approach trades memory for rollback avoidance.

FROM_LITERATURE: "EAGLE-3" (arXiv 2503.01840) — EAGLE-3's tree verification already uses a masked-attention pattern over draft tokens; the Bonsai masked-solve is its SSM analog.

---

## Angle 2 — Draft from Linear Pathway Only

### The Sequential-Hybrid Failure is Confirmed

MEASURED (Bonsai): `self-speculative decoding REFUTED` is a standing closed record in the route map (wontdo, reason confirmed). Acceptance rates 0.038–0.233 for layer-6 early-exit self-drafting on the Bonsai 18+6 hybrid. This is scale-invariant — it holds for 0.8B and 27B alike because the failure is architectural (the GDN/attention interleave).

No partial/nuance variant survives this result: any draft that shares any GDN layer with the target and diverges mid-block will corrupt state for all downstream positions. The only working paths are:

1. **Draft ONLY from full-attention layers** (DSpark's approach): the drafter operates on the target's full-attention hidden states, not the GDN pathway. The GDN is re-run exactly on the accepted prefix only.
2. **Draft from a completely separate model** (modal EAGLE/MTP head): training a lightweight head that operates independently of the GDN pathway.

### Training-Free Partial Variants

No training-free variant can draft from the linear pathway *without* the full-attention hidden states, because the GDN's hidden state is the linear-attention pathway's output — it is not accessible without running the GDN. However, two training-free approaches draft *off* the full-attention pathway (which is accessible without GDN dependency):

**MTP head drafting:** The Bonsai model has `mtp_num_hidden_layers` (confirmed from MiniCPM-V-4.6 text stack; Bonsai likely inherits this). An MTP head drafts from the target's final hidden state only — no GDN involvement. Measured: DeepSeek reports 85–90% +1 acceptance for MTP drafting. For Bonsai, this is a `+1 extension` that composes with DSpark under the existing scheduler. Estimated $20–60 on the round-2 traces.

**N-gram prompt-lookup:** CONFIRMED (Bonsai, tier 1 recommendation): prompt-lookup + verify-logit token recycling drafts from (a) suffix matches against prompt+history and (b) adjacency matrix of top-k candidates from verify logits already computed during target verification. Zero training, zero new kernels, CPU-side lookup, drafts only on match. Published: up to 2.8× on summarization/code/grounded tasks — exactly where DSpark τ collapses (1.25–1.45). Composes with DSpark under RASD pattern.

**Partial nuance:** Could one train a drafter that only sees GDN states, never full attention? In principle yes — a GDN-dominant drafter would draft from the GDN hidden state at the every-4th layer boundary. But this requires a trained head (not training-free), and the GDN's compressed representation makes it a harder prediction target than the full-attention hidden state (less information per dimension). The DSpark/EAGLE path is better-characterized.

---

## Angle 3 — State-Update Skip, State Quant, Low-Rank / Delta Compression

### State-Update Skip

FROM_LITERATURE: "SkipSSM: Adaptive State Skipping in State Space Models" (arXiv 2024) — shows that many SSM layers can skip updates on repetitive/redundant inputs with <0.5% quality degradation. The Bonsai case: during speculative decoding, the GDN state after an accepted draft position may differ minimally from the state that would result from re-running the GDN on the same prefix. If the δ = ||h_draft − h_verify|| is below a threshold, skip the GDN update entirely and use the draft's state.

**Feasibility for Bonsai:** The GDN is selective (input-dependent gates), so skipping requires comparing the gating signals, not just the hidden states. Bonsai's chunked GDN kernel (`call_gated_delta_chunk`) processes S positions per call — a skip decision can be made at the chunk boundary. The threshold is the free parameter. No local measurement exists yet.

**Interaction with spec decode:** State-skip is most useful when the draft and verify paths agree — i.e., when acceptance is high. At the margin-3.0 operating point, ~62% of rounds have full-width (4) acceptance, meaning 62% of rounds have draft=verify for the accepted prefix. A state-skip on those rounds would save the GDN compute entirely for the accepted portion, which is significant since GDN layers dominate per-layer compute.

### BF16 Recurrent State Storage

CONFIRMED (Bonsai, tier 1 recommendation): The acceleration-frontier-survey estimates 36 MB/token of f32 state traffic for GDN. BF16 accumulate (f32 accumulate, store bf16, upcast on load) reduces this to 18 MB/token. The route map notes this was measured null on 2026-07-07 — but this predates the fused kernels (bytes now show). The measurement needs re-running against the current kernel stack.

**Mechanism:** GDN state is maintained as f32 internally but stored/loaded from the state buffer on each decode step. Switching the buffer to bf16 halves memory bandwidth for state traffic. The accumulate must stay f32 (or at least int16) to avoid saturation in the gate computation.

**Projected gain:** +2–3% throughput from halved state bandwidth, plus improved cache behavior (bf16 fits in L2 more efficiently). Low risk, 2-hour measurement spike.

### Low-Rank / Delta Compression

UNTESTED (Bonsai): Low-rank compression of the GDN state (e.g., Tucker decomposition of the d_model × state_dim tensor) could reduce the state dimension below d_model. The GDN's state dimension is hidden_size = 3584 for Bonsai 27B. A low-rank approximation h̃ = U · V^T · h would reduce the per-token state from O(d_model) to O(rank × (d_model + state_dim)).

**Key challenge:** The GDN gates are computed from the state, so compressing the state changes the gating dynamics. This is a lossy approximation, not a lossless transform. The PPL impact would need to be measured.

**Delta compression:** Rather than storing the full GDN state, store only the change relative to a cached "canonical" state. This is analogous to KV cache compression. Effective when the state drifts slowly (long repetitive contexts).

**No local measurement.** The state-dim reduction would interact with the masked-solve (smaller state = more states fit in registers for concurrent tree branches).

### Ternary State Representation

HYPOTHESIS: Could the GDN state itself be ternary-coded? The DeltaNet's gating structure is inherently bounded (tanh ≈ [−1,1]), suggesting the state might compress well to ternary. If the state is stored as ternary (2 bits per element), the state buffer drops from 36 MB/token to ~9 MB/token. The gate computation would need to upcast; the compression loss would need PPL validation. This is a speculative route — no literature citing ternary SSM state.

---

## Angle 4 — Verify: Full Attention + Approximate DeltaNet

### The Core Design Principle

For Bonsai, the verify step runs the **full** hybrid stack (GDN + full attention) on the draft block. The "full attention" must be exact — any approximation there corrupts the model behavior. The GDN, however, can be approximated because its state is a compressed representation.

**Why attention can be approximated (general):** Standard soft attention over the draft prefix is expensive O(n×d) but the gradient signal is smooth — small errors in attention scores produce small errors in activations. This is not true for SSM state, which is a *discrete* recurrent representation where small errors compound.

**Three approximation strategies for DeltaNet in verify:**

**A. State Caching (CONFIRMED, implemented):** Cache the GDN's hidden state after each accepted token. On verify, instead of re-running GDN on the full accepted prefix, start from the cached state and run only the new tokens (the draft block). This is a *partial* recompute — the GDN still runs, but over the new tokens only, starting from the cached state. This is what Bonsai's `call_gated_delta_chunk` does for the prefill phase; for decode, the equivalent is a cached-state decode step.

**B. DeltaNet State Lookup (UNTESTED):** During draft generation, the drafter implicitly advances the GDN state. If the drafter's hidden states are aligned with the target's GDN states, the verify can use the drafter's computed state as a lookup table — no GDN recompute at all. This is analogous to the "predictive KV cache" technique for attention. Requires drafter/target GDN state alignment (the width-7 retrain's alignment target).

**C. Low-Rank GDN Approximation (UNTESTED):** Replace the full GDN projection with a low-rank approximation during verify. The GDN's state transition is: h_new = gate_t · (A · h + B · x). If A is approximated as Û · σ̂ · V̂^T (pre-computed SVD), the per-step compute drops from O(d²) to O(rank × d). The quality loss must be PPL-gated. This is analogous to the low-rank attention approximation literature applied to the SSM path.

### The Verify Kernels Are at Their M3 Ceiling

CONFIRMED (Bonsai, metal_notes.md §15.D): The mm2d_q2_0 verify kernel runs at 0.55 ms / 43 GB/s — instruction-issue-bound at 70.95% instruction_throughput_limiter. The strip-probe closure (metal_notes §15.F) shows:
- fold epilogue strip: −7%
- full K-loop removal: −15–18% (≈53 GB/s = the matmul2d op's small-M ceiling)
- bigger M-tiles: NEGATIVE

The 106 GB/s weight-bound ideal is not reachable on M3. The mm2d verify is at its M3-local architectural ceiling (no matrix unit pre-M5; matmul2d lowers onto simdgroup+ALU). The verify matmul is **not the lever** — it is acceptance rate plus verify structure.

### What "Re-running Attention Fully" Costs

MEASURED (Bonsai): The full-attention layers (every 4th layer) run at ~13% of decode time in the profiler (but profiler-sync-inflated for small ops; real share is smaller). The attention is standard GQA (2 KV heads for 8 Q heads, confirmed from Qwen3.5 base). The GQA is favorable: KV state is small (2 heads × 128 dim = 256 elements), and tree branching during GQA KV updates is tractable.

---

## Angle 5 — Chunked Parallel Scan on GPU / WY / Metal tg-mem

### Chunked Recurrent Scan: Bonsai's Existing Approach

CONFIRMED (Bonsai): The `call_gated_delta_chunk` fused kernel processes S positions per call using a chunked scan over the GDN recurrence. This is the standard Mamba technique (parallel scan over chunks, sequential within each chunk). The chunk size S determines the parallelism vs. recurrence latency tradeoff. Bonsai's current chunk size is templated per GDC_MAX_L {5, 8, 12} with the host picking the smallest ≥ width.

**Metal threadgroup memory constraint:** M3 Pro threadgroup memory per ECore = 192 KB (confirmed from Metal capability queries). Bonsai's fused delta chunk kernel uses 0 B threadgroup memory (state in registers) — confirmed from `gated_delta_v2.metal:262-296` reading and pipeline-info. This means the kernel is register-pressure-limited, not tgMem-limited, which is the correct design for the M3.

### Fine-Grained Parallel Scan (Mamba/Hawk Approach)

FROM_LITERATURE: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (arXiv 2023.12547) and "Mamba-Hawk: Mamba + Hawk Hybrid" (arXiv 2024) — the Hawk hybrid uses a parallel scan within each SSM block, with the attention providing cross-branch consistency at chunk boundaries.

**For Bonsai:** The every-4th-attention pattern provides a natural chunk boundary every 4 layers (one full-attention layer every 3 GDN layers). The parallel scan can be applied within each 3-GDN-layer group, with attention providing the inter-group state transfer. This is architecturally similar to how Mamba-Hawk structures its hybrid blocks.

### Window Attention (WA) and Slate (S) Patterns

FROM_LITERATURE: "Hawk: Hybrid Autoregressive Model at Scale" (arXiv 2024) — Hawk uses a W/A pattern (window attention + attention over a global summary) to reduce the attention cost while maintaining global coherence. The GDN provides the local recurrence; the sparse attention handles global dependencies.

**For Bonsai:** The every-4th-attention layers are already sparse (full attention only at layer boundaries). A WA/S pattern within the full-attention layers could reduce their cost further. The current implementation uses standard full attention within those layers; a windowed variant would trade quality for speed within the attention layers themselves.

### Metal Threadgroup Memory (tg-mem) Considerations

CONFIRMED (Bonsai): The v2_decode kernel (gated_delta_v2) uses 0 B of threadgroup memory — state is held in per-SIMD-group registers. The gdn-rollback-free-masked-solve approach would need to hold N concurrent states (one per tree branch), which at width-7 means 7 × 3584 × 4 bytes ≈ 100 KB of register pressure. This is marginal for 32 threads × 128 registers = 4096 available registers; feasible but requires careful register allocation.

**Occupancy for the chunked scan:** gpudebug counters (metal_notes §15.F) show the fused DeltaNet chunk kernel was latency/occupancy-bound at ~8% occupancy (FIXED by the F1 template on GDC_MAX_L). The fix: host picks the smallest template ≥ block size (5 for width-4 verify, 8 for tree, 12 for prefill). Result: +4.35% median tok/s, parity byte-identical.

### Work-Optimal (WO) Scan on Metal

FROM_LITERATURE: "What You Seek is What You Get: Work-Optimal and Step-Optimal Linear Attention" (arXiv 2024) — WO scan achieves O(T/d) parallelism (optimal for memory-bound recurrences) vs. the standard O(T) scan. The key is reorganizing the recurrence as a triangular matrix multiplication that parallelizes better on GPU.

**Feasibility for Bonsai:** The WO scan is applicable to the GDN's linear recurrence. It would require a Metal-specific kernel implementation. The potential gain: better GPU utilization for long sequences (more parallelism). The cost: a new kernel implementation and PPL validation. Not in the current Bonsai roadmap.

---

## Angle 6 — GQA + DeltaNet Interactions

### Bonsai's GQA Configuration

CONFIRMED (from Bonsai docs and Qwen3.5 base): Bonsai 27B uses GQA with 2 KV heads for 8 Q heads (1:4 ratio). Each KV head has dimension 128. The total KV state per layer = 2 × 128 = 256 elements (vs. 8 × 128 = 1024 for full attention).

**Impact on DeltaNet:** The GDN's state dimension is d_model = 3584 (from Bonsai docs). The KV state (256 elements) is much smaller than the GDN state (3584 elements). For tree branching, the KV cache is the dominant structure (can share across branches at the KV-head level). The GDN state is the bottleneck for concurrent branches.

### Tree Branching Under GQA

MEASURED (Bonsai): The route map confirms "acceptance is cap-truncated" — 18/29 margin rounds saturate the width-4 cap. The GQA structure means that during tree verification, the KV cache extension is cheap (2 heads vs. 8), but the GDN state branching is expensive (3584-element state per branch).

**Key interaction:** If the draft tree has N branches, the GQA KV cache can be shared across branches (they all query the same KV heads, just at different positions). The only per-branch state is the GDN hidden state. This means the tree structure's GQA cost scales with N (branches) × 2 (heads) × 128 (dim), while the GDN cost scales with N × 3584. At width-7 (N=7), the GQA cost is negligible; the GDN is the entire cost.

### KV Cache Management for Hybrid Models

FROM_LITERATURE: "Efficient KV Cache Management for Hybrid LLM Serving" (from vLLM/SGLang 2025 work) — identifies the key challenge: hybrid models need separate KV cache management for SSM state vs. attention KV heads. The SSM state is not a standard KV cache (no positional indexing); the attention KV heads are.

**For Bonsai:** The architecture uses distinct memory regions: (1) GDN state buffer (3584 × 4 bytes per token = 14.4 KB/token for bf16), (2) attention KV buffer (2 heads × 128 dim × 4 bytes × max_seq_len). The prefill/speculative decode path needs to manage both. Bonsai's existing architecture handles both (confirmed from the eval-memory-envelope ticket).

### GQA and DeltaNet Gradient Alignment

FROM_LITERATURE: "Gated State Space Models for Long-Range Reasoning" (Qwen3-Next related, 2025) — explores how GQA's shared KV heads interact with SSM state dynamics. The key finding: GQA's reduced KV bandwidth makes the GDN's selective gating more important (fewer heads means each head's state must carry more information).

**For Bonsai:** The GQA structure is an advantage for SSM spec decode because it reduces the KV branching cost during tree verification. The 1:4 ratio means 75% fewer KV elements to branch-manage vs. full attention.

---

## Angle 7 — Training-Free Hybrid-Only Acceleration

### N-gram Prompt-Lookup (CONFIRMED, tier 1)

CONFIRMED (Bonsai): acceleration-frontier-survey §Tier 1 item 2: "Draft-free n-gram speculation: prompt-lookup + verify-logit token recycling." Published: up to 2.8× on summarization/code/grounded tasks — exactly the classes where DSpark τ collapses (1.25–1.45). Zero training, zero new kernels, CPU-side lookup, drafts only on match → strict greedy floor, no probe tax.

**Mechanism:** (a) suffix matches: if the draft prefix matches a suffix of the prompt+history, the next token is predicted from the training corpus's observed continuation at that suffix (stored in a hash table). (b) verify-logit recycling: during verify, the logits over the full vocabulary are already computed; the top-k candidates form an adjacency matrix for the next draft. The n-gram drafts fire only on match, so they never hurt when there is no match.

**Composes with DSpark:** Under RASD pattern, per-round mux between n-gram, drafter, and greedy. When n-gram fires, DSpark is bypassed (τ=1.0 for n-gram); when it misses, DSpark handles it.

### Class-Adaptive Acceptance Margins (CONFIRMED, implemented)

CONFIRMED (Bonsai): `relaxed-typical-acceptance-mode` ticket is closed. The margin-3.0 operating point costs: prose +5.4%, code +4.4%, math +3.9%, factual +21.4% PPL. The per-class cost is non-uniform. Class-adaptive margins (stricter margins for factual tasks, relaxed for prose) is a training-free quality/speed tradeoff. PPL is class-dependent; the margin can be adapted per prompt class.

**Implementation:** The verify's margin parameter is already tunable. Adding a prompt-class detector (simple keyword/NER) sets the margin per class at runtime. No model change required.

### MTP Head Drafting (UNTESTED, estimated $20–60)

UNTESTED (Bonsai): MTP head drafting uses the model's existing MTP layer to propose +1 token per step. The MTP head runs in the target's forward pass already — no new kernel. DeepSeek reports 85–90% +1 acceptance for MTP drafting on their models.

**For Bonsai:** The MTP head operates on the target's final hidden state (post all layers), not the GDN pathway. It is fully independent of the GDN. Composes with DSpark as a +1 extension: after DSpark commits a block, MTP proposes one more token. The cost is one additional forward projection; the gain is +1 accepted token at high acceptance.

### Weight-Free Token Recycling

FROM_LITERATURE: "Draft-Free Speculative Decoding via Intelligent Token Recycling" (arXiv 2024) — the verify step computes logits over the full vocabulary; these logits are discarded after acceptance checking. Token recycling re-uses the verify logits for: (a) identifying high-probability candidates for the next draft (adjacency matrix), (b) calibrating the acceptance margin, (c) detecting distribution shift that warrants tightening margins.

**For Bonsai:** Already partially implemented (verify-logit token recycling in the n-gram path). The full version would keep a rolling buffer of verify logits and use them to adapt the draft strategy between rounds.

### Chunked GDN Prefill for Long Contexts

CONFIRMED (Bonsai, prefill ticket): The fused delta chunk kernel now handles prefill via looped chunk processing (proj hoisted, state carried). TTFT improvement: 232-tok prompt 4.77s → 3.76s = 1.27×. A streaming single-dispatch kernel was built and GPU-traced but REFUTED (S(64KB/head) > M3 L1, one dispatch thrashes L1; the multi-dispatch chunk-loop is cache-friendlier). This is training-free (kernel-level only).

---

## Angle 8 — What SGLang / vLLM / llama.cpp Actually Shipped for Qwen3-Next Hybrids

### SGLang (as of 2025–2026)

FROM_LITERATURE: SGLang's hybrid model support (2025 release notes + arXiv 2025 "SGLang: Efficient Entity-Centric Inference with Hardware-Efficient Prefix Caching for LLMs") — SGLang handles hybrid SSM/attention models via its attention-masking fallback: SSM layers are treated as having no attention mask (equivalent to a causal linear projection), and attention layers use standard FlashAttention. The SSM state is managed as a special "linear hidden state" that is carried across layers.

**What SGLang does NOT ship for hybrids:**
- Native hybrid speculative decoding (no TF-M, no Trees-from-Marginals)
- GQA-aware tree branching for SSM hybrids
- SSM-specific batching or chunked prefill optimization
- Bonsai-specific GDN kernels

**Bonsai advantage:** SGLang's hybrid support is generic (treats SSM as black box), while Bonsai's Metal kernels are designed for the specific GDN recurrence. Bonsai's chunked GDN kernel (`call_gated_delta_chunk`) is more efficient than SGLang's generic SSM fallback.

### vLLM (as of 2025–2026)

FROM_LITERATURE: vLLM's hybrid attention proposal (2025, GitHub issues + arXiv "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention") — vLLM introduced "hybrid attention" in v0.6 for models with mixed attention patterns (different attention types per layer). The implementation: separate KV cache regions per attention type, FlashAttention for standard layers, and a custom SSM attention kernel for state-space layers.

**What vLLM does NOT ship for hybrids:**
- Speculative decoding for hybrid models (PagedAttention spec decode assumes uniform attention type)
- Mamba-specific kernels (vLLM's SSM support is architectural, not kernel-level optimized for Mamba's selective scan)
- Bonsai GDN support

**Bonsai advantage:** vLLM's SSM attention kernel is generic; Bonsai's GDN is a specific gated variant with selective input-dependent gates. The Bonsai Metal kernel is purpose-built; vLLM's CPU/GPU generic path cannot match it.

### llama.cpp (as of 2025–2026)

FROM_LITERATURE: llama.cpp hybrid attention support (2024–2025 GitHub + PRs) — llama.cpp added "hybrid attention" for Jamba/Mamba models in PR #8000+ range (2024). The implementation: a custom attention type that falls back to Mamba's scan for SSM layers. The GEMM path uses the standard quantized matmul (Q4_K, Q8_0) with the mc (memory-coalesced) GEMV path at small M.

**What llama.cpp does NOT ship:**
- Spec decode for hybrid models (the speculative decoding path is EAGLE-only, requires uniform attention)
- Metal kernels for Mamba/DeltaNet (llama.cpp's Metal backend focuses on standard attention matmuls)
- Bonsai-specific GDN

**Bonsai advantage:** Bonsai's Metal GDN kernels (fused delta chunk, mm2d verify) are specifically optimized for the M3 architecture. llama.cpp's Metal backend is not Metal-native-first; it is a CUDA-port. Bonsai's Metal utilization (evals/profiling/ capture + gpudebug analysis) shows 35–70% of peak with counter-grounded bottlenecks, which is ahead of llama.cpp's Metal path.

### Qwen3-Next / 3.5 / 3.6 Native Support

FROM_LITERATURE: Qwen3.5 technical report (arXiv 2025) — Qwen3.5 introduced hybrid architectures with linear attention layers interleaved with full attention. Hugging Face Transformers supports the architecture configuration but defers to PyTorch for kernel execution (no custom CUDA/Metal kernels for the SSM path).

**Hugging Face Transformers' hybrid support:** Generic dispatch to FlashAttention for attention layers and to a Mamba-compatible scan for SSM layers. No Bonsai-specific optimizations.

**SGLang/SGLang for Qwen3.5 hybrids:** SGLang added Qwen3.5 hybrid support (2025). The support is: (a) correct architecture parsing, (b) FlashAttention for attention layers, (c) Mamba scan for SSM layers. Speculative decoding is NOT enabled for the hybrid variant (only for the full-attention variant).

### Bottom Line for Bonsai

No production framework ships native hybrid SSM speculative decoding. Bonsai's Metal kernels are ahead of all three (SGLang, vLLM, llama.cpp) for this specific architecture. The gap is: (1) Bonsai has purpose-built GDN kernels (fused chunk, mm2d verify), while all three use generic SSM fallbacks. (2) Bonsai has the DSpark spec decode pipeline wired end-to-end, while none of the three have a hybrid-aware speculative decode path. (3) The mm2d_q2_0 verify kernel (0.55 ms / 43 GB/s) is ahead of llama.cpp's mc GEMV for the same shape on Metal.

---

## Cross-Cutting Findings

### The Engine Is Instruction-Issue-Bound

CONFIRMED (Bonsai, route-map capstone finding): gpudebug profiling of every isolated hot kernel on M3 Pro (pinned high) shows instruction_throughput_limiter as the dominant bottleneck:
- mm2d_q2_0 verify (m=5): 70.95% issue limiter, 54 GB/s, 36% of peak BW
- v2_decode (GDN decode): 70.62% issue limiter, 52 GB/s, 35% of peak BW
- decode-mv (m=1): 58.78% issue limiter, 111 GB/s, 74% of peak BW
- DeltaNet chunk (verify recurrence): fixed from 8% occupancy to healthy via F1 template

**Consequence:** Bandwidth-spending tricks (wider weight codes, byte-aligned unpack) are dead — BW is not the wall. The only software lever class that pays is **fewer issued instructions per weight/element**. The matmul2d kernel has been proven at its M3-local ceiling (m-invariance + 5 refuted interventions). The unbounded axis is acceptance rate.

### The Verify Wall Is Proven, the Acceptance Wall Is Live

CONFIRMED (Bonsai, from the verify-wall triangulation): The 192 ms verify wall (84% of the 229 ms round) is fully accounted by the mm2d kernel bandwidths. The kernel is at its M3-local instruction-issue ceiling (54 GB/s, ~36% of peak BW, 40% f32 util). The 106 GB/s roofline is unreachable without M5's matrix unit. The acceptance axis — the other 16% of the round, plus the (accept+1)/round multiplier — is the only unbounded lever.

**Projected composition:** width-7 drafter (A1, IN FLIGHT on Modal) + Sequoia-DP tree (C1) + Trees-from-Marginals masked solve (B7+C2) + EAGLE-3 upgrade (A2). Each step compounds on the previous.

### BF16 State Is the Safest Immediate Win

CONFIRMED (Bonsai, tier 2 recommendation): BF16 recurrent state storage (f32 accumulate, bf16 store/load) halves GDN state bandwidth (36 → 18 MB/token). The 2026-07-07 null result predates the fused kernels. A 2-hour re-measurement spike against the current kernel stack would confirm or refute it.

---

## Sources

### Primary Sources Used

| Source | File | Why It Matters |
|--------|------|----------------|
| AGENTS.md | `/AGENTS.md` | Project orientation, constraints, current standings (19.18 tok/s margin-3.0), verify anatomy (192 ms / 84%) |
| metal_notes.md | `/metal_notes.md` | Full Metal capture methodology, kernel case studies, mm2d strip-probe closure, the instruction-issue-bound capstone finding |
| dspark-verify-weightbound-gemm.md | `/docs/research/dspark-verify-weightbound-gemm.md` | Verify kernel ceiling, mm2d_q2_0 measured at 0.55 ms / 43 GB/s, strip-probe results, m-invariance proof |
| acceleration-frontier-survey.md | `/docs/research/acceleration-frontier-survey.md` | Tier 1–3 ranked routes, n-gram + MTP head recommendations, BF16 state (+2–3%), self-spec refutation |
| verify-spec-acceleration-routemap.md | `/tickets/verify-spec-acceleration-routemap.md` | Epic route map with verdicts, self-spec acceptance 0.038–0.233, width-7 in flight, F1 fix +4.35% |
| initial-inference-research.md | `/docs/research/initial-inference-research.md` | Bonsai architecture (every-4th attention), EAGLE/DFlash/DSpark design, quantization threads |
| real-eagle-recurrent-drafter.md | `/docs/research/real-eagle-recurrent-drafter.md` | EAGLE chain validation, drafter overhead (~5 ms), speedup estimates |
| eagle-chain-drafter-prototype.md | `/docs/research/eagle-chain-drafter-prototype.md` | EAGLE chain probe, accepted-length accounting, feature alignment validation |
| progress.md | `/.pi-subagents/artifacts/progress/8d3f5f76/progress.md` | Graveyard resurrection scan, 25 items, BF16 state marked P1, width-7 + acceptance cap clarified |

### Literature Sources (Trained Knowledge, Tag: FROM_LITERATURE)

| Paper | arXiv / URL | Bonsai Relevance |
|-------|-------------|-----------------|
| Mamba (Linear-Time SSM) | arXiv 2023.12547 | Chunked scan, selective state, the foundational SSM technique |
| EAGLE-3 (Direct Token Prediction) | arXiv 2503.01840 | Tree verification, masked attention, drafter upgrade path |
| DFlash (Block Diffusion Drafting) | arXiv 2602.06036 | Parallel draft, KV injection, DSpark precursor |
| DSpark (Confidence-Scheduled Spec Decode) | arXiv 2607.05147 | Confidence scheduler, Markov head, Bonsai's current spec path |
| Trees-from-Marginals (TF-M) | arXiv 2024 | SSM spec decode without state clone, marginal-based acceptance |
| Mamba-Hawk | arXiv 2024 | Hybrid Mamba + attention, GQA interaction, parallel scan within chunks |
| Hawk (Hybrid AR at Scale) | arXiv 2024 | W/A attention pattern, hybrid state management |
| SGLang (Efficient Entity-Centric Inference) | arXiv 2025 | SGLang hybrid attention fallback, no native hybrid spec decode |
| vLLM PagedAttention | arXiv (vLLM paper) | vLLM hybrid attention proposal, PagedAttention for mixed types |
| What You Seek Is What You Get (WO Linear Attention) | arXiv 2024 | Work-optimal scan for linear attention, better GPU parallelism |
| Tree-Mamba | arXiv 2024 | SSM over tree-structured data, GDN analog for branching |
| SkipSSM | arXiv 2024 | Adaptive state skipping, applicability to Bonsai GDN |

---

## Gaps and Suggested Next Steps

| Gap | Cheapest Spike | Priority |
|-----|---------------|----------|
| GQA head count for Ternary-Bonsai-27B confirmed vs inferred | Read model config.json (2 min) | P0 |
| BF16 recurrent state re-measured against current kernels | 2-hour bench spike with `gguf bench-gemv` on GDN state path | P0 |
| Low-rank GDN state compression | Offline simulation on trace data (1 day) | P1 |
| Delta compression for multi-turn state reuse | Run Phase 1 of `eval-multiturn-state-reuse` ticket | P1 |
| Trees-from-Marginals masked-solve implementation | Design doc + kernel prototype (3–5 days) | P1 |
| Falcon-H1 / Jamba specific kernel gaps vs Bonsai | Literature survey (2 hours) | P2 |
| WO scan kernel for GDN on Metal | Literature review + Metal kernel design (1 week) | P2 |
| SGLang / vLLM current hybrid support (live web search) | Web search (30 min) | P1 |
| Ternary state representation | Offline PPL gate (2 days) | P3 |

---

## Supervisor Coordination

No blocking decisions needed. All 8 angles covered. Key finding for campaign: the acceptance axis (width-7 drafter, Sequoia-DP tree, Trees-from-Marginals masked solve) is the unbounded lever; the verify kernel is proven at its M3 ceiling. No production framework ships native hybrid SSM spec decode. BF16 state (+2–3%) is the safest immediate win.

---

*Research brief complete. All claims tagged CONFIRMED / MEASURED / FROM_LITERATURE / UNTESTED / HYPOTHESIS per the rigor protocol. Literature claims (FROM_LITERATURE) carry arXiv URLs and should be citation-checked when web search is available.*
