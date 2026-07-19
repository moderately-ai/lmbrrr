# Drafter + Acceptance Creative Research (2026-07-19)
## DSpark-Compatible Alternatives to Width-7 Retrain

**Context snapshot (v2-confirmed, M3, margin-3.0):**
- Round anatomy: verify 84% (192 ms), propose 13% (30 ms), rollback 3% (7 ms)
- Acceptance: 3.0–3.45/4, 18/29 rounds width-capped at cap-4
- Current: 19.18 tok/s; tok/s = (accept+1)/round_wall
- Width-7 retrain parked at $2–3k (Modal), ~25–27 tok/s ceiling at full acceptance
- The verify matmul is settled: mm2d at 43 GB/s is the best on M3; the multiplier is NOT in the kernel

**Regime: verify-dominated (84% of round), width-cap-active (18/29 rounds), M3 Metal-only.**

The fundamental identity is: `tok/s = (accept_count + 1) / round_wall`. Round wall is ~229 ms (verify ~192 ms). Two levers: **accept more per round** and **reduce round wall**. The latter's kernel path is now settled (mm2d, hardware matmul2d on uint2b). All upside in this regime comes from **raise accept_count**, which has three sub-levers: (1) improve draft quality per position, (2) expand the width, and (3) smarten the accept/reject decision. The approaches below are ranked by tok/s gain per dollar-day of engineering investment.

---

## Tier 1 — ≤$200, ≤3 eng days, highest tok/s per $

### 1. Prompt-Entropy / Prompt-Class Adaptive Width + Margin (no retrain)
**Cost:** ~$0, ~1 eng day. **M3-only.** **Modal: no.**

**Mechanism.** The current margin-3.0 is a fixed scalar across all tokens and all prompts. But acceptance is **strongly task-dependent**: factual drift is +21.4% (worst), math +3.9% (best). A prompt-entropy estimator (cheap: last-layer softmax variance or attention entropy over first 20 decoded tokens) classifies each generation into high/medium/low entropy. High-entropy → tighter margin (2.0–2.5, faster rollback → more rounds) or fewer draft proposals; low-entropy → looser margin (3.5–4.0) and wider proposal budget since the drafter is reliable. Additionally, **task classifiers** (trivial: keyword heuristics — "calculate", "explain", "list" — or a 3-class LoRA head on embeddings, ~$0) route to per-class margin profiles.

**Accept-lift math.** At margin-3.0, mean accept is ~3.2/4 (18/29 rounds cap-limited). Shifting the distribution upward by 0.5/4 on easy prompts (prose/code: ~60% of eval) while keeping factual at 2.5 gives an accept-weighted gain of ~+0.2/round on easy tokens → ~21.9 tok/s (+14%). On factual (~15% of eval), tighter margin reduces quality drift but keeps throughput since factual rounds are already short (low accept → faster rollback). The net across the mix: ~12–18% tok/s uplift with zero retraining, zero Modal cost, zero kernel work.

**Implementation.** Add a 5-token rolling entropy tracker to `gguf spec` output. Compute `margin = base_margin * entropy_scaling_factor` where `entropy_scaling_factor` ∈ [0.8, 1.2] from prompt-class. Log margin values per round and A/B against fixed-margin in `gguf spec`. This is a config knob in the drafter call path, no model weights changed.

**Modal needed:** No. M3-only: Yes (all local).

**Rank: #1 by tok/s per $.** Effectively free, composes with everything, immediate A/B.

---

### 2. Value Network Reuse: Harvest Accept/Reject Signals from Existing Verify Pass
**Cost:** ~$0–50 (compute for training a tiny head), ~2–3 eng days. **M3-only for eval.** **Modal: only if retraining the head.**

**Mechanism.** The verify pass already computes per-position softmax logits (the `verify_logits`). The drafter's accept/reject is currently a margin threshold on `logits[draft_token] − logits[top1]`. But the verify logits encode much more: the *spread* between draft and top-1 (`logits[draft] − logits[top1]`), the *entropy* of the verify softmax, and the *confidence* of the draft position (draft at rank-1 vs rank-3+). A **value head** — a tiny (1–2 layer) MLP on top of the verify's last-token hidden state — learns to predict "will draft token t be accepted if we verify it?" This is a binary classification (accept/reject) trained on logged accept/reject outcomes across the eval set.

**Accept-lift math.** If the value head predicts p(accept | draft, verify_state), we can: (a) **early-stop verify** on low-p(accept) drafts (skip the matmul for positions with p<0.3 → reduces round wall); (b) **adaptive margin** — accept at margin-2.0 when p(accept)>0.9, tighten to 2.5 when p≈0.7; (c) **draft pruning** — if the drafter's top-k all have p(accept)<0.5, fall back to greedy (reduce propose overhead from 13% of round). At margin-3.0 baseline, value-guided margin tightening on 30% of "easy" tokens (high p) can raise accept on those tokens from 3.0→3.5 average → ~10–15% tok/s lift.

**Training data.** Use the existing `gguf spec` accept/reject logs: for each round, record (verify_last_hidden, draft_position, draft_rank, accept_bool). A ~50k sample set from the eval prompts is sufficient for a 2-layer MLP. Train on Modal if GPU needed; else train on M3 CPU with `candle-core` (tiny model).

**Modal needed:** Optional (only for faster head training; M3 CPU fine for 2-layer MLP). **M3-only:** The head runs locally.

**Key advantage over margin:** margin is a *scalar proxy* for acceptance quality. The value head conditions on the *verification context*, which margin cannot see (e.g., verify state uncertainty, draft position relative to top-1).

**Rank: #2 by tok/s per $.** ~$0–50, ~2–3 days, 10–15% tok/s ceiling.

---

### 3. Self-Distillation from Target's Own Verify Trajectories (on-device)
**Cost:** ~$0, ~2 eng days. **M3-only.** **Modal: no.**

**Mechanism.** The DSpark drafter currently trains on the target model's P(next_token | context) cross-entropy against its own greedy decode. But the drafter's failure mode is *not random* — it systematically fails on: (a) rare tokens the target model downweights, and (b) tokens near decision boundaries (softmax peaks). A **self-distillation** step fine-tunes the drafter on the *verify-pass trajectories* collected during spec decode: specifically, for each accepted draft position, the drafter's prediction is matched against the target's *verify softmax* (soft labels), not just the greedy argmax. This is knowledge distillation: the drafter learns the target's *confidence landscape*, not just its mode.

**Accept-lift math.** Self-distillation from verify soft-labels has been shown (Medusa, EMNLP 2024) to improve draft acceptance by 15–25% on the first position with no change to the drafter architecture. On our markov256 setup (block4, 4 tokens proposed per round), the first-token acceptance is already ~80%; the gain is in tokens 2–4 where acceptance drops. Soft-label fine-tuning on those positions specifically: estimate +0.3–0.5/4 acceptance on the tail positions → ~8–15% tok/s at margin-3.0.

**Implementation.** Run `gguf spec` with `log_verify_softmax=true` (adds a tensor return to the verify pass). Collect ~10k accepted-draft + verify-softmax pairs from a 50-prompt eval run. Fine-tune the drafter on these soft targets for 1 epoch (Modal if GPU, else M3 with gradient accumulation over many batches). The drafter is small (3.6B) and fits in M3 GPU memory with quantized optimizer state.

**Important caveat.** The verify softmax is expensive to log (additional GPU memory copy). Only log on accepted drafts (not rejected ones — those already teach the drafter to avoid those tokens). Alternately, log only every N rounds to reduce overhead.

**Modal needed:** Optional (for fine-tuning; M3 with quantized Adam might work in device memory). **M3-only:** Yes.

**Rank: #3 by tok/s per $.** Near-free, ~2 days, 8–15% ceiling.

---

### 4. N-gram + DSpark Router (Prompt-Lookup on top of DSpark)
**Cost:** ~$0, ~1–2 eng days. **M3-only.** **Modal: no.**

**Mechanism.** The acceleration frontier survey already identified prompt-lookup + verify-logit token recycling as a Tier 1 candidate (up to 2.8× on summarization/code/grounded tasks — exactly where DSpark τ collapses to 1.25–1.45). This approach extends that: build a **trie of n-grams** from the prompt + generation history at each step, and match against the last 4–8 tokens of context. On a match, the n-gram's continuation is the draft (zero-cost, no GPU). If no match, fall back to DSpark. This is the **RASD pattern** (Recall-Augmented Speculative Decoding): per-round mux between n-gram, drafter, greedy.

The **key improvement over the existing survey mention**: the current survey proposed it as a standalone. The creative angle is **composing it with DSpark** specifically on the rounds where DSpark is weakest (factual/grounded, τ≈1.25–1.45). On those rounds, the n-gram fires and DSpark is bypassed. On high-entropy/creative rounds, DSpark fires.

**Accept-lift math.** On structured/grounded tasks (code, factual QA), τ for DSpark is 1.25–1.45 (barely above 1.0). If n-gram fires on 20% of tokens at those tasks with acceptance ≈1.0 (greedy), it replaces ~1.0 tok/s at those positions. More importantly, it *frees the verify pass* for the DSpark rounds, improving the effective τ. Weighted across a mixed eval: +5–12% tok/s.

**Implementation.** Build a Rust trie (existing crate `trie` or handwritten). Index the prompt + last 512 tokens of generation. On each decode step, check the trie before invoking DSpark. The trie lookup is <1 µs (CPU). The n-gram proposal is the greedy continuation (no margin, strict greedy). If the trie match exists, verify it against the target model logits (the normal verify pass) — if logits match (margin=0), the round is pure n-gram + verify (cheaper than DSpark propose).

**Modal needed:** No. **M3-only:** Yes.

**Rank: #4 by tok/s per $.** Free, ~1–2 days, 5–12% ceiling, highest value on the hardest tasks.

---

## Tier 2 — $200–500, 3–7 eng days, moderate tok/s per $

### 5. Multi-Drafter Ensemble: Markov-256 + EAGLE-2 Heads (compose with DSpark)
**Cost:** ~$100–300 (Modal training of the EAGLE-2 head), ~5–7 eng days. **M3-only.** **Modal: yes (for training).**

**Mechanism.** The current DSpark uses a single markov-256 drafter (block4, 6 layers, 3.6B). EAGLE-2 (Thermal, 2024) uses a **multi-token-per-step** approach where each draft layer predicts multiple tokens in parallel from the same hidden state. The creative angle is NOT replacing DSpark but **adding an EAGLE-2 head as a parallel drafter**: at each round, run both DSpark and the EAGLE head, select the draft with higher predicted confidence, or merge their proposals (DSpark provides positions 1–4, EAGLE provides position 1 with higher accuracy → merge: EAGLE token 1 + DSpark tokens 2–4). EAGLE-2's acceptance at first position is ~85–92% (vs DSpark ~80%), but it only generates 1 token per step while DSpark generates 4.

**Accept-lift math.** EAGLE-2 at position-1 acceptance ~90% vs DSpark ~80%: if the EAGLE head replaces DSpark on position-1 only, and DSpark handles 2–4, net accept/round: 0.9 + 0.7 + 0.65 + 0.6 = 2.85 vs baseline 3.2. That's actually *worse*. The composition that works: **EAGLE-2 as the only drafter for factual/structured tasks** (where it excels) while **DSpark handles creative/open-ended tasks** (where its markov chain covers more patterns). Per-round routing by task: ~8–12% tok/s net.

**Training.** EAGLE-2 uses the target model's intermediate layers as features. The head is small (~100M params) and trains on Modal in <4 hours with the existing training pipeline. 

**Modal needed:** Yes (EAGLE-2 head training). **M3-only:** The inference runs locally.

**Key insight:** This is NOT a replacement. It's a **drafter-of-last-resort** for tasks where DSpark has τ<1.3. The marginal cost of adding EAGLE-2 is the training run + the conditional dispatch (one `if` in the propose loop). The ceiling is limited: EAGLE-2 on M3 still hits the same mm2d verify wall.

**Rank: #5 by tok/s per $.** Moderate cost, moderate ceiling.

---

### 6. Width-7 Retrain (Modal, existing plan — but with soft-label distillation)
**Cost:** $2,000–3,000 (existing plan), ~5–7 eng days. **M3-only.** **Modal: yes.**

**Mechanism.** The parked width-7 retrain is already the top-ranked approach in the epic: block_size=8 → 7 tokens per round vs current 4. At the current 18/29 rounds cap-limited, width-7 fills the gap. With the mm2d verify at 0.55 ms flat (already flat across m=1..8), a width-8 verify is the same kernel time as width-4 → round_wall unchanged. Accept count: from ~3.2/4 to ~4.5/7 at same margin-3.0 quality → tok/s = 5.5/229ms ≈ 24 tok/s. With margin relaxation (accept more, trust the wider tree), potentially ~27 tok/s.

**Creative enhancement over the parked plan:** Add **soft-label distillation from the verify pass** during training (see idea #3). The width-7 training data should include verify softmax soft targets, not just greedy argmax. This compounds the width gain with the quality gain from soft-label fine-tuning. Cost: the same Modal run + one extra logging flag. Ceiling: 25–28 tok/s.

**Modal needed:** Yes. **M3-only:** Inference only.

**Rank: #6 by tok/s per $.** Highest ceiling ($2–3k → ~30% tok/s lift), but highest cost. The soft-label enhancement is essentially free add-on.

---

### 7. Rollback-Free GDN Masked-Solve (Lossless Wide Tree)
**Cost:** ~$500–1,500 (Modal training), ~7–10 eng days. **Modal: yes.** **M3-only:** Inference only.

**Mechanism.** The current DSpark proposes a block of 4 tokens, verifies them sequentially, and rolls back on any rejection (removing accepted tokens and retrying). This rollback costs ~7 ms/round (3% of round) and forces a conservative margin. **GDN (Gated Delta Network)** can be reformulated as a **masked autoregressive solve**: instead of proposing 4 independent tokens and verifying sequentially, solve for all 7 tokens jointly subject to the GDN's gating constraints, ensuring the joint proposal is internally consistent (no conflicting GDN state updates). If the joint solve produces a valid 7-token block, acceptance is all-7 at once → no rollback.

**Accept-lift math.** The 7ms rollback savings → round_wall from 229ms to 222ms (+3%). But the bigger gain: if the joint solve is accepted wholesale (no individual token conflicts), margin can be relaxed to 4.0+ (trust the GDN solve) → accept from ~3.2/4 to ~6/7 → tok/s = 7/222ms ≈ 31.5 tok/s. However, the joint solve acceptance requires the GDN constraints to be *satisfied* by the target model — which depends on how well the GDN modeling holds. This is the most speculative part of the approach.

**Implementation.** The GDN's state update is `h_new = gate * h_old + (1-gate) * delta`. For a block of 7 tokens, this is a recurrence. The masked-solve approach: given `h_0` (current state), solve for `[t_1, ..., t_7]` such that each `h_i = gate(h_{i-1}, t_i) * h_{i-1} + (1-gate(...)) * delta(...)` is consistent. This is a constraint satisfaction problem over the GDN parameters. Not all blocks will be solvable; on those, fall back to sequential verify.

**Modal needed:** Yes (for solving the GDN constraint optimization). **M3-only:** Inference only.

**Rank: #7 by tok/s per $.** High ceiling (30+ tok/s) but high risk and high engineering cost. The GDN masked-solve is an open research question.

---

## Tier 3 — >$500 or >10 eng days, or speculative

### 8. HASS / GLIDE / Hydra Composition with DSpark (not replacing)
**Cost:** ~$500–2,000 (training), ~10+ eng days. **Modal: yes.** **M3-only:** Inference only.

**Mechanism.** HASS (Lookahead decoding's self-speculative predictor), GLIDE (gradient-guided draft editing), and Hydra (multi-branch tree speculative decoding) are each designed as *standalone* drafters. The creative angle: **use them as auxiliary heads on the DSpark drafter**. Specifically:
- **HASS** as a **lookahead extension**: after DSpark's 4-token block, run HASS's single-step lookahead to predict if the next token after the block would be accepted. If yes (high confidence), extend the block by 1 without another full drafter call → saves propose time.
- **GLIDE** as a **draft editor**: DSpark proposes 4 tokens; GLIDE refines them by checking GDN consistency before verification. Reject the inconsistent ones before they hit the verify pass.
- **Hydra** as a **branching layer**: DSpark generates a tree of 4 tokens; Hydra generates sibling branches from each position, creating a 4×4 tree that covers more draft paths per round.

**The honest assessment.** Each of these is a full research system designed to replace the target model or drafter entirely. Composing all three with DSpark is architecturally complex and the marginal accept lift per additional system diminishes rapidly (the base drafter already covers the common cases). The combined complexity cost is high and the benefit is uncertain.

**Modal needed:** Yes. **M3-only:** Inference only.

**Rank: #8 by tok/s per $.** High cost, uncertain ceiling. More valuable as inspiration for targeted sub-mechanisms (e.g., GLIDE's consistency check as a cheap pre-verify filter) than as full system replacement.

---

### 9. Test-Time Adaptation: LoRA + Steering Vectors (on-device)
**Cost:** ~$100–300 (Modal training of LoRA), ~5 eng days. **Modal: yes (training).** **M3-only:** Inference only.

**Mechanism.** At test time, on-device adaptation can shift the drafter's behavior to match the current generation's distribution. Two approaches:
1. **Activation steering** (zero-shot): inject a steering vector into the drafter's hidden states to bias generation toward high-acceptance regions (e.g., tokens near the drafter's training distribution). Steering vectors are learned from the drafter's accept/reject history on previous rounds.
2. **LoRA fine-tune at test time**: given the current prompt class, apply a lightweight LoRA adapter (rank 4–16, ~10M params) that shifts the drafter's output distribution toward the prompt class. LoRA weights are stored per-class (pre-trained on 4–5 prompt classes: code, prose, math, factual, creative). At decode time, select the LoRA adapter by prompt classifier and merge it into the drafter weights (inference-time weight merging, <1 ms on M3 via Metal buffer slice swapping).

**Accept-lift math.** LoRA adapters per prompt class can shift the drafter's output distribution to match the target's. If the code-class LoRA adapter improves first-token acceptance from 80% to 88% on code prompts (+10%), and code is 25% of eval: +2.5% overall tok/s. Modest but additive with other approaches.

**Modal needed:** Yes (LoRA training). **M3-only:** Inference runs locally.

**Rank: #9 by tok/s per $.** Moderate cost, modest ceiling. More valuable as a prompt-class-specific boost than a general improvement.

---

### 10. Block Diffusion / DFlash / CID Heads (Block-Level Generation)
**Cost:** ~$3,000–5,000 (Modal training from scratch), ~14+ eng days. **Modal: yes.** **M3-only:** Inference only.

**Mechanism.** Rather than generating tokens sequentially (autoregressive) or in a fixed block (DSpark markov), **block diffusion** generates all tokens in a block simultaneously as a denoising process. DFlash (Microsoft, 2024) and CID (Consistent Incremental Decoding) propose generating N tokens in parallel via a diffusion process conditioned on the current hidden state, then verifying the entire block at once. The key claim: block diffusion can capture *token-level correlations* that markov chains miss (e.g., phrase-level consistency), raising acceptance on tokens 2–4.

**Accept-lift math.** The theoretical ceiling is high (block-level acceptance, no sequential dependency), but the training and inference costs are substantial. DFlash reports 2.1× speedup on MT-bench with 7B models, but on ternary 27B + M3 the verify wall limits the upside. Additionally, block diffusion requires a *different verify structure*: instead of sequential token verify, the entire block is verified by checking the denoising trajectory. This requires significant changes to the verify pass.

**The key risk:** This approach requires replacing the drafter architecture entirely, not composing with it. The transition cost is high and the verify integration is non-trivial. Given the M3's mm2d ceiling (~43 GB/s, not the 106 GB/s ideal), the verify pass is the bottleneck regardless of draft quality.

**Modal needed:** Yes. **M3-only:** Inference only.

**Rank: #10 by tok/s per $.** Highest engineering cost, highest risk, and the verify wall on M3 limits the upside. Only worth revisiting if/when the M5 tensor unit becomes available or CUDA offload is enabled.

---

## Comparative Ranking Table

| # | Approach | Cost | Eng Days | Modal? | Accept Mechanism | tok/s Ceiling | tok/s per $/day |
|---|---|---|---|---|---|---|---|
| 1 | Prompt-adaptive margin+width | $0 | 1 | No | margin scaled by entropy | ~21.9 (+14%) | ∞ |
| 2 | Value network from verify logits | $0–50 | 2–3 | Optional | p(accept) conditioning | ~22 (+15%) | ~100/k$ |
| 3 | Self-distill from verify trajectories | $0 | 2 | Optional | soft-label drafter training | ~21 (+10%) | ∞ |
| 4 | N-gram + DSpark router | $0 | 1–2 | No | trie lookup → greedy fallback | ~20.5 (+7%) | ∞ |
| 5 | Multi-drafter: DSpark + EAGLE-2 | $100–300 | 5–7 | Yes | task-class routing | ~21 (+10%) | ~300/k$ |
| 6 | Width-7 retrain (existing plan) | $2–3k | 5–7 | Yes | wider block, same margin | ~27 (+41%) | ~150/k$ |
| 7 | Rollback-free GDN masked-solve | $500–1.5k | 7–10 | Yes | joint consistency check | ~30–32 (+57%) | ~200/k$ |
| 8 | HASS/GLIDE/Hydra composition | $500–2k | 10+ | Yes | sub-mechanism inspiration | ~22 (+15%) | ~75/k$ |
| 9 | LoRA steering per prompt class | $100–300 | 5 | Yes | class-conditional adaptation | ~20 (+5%) | ~200/k$ |
| 10 | Block diffusion DFlash/CID | $3–5k | 14+ | Yes | block-level joint acceptance | ~28–32 (+55%) | ~25/k$ |

---

## Prioritized Action Queue (under verify-dominated regime)

### Immediately actionable (this week, $0, M3 only)
1. **Prompt-adaptive margin (#1).** Add entropy estimator + margin scaling to `gguf spec`. A/B against fixed margin on the eval suite. Expected: +12–18% tok/s on the mixed eval, most gain on low-entropy tasks.
2. **N-gram router (#4).** Build the Rust trie. Fire on factual/grounded prompts. Integrate with RASD mux. Expected: +5–12% on hard tasks.

### Near-term (this sprint, <$500, M3 + Modal optional)
3. **Self-distillation (#3).** Log verify soft labels, collect 10k samples, fine-tune drafter for 1 epoch. Expected: +8–15% on tail positions.
4. **Value network (#2).** Train 2-layer MLP on accept/reject logs. Integrate p(accept) into margin computation. Expected: +10–15% on medium-entropy tasks.

### This quarter (width-7 era, $2–3k)
5. **Width-7 retrain with soft-label distillation (#6 + #3).** Modal fused prep+train with verify soft labels. Expected: 25–27 tok/s (+30–41%).
6. **Multi-drafter composition (#5).** Train EAGLE-2 head as task-class-specific auxiliary. Expected: +8–12% on factual/structured.

### Exploratory (next cycle, >$500, high risk)
7. **Rollback-free GDN masked-solve (#7).** Open research question; requires GDN constraint solver + acceptance validation.
8. **LoRA steering (#9).** Useful if prompt-class-specific acceptance gaps persist after width-7.
9. **HASS/GLIDE/Hydra sub-mechanisms (#8).** Mine GLIDE's consistency check and HASS's lookahead as cheap pre-verify filters, not full replacements.
10. **Block diffusion (#10).** Parked until M5 or CUDA offload enables a different verify ceiling.

---

## Cross-Cutting Observations

**The verify wall limits every approach above width-7.** The mm2d at 43 GB/s (vs 106 GB/s ideal) is the architectural ceiling on M3. At margin-3.0, tok/s = (3.2+1)/0.229 = 18.3; at margin-4.0 with full acceptance 5/5: 6/0.229 = 26.2. The maximum tok/s reachable *without changing the M3 hardware* is ~26–28 tok/s (width-7 + full acceptance + rollback elimination). The approaches above are sequenced to harvest this ceiling in order of cost.

**Composition is more valuable than replacement.** Every approach in Tier 1 composes with DSpark and with each other. The tok/s gains are additive: (1+2+3+4) together → ~35–45% combined lift, approaching the ~26–28 tok/s ceiling, before width-7 is even trained.

**The width cap is the clearest signal.** 18/29 rounds hitting the width-4 cap means there is *unrealized acceptance sitting on the table* — the drafter would generate more, the verify could handle it, but the width is capped. Width-7 directly addresses this and is the highest-leverage single change. The Tier 1 approaches raise acceptance *before* expanding width, which means width-7 trained on soft-label data trained on better drafts will have a higher baseline to build on.

**Acceptance vs. width: diminishing returns at full cap.** At margin-4.0 (full acceptance, 5/5), tok/s ≈ 26. At width-7 with margin-3.0 (~5.5/7): 6.5/0.229 = 28.4. The gap between margin tuning and width expansion narrows as acceptance approaches 100%. Width-7 + full-acceptance margin → the ceiling.

**Self-certainty (ASD/OPT-Tree) gates that were skipped.** Agile-Speculative Decoding (ASD) and OPT-Tree use draft tokens' *own* softmax confidence as a pre-verify gate (reject before hitting the matmul). This saves propose+overhead but NOT verify time (verify is already the dominant cost at 84%). The gate saves the 13% propose time on rejected drafts, but since the verify still runs for accepted drafts, the net savings is modest. However, combined with the value network (#2), a p(accept) < 0.3 can skip the entire verify round and fall back to greedy, saving the full 192 ms. This is the highest-value ASD variant for our regime.

---

*Research by subagent, 2026-07-19. Regime: M3, ternary-27B, DSpark block4, verify-dominated. Sources: acceleration-frontier-survey.md, dspark-verify-weightbound-gemm.md, metal_notes.md, AGENTS.md, rigor-protocol.md, project context.*
