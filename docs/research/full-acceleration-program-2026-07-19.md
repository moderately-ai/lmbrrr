# Full Bonsai acceleration program (2026-07-19)

Canonical, size-agnostic program from the adversarial audit + creative second
pass. Supersedes the stale ranked-actionable block on
`verify-spec-acceleration-routemap` (that epic's body is rewritten to point
here). Measurement law remains `docs/research/rigor-protocol.md`.

## Non-negotiables

1. **Identity:** `tok/s ≈ (accept+1) / round_wall`. Round ≈ verify 84% + propose
   13% + overhead ~3% (margin arm, v2 ruler, ~229 ms flat).
2. **Physics:** hot kernels are **instruction-issue-bound**, not DRAM-bound.
   Bandwidth-spending tricks are dead on the M3 hot path. Skipping work and
   raising `(accept+1)` are the live classes.
3. **Verify matmul settled:** mm2d @ m≤8 is M3-local ceiling (m-invariance +
   five refuted interventions). Kernel micro on mm2d is closed except gated
   Modal fullk-from-original (uncertain PPL).
4. **Exactness classes** (label every spike):
   - **E** exact / byte-match greedy
   - **Q** quality-gated (PPL/KLD battery required before default)
   - **P** product-mode (changes UX contract; user-facing flag)
5. **Blessed standings config** (re-baseline after Phase 0):
   - Machine: M3 Pro referee, quiet, rotated arms
   - Binary: main + candle pin in Cargo.toml
   - Default product OP: planar mm2d, margin-1.0, Q8_0 drafter, fused prefill
   - Campaign speed OP: same + `--fast` (margin-3.0)
   - Ruler: gguf v2 fields; N=128; 3 rotated reps; report median + spread
6. **Kill criteria are mandatory.** A spike that cannot name its kill in one
   sentence is not a spike.

## Do-not-retry (load-bearing negatives)

| Route | Why closed |
|---|---|
| dequant→bf16→dense simdgroup @ m≤8 | 2.8× slower; MLX qmm same wall |
| int8 matrix pre-M5 | no integer datapath |
| algebraic sparsity/WHT/mult-free | no HW / more instructions |
| component-aware self-spec (drop attn) | sequential hybrid 38× slowdown |
| replace DSpark with Medusa/lookahead/cascade | acceptance regresses |
| certified CSV sub-vocab head (MiniCPM geom) | bounds 50–100× gaps |
| fullk-from-deployed-Q2_0 pow2 | +18.2% weight error |
| extract-c Q2_0 mv map | −30% vs select-form |
| streaming DeltaNet prefill on M3 | S>L1 thrash |
| row peakedness gate for factual margin | trajectory-level drift |
| BF16 recurrent state for speed | <0.05 ms/tok; quality hazard |
| global FR-Spec 64k | suite-negative |
| q4_K schedule/layout/SoA micros (MiniCPM) | issue-rate bound |

---

## Phase map (everything)

```text
P0  Truth          board + baseline + docs          [blocking for claims]
P1  Free flags     tree/PLD/argmax/oracles          [E, 0-1 day]
P2  Exact scraps   bitplane decode, int2b, MLX gap  [E]
P3  Control flow   scheduler, recycle, rollback     [E]
P4  Accept policy  class/entropy/grammar margin     [Q/P]
P5  Approx verify  early-exit / SPRINTER family     [Q]
P6  Hybrid native  selective GDN, checkpoints       [E then Q]
P7  Drafter        fidelity, EAGLE-3, Weaver, τ-loss[Q, Modal-gated]
P8  Tree deep      Sequoia-DP, B7 masked-solve      [E, after P1 tree]
P9  Product surface multi-turn, TTFT archive, mem   [P]
P10 Gated bets     fullk-original, width-7, M5      [Q/roadmap]
```

Dependencies (hard):

- No tok/s **claim** without P0 re-baseline after the change lands.
- P5 approx-verify requires P0 quality battery for Bonsai.
- P7 width-7 / big Modal requires P7a fidelity check first.
- P8 B7 only if P1/P8 tree EV > 0 on hard prompts.
- P10 fullk only after explicit go/no-go on PPL risk.

---

## P0 — Truth layer (blocking)

| ID | Work | Exact | Spike | Kill / done-when |
|---|---|---|---|---|
| P0.1 | Re-baseline blessed configs (plain / exact / m1 / m3) post-F1 + defaults + Q8_0 | E | M3 rotated 3-rep | numbers in AGENTS + performance.md agree |
| P0.2 | Board hygiene: ship done tickets; clear 8 stale `in-progress`; park MiniCPM-only lane under tag | — | local | `tkt reconcile` clean or explained |
| P0.3 | Rewrite epic ranked-actionable → pointer to this doc | — | local | epic not claiming width-7 in flight |
| P0.4 | Amend rigor-protocol.md: v2 ruler live for gguf | — | local | no “v1 biased only” standing text |
| P0.5 | Bonsai quality battery: exact / m1 / m3 × ≥3 prompts/class PPL+KLD | Q | M3 | table on `eval-quality-reference-battery` Bonsai section |
| P0.6 | Single standings table schema (config×margin×drafter×N×ruler) | — | local | template used by all later spikes |

---

## P1 — Free flags & audits (exact, code already exists)

| ID | Work | Where | Spike | Kill |
|---|---|---|---|---|
| P1.1 | **Tree ON** hard-prompt suite (TW=3, m=7 flat) | `gguf_run.rs:2086` `use_tree` | M3 A/B tree vs chain, exact+m1 | kill if tok/s ≤ chain −2% on hard set AND alt-win rate <5% |
| P1.2 | **PLD ON/OFF** on Bonsai gguf | already wired | M3 | kill if no class ≥ +2% non-overlap |
| P1.3 | `LMBRRR_FUSED_VERIFY_ARGMAX` A/B | env | M3 | kill if < +1% exact |
| P1.4 | Offline **oracle best-width** from logged confidences vs fixed block_size | logs + notebook | local/M3 | if oracle lift < +3% tok/s equivalent, scheduler deprioritized |
| P1.5 | **Gate/β/state-Δ histograms** per GDN layer (selective-compute prereq) | new diag | M3 | if <10% near-zero gates, selective GDN killed |
| P1.6 | **Layer-hidden → accept probe AUC** at attn checkpoints {15,31,47,63} | capture + sklearn | M3 | if best AUC < 0.85, early-exit verify killed |
| P1.7 | Readvance vs capture rollback cost under current stack | env | M3 | pick default; document |

---

## P2 — Exact kernel / decode scraps

| ID | Work | Exact | Spike | Kill |
|---|---|---|---|---|
| P2.1 | **m=1 bitplane decode GEMV** (B3 successor; 133.7 vs 111 GB/s measured) | E | isolated then e2e plain | kill if e2e plain < +2% |
| P2.2 | **int2b_format signed** path (delete P−rowsum); bound ≤7% by probe_nofold | E | kernel A/B | kill if spec < +1% |
| P2.3 | **MLX vs lmbrrr decode gap** decomposition (format vs kernel) | E | same-session referee | produce % split; open port ticket only if kernel ≥8% of gap |
| P2.4 | LUT/T-MAC Metal probe (ticket says never run) | E | micro only | kill if issued-instr not down vs mv |
| P2.5 | mm2d geometry leftovers only if in-loop gated (split-K body already neutral) | E | — | default: **closed** unless new counter |

---

## P3 — Control-flow exact (compose with P1)

| ID | Work | Exact | Spike | Kill |
|---|---|---|---|---|
| P3.1 | **Port SpecScheduler + STS** into gguf Bonsai path (today fixed width) | E | after P1.4 | kill if suite ≤ fixed-width |
| P3.2 | Token-recycle harvest gate (mm2d flat lowers BE) | E | M3 | kill if no class +2% |
| P3.3 | Per-position GDN state emission → free partial-accept rollback | E | fork kernel | kill if rollback path < −1 ms/round realized |
| P3.4 | Chain-handoff for parked/weak classes (MiniCPM lesson) | E | after scheduler | kill if weak classes already ≥ plain |
| P3.5 | Device chunk assembly / one-sync audit on Bonsai path | E | code read + A/B | document residual syncs |

---

## P4 — Acceptance policy (quality-gated / product)

| ID | Work | Class | Spike | Kill |
|---|---|---|---|---|
| P4.1 | **Class-adaptive margin table** (math/code 3, prose 1, factual 0) | Q/P | after P0.5 | kill if mean tok/s ≤ global m1 OR any class PPL worse than global m3 factual |
| P4.2 | Entropy-adaptive margin (online, no classifier) | Q | M3 | kill if factual PPL not improved vs m3 at equal speed |
| P4.3 | Trajectory PPL governor (snap to exact on drift) | Q | M3 | kill if overhead > gain |
| P4.4 | Grammar/schema forced-accept mode (JSON/code) | P | unit + suite | product flag; no standings claim |
| P4.5 | Speculative sampling T>0 path | P | correctness first | out of standings until scope yes |

---

## P5 — Approximate / early-exit VERIFY (the paradigm break)

Only structural escape from “verify 84% at kernel ceiling.”

| ID | Work | Class | Spike | Kill |
|---|---|---|---|---|
| P5.1 | Checkpoint early-exit verify (heads at full-attn layers) | Q | after P1.6 | kill if AUC gate fails OR PPL delta > battery bar |
| P5.2 | SPRINTER-class accept-predictor + periodic full audit | Q | train on trajectory logs | kill if audit catch rate insufficient or PPL bar fail |
| P5.3 | Two-stage verify (cheap mc/sketch → mm2d on uncertain) | Q | M3 | kill if wall not down at iso-quality |
| P5.4 | Confidence-gated commit (margin mode only) + audit | P/Q | M3 | kill if factual PPL cliff |

**Audit protocol (all P5):** fixed audit rate R∈{1/8,1/16}; log false-accept rate; teacher-forced PPL vs exact; never default-on without user gate.

---

## P6 — Hybrid-native structure

| ID | Work | Class | Spike | Kill |
|---|---|---|---|---|
| P6.1 | Selective GDN skip when gate≈0 (after P1.5) | E→Q | kernel flag | kill if <10% skip mass or PPL move |
| P6.2 | Mid-layer feature draft (taps from target mid stack; NOT drop-attn self-spec) | Q | tiny head | kill if τ ≤ DSpark |
| P6.3 | Attention-hint / skip SDPA when state stable (long-ctx only) | Q | measure first | kill if short-ctx standings unaffected and long-ctx unmeasured |
| P6.4 | REST-style state predictor (research) | Q | Modal | kill unless P1.5 shows strong state autocorrelation |

---

## P7 — Drafter / training (Modal-gated)

| ID | Work | Class | Spike | Kill |
|---|---|---|---|---|
| P7.1 | **Unpacked vs Q2_0 tap-hidden fidelity** (still owed; blocks width-7 narrative) | — | Modal or local dequant | settles mismatch theory |
| P7.2 | Width-7 deployment-faithful retrain | Q | only if P7.1 + 120k pilot EV | user go/no-go; parked by default |
| P7.3 | EAGLE-3-style head on existing taps (compose with DSpark) | Q | small Modal | kill if τ ≤ width-4 DSpark |
| P7.4 | Weaver embed/unembed share adapter | Q | eng | kill if propose cut < 5 ms without τ loss |
| P7.5 | τ-aware distillation objective (not CE proxy) | Q | Modal | kill if suite τ flat like r2 |
| P7.6 | Weak-class corpus rebalance | Q | Modal | after fidelity; MiniCPM lesson |
| P7.7 | On-device overnight self-distill / trajectory mining | Q | M3 idle | kill if no τ move in 1 week data |

---

## P8 — Deep tree (after P1.1 positive)

| ID | Work | Class | Spike | Kill |
|---|---|---|---|---|
| P8.1 | Sequoia-DP / confidence-placed branch (not uniform root runner-up) | E | code | kill if ≤ naive tree |
| P8.2 | B7 GDN masked-solve (Trees-from-Marginals) M3 spike | E | micro then e2e | kill if no win vs capture rollback at tree width |
| P8.3 | KVBuffer deferred lin-attn state for wide trees | E | after P8.2 | memory enabler only |
| P8.4 | Entropy-triggered depth (branch when flat) | E/Q | after P8.1 | kill if mean regresses |

---

## P9 — Product surface (not standings-primary)

| ID | Work | Class | Spike | Kill |
|---|---|---|---|---|
| P9.1 | Multi-turn state reuse — **measure tax first** | P | 1-day eval | always do measure |
| P9.2 | Multi-turn implement (KV+GDN+conv persist) | P | after P9.1 | — |
| P9.3 | MTLBinaryArchive pipeline cache (cold TTFT) | P | 2-day | kill if cold TTFT < −5% |
| P9.4 | Memory envelope 16GB verdict | P | eval procedure | ship/no-ship 16GB |
| P9.5 | Latency surface (TTFT, position curve, sustained) | P | eval | publish both burst/sustained |
| P9.6 | M4 Max re-profile (spec ~break-even there) | P | M4 | separate OP doc |
| P9.7 | Detokenize batching / host residual | P | if census says so | kill if <0.2 ms/tok |

---

## P10 — Gated bets / roadmap

| ID | Work | Class | Gate |
|---|---|---|---|
| P10.1 | Modal fullk ue8m0-from-**original** bf16 | Q | explicit user $ + PPL bar |
| P10.2 | Width-7 full retrain 400k-class | Q | P7.1 + pilot + user $ |
| P10.3 | M5 int8 / matrix unit path | roadmap | hardware in fleet |
| P10.4 | Affine-2bit MLX-format migration | Q | unpark condition A ticket |
| P10.5 | Async cross-engine draft/verify | E | only if P5 creates GPU bubbles |
| P10.6 | imatrix / mixed higher-bit sensitive linears | Q | quality battery |

---

## Execution order (dispatch batches)

Conflict-free parallel batches (scopes approximate):

**Batch 0 (now, local):** P0.2 P0.3 P0.4 P0.6  
**Batch 1 (M3 referee day):** P0.1 P0.5 P1.1 P1.2 P1.3 P1.5 P1.7  
**Batch 2 (parallel after B1):** P1.4 P1.6 P2.3 P3.5 P9.1 P9.4  
**Batch 3 (code):** P3.1 P3.2 P2.1 P2.2 P4.1  
**Batch 4 (depends):** P5.1←P1.6; P6.1←P1.5; P8.∗←P1.1; P7.1 anytime Modal  
**Batch 5 (user-gated $):** P7.2 P7.3 P7.5 P7.6 P10.1 P10.2  

---

## Economics cheat sheet (for prioritization, not exclusion)

| Move | Hits which term | Ceiling sketch |
|---|---|---|
| +0.5 accept/round at fixed 229 ms | numerator | ~+11% tok/s |
| −20% verify via early-exit on 50% rounds | 84% term | ~+9% if quality holds |
| Tree alt-win on 15% hard rounds +0.5 tok | numerator | class-local |
| Scheduler truncate dead tail | verify m | few % |
| Bitplane decode +5% plain | floor | plain 14.5→~15.2; spec indirect |
| Propose −50% | 13% term | ≤ +7% hard cap |
| Fullk +15% mm2d | verify | ~+10–12% if PPL OK |

Under verify-dominated regime, **acceptance and verify-skip dominate propose micros** — but propose/scheduler still run because they are cheap and compose.

---

## Reporting contract

Every spike comment on its ticket must include:

1. Exactness class (E/Q/P)  
2. Blessed config + ruler version  
3. Regime tags (M, machine, margin, drafter)  
4. Measured vs inferred  
5. Kill criterion result (pass/kill/extend)  
6. What it does **not** prove  

Campaign dashboard = this file + epic rollup + ticket comments (append-only).

---

## Mapping to existing tickets

| Program ID | Existing ticket(s) |
|---|---|
| P0.5 | `eval-quality-reference-battery` |
| P1.1 P8.* | `tree-speculation-over-dspark` |
| P1.2 | `ngram-draft-source-mux` |
| P2.1 | child of `bitplane-popcount-twotier-verify` / `wider-unpack-weight-code` |
| P2.3 P10.4 | `eval-mlx-quantization-format-migration-spec-lane-unpark-condition-a` |
| P3.1 | new + `implement-dspark-hardware-aware-prefix-scheduler` |
| P4.1 P4.2 | `prompt-class-adaptive-drafting`, `relaxed-typical-acceptance-mode` |
| P5.* | `sprinter-approx-verify-audit` + new early-exit |
| P6.1 | new `selective-gdn-compute-skip` |
| P7.1 P7.2 | `drafter-width7-retrain-bonsai` |
| P7.3 | `eagle3-drafter-upgrade` |
| P7.4 | `weaver-feature-reuse-adapter` |
| P8.2 | `gdn-rollback-free-masked-solve` |
| P8.3 | `kvbuffer-deferred-linattn-state` |
| P9.1 P9.2 | `eval-multiturn-state-reuse` |
| P9.3 | folded into `eval-latency-surface` / new |
| P9.4 | `eval-memory-envelope` |
| P10.1 | `mm2d-fullk-pow2-requant-verify` |
| P10.3 | `m5-matrix-unit-roadmap` |

New tickets created alongside this doc for gaps (see git).

## Measured results log (append-only)

### 2026-07-19 — v3 baseline + flag battery (M3, Q8_0)

- **v3 baseline:** plain 14.47 / exact 15.35 (byte-match) / m1 18.19 / m3 20.09
- **Tree TW=3 exact:** KILL −12–15% tok/s (alt_wins real but cost-bound)
- **PLD m1 prose:** KILL −20%
- **FUSED_VERIFY_ARGMAX m1:** KILL ~0%

- **Oracle width EV:** prefix-width admission 0% under flat mm2d; **skip-on-zero-accept +12–22%**. Scheduler rescoped to skip/handoff not width.
- **ORACLE_LOG** env shipped for future EV work.

- **Deep analysis** (`flag-battery-deep-analysis-2026-07-19.md`): tree tax = verify 56% + rollback 38%; PLD = verify-tail skew; skip-on-zero +9–22% on real walls; flat cost corr=−0.72; host CPU ~89% both chain/tree (delta not host).

- **Skip policies LIVE REFUTED:** `--skip-low-conf 1.0` −25%; `--skip-after-reject` −5–9% exact/m1. Oracle +22% was non-causal (inplace zero after seeing it). Reactive is wrong time offset. Conf thr lacks precision. Flags default OFF. See bonsai-gguf-port-specscheduler ticket.

- **Accept-probe AUC (P5 prereq):** 424 rows; best layer RMS AUC ~0.63; **conf AUC 0.79 < 0.85 kill**. Early-exit on checkpoint scalars NOT unparked. `LMBRRR_ACCEPT_PROBE` shipped.

- **adapt-margin:** shipped `--adapt-margin 1,2`. N=128 prose ≈m3 tps with better PPL than m3; N=96 multi-prompt mixed (often m1-class). Opt-in only.
- **bitplane m=1 exact:** blocked without act-quant; isolated int4 bitplane ~107–122 GB/s vs mv 106.
