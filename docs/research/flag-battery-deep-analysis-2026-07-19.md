# Deep analysis: v3 baseline + flag battery + oracle (2026-07-19)

Instruments used (per `metal-gputrace-cli` / `metal-live-gpu-probe` / rigor protocol):

1. **In-loop timers** (authoritative for tok/s identity): `verify_seconds`, `propose_seconds`, `rollback_seconds`, `round_wall_ms`, `round_accepted`
2. **Identity check**: `(accept+1) / med_wall` vs measured tok/s
3. **Factorization** of Δtps into accept-only vs wall-only
4. **Live host CPU probe** during chain vs tree (ps pcpu)
5. **xctrace Metal System Trace** 12s attach (intervals + perf-state) — qualitative; leaf-interval sums under-count long matmuls (known limitation of label granularity), so **do not** use xctrace leaf sums as absolute GPU busy for this stack

Receipts: `m3:/tmp/v3_baseline_20260719_162904`, `/tmp/flag_battery_20260719_163520`, `/tmp/oracle_*.out`, `/tmp/live_probe_20260719_165801`.

---

## 1. v3 baseline — identity and flat-cost model

| arm | tps | accept | verify% | propose% | rollback% | med_wall_ms | identity_tps | gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plain | 14.47 | — | — | — | — | — | — | — |
| exact | 15.35 | 2.342 | 83.0 | 13.6 | 2.1 | 218.7 | 15.28 | +0.4% |
| m1 | 18.19 | 2.969 | 83.1 | 13.5 | 1.5 | 218.7 | 18.14 | +0.3% |
| m3 | 20.09 | 3.448 | 83.0 | 13.6 | 1.3 | 218.4 | 20.37 | −1.4% |

**Finding A — flat verify cost CONFIRMED on v3.**  
`corr(accept, wall) = −0.72` on m3 (full accepts slightly *faster* 218 vs 222 ms). Wall is **not** proportional to accepted count. Round cost ≈ constant ~218 ms. Throughput moves almost only with `(accept+1)`.

**Finding B — component shares stable.** Verify ~83%, propose ~13.5%, rollback 1.3–2.1%, overhead ~1.5%. Propose is **not** free but is second-order.

**Finding C — zero-accept mass is the skip lever.**  
exact: 26.3% rounds a=0; m1: 12.5%; m3: **0%**.  
Skip-on-zero simulation using **real walls** + plain 69.1 ms:

| arm | fixed tps | skip0 tps | lift | time fraction on a=0 |
|---|---:|---:|---:|---:|
| exact | 15.23 | 18.56 | **+21.9%** | 26.2% |
| m1 | 18.05 | 19.75 | **+9.4%** | 12.5% |

This **validates the oracle** without assuming flatness for the zero case: replacing a full 218 ms reject round with a 69 ms plain step is the only width-policy win under flat mm2d.

---

## 2. Tree TW=3 — mechanism (not just “slower”)

| | prose Δ | code Δ |
|---|---:|---:|
| tok/s | **−15.2%** (13.00 vs 15.34) | **−12.1%** (14.64 vs 16.65) |
| accept | −0.167 | −0.102 |
| +verify total | **+832 ms (55.5% of Δdecode)** | +610 ms (57.8%) |
| +rollback total | **+570 ms (38.0%)** | +403 ms (38.2%) |
| +propose total | +196 ms (13.1%) | +152 ms (14.4%) |
| med_wall | +16.0 ms/rd | +13.9 ms/rd |

**Factorization (identity model):**

| | prose | code |
|---|---:|---:|
| total identity Δ | −11.5% | −8.6% |
| accept-only (wall fixed) | −5.0% | −2.8% |
| wall-only (acc fixed) | −6.8% | −6.0% |

**≈ half accept, half cost** — both legs real. Alt_wins (1 prose / 3 code per run) do not pay for the tax.

**Wall bimodality (tree only):**

| arm | lo med (<250ms) | hi med (≥250) | hi_frac |
|---|---:|---:|---:|
| tree prose | 231.8 | **262.3** | **47.5%** |
| tree code | 231.3 | 262.3 | 33.3% |
| chain (both) | ~219 | — | 0% |

Full TW=3 accept (a=3) sits on the **cheap** mode (~232 ms). Partial accepts (a=0..2) sit on **~262 ms** — consistent with extra rollback reconstruction when the tree does not fully accept. Chain has no high mode.

**Live host CPU (N=64):** chain med_tail **89.1%**, tree **88.4%** — both host-hot on encode, **not differentiated**. Tree tax is not “more host”; it is longer GPU-synchronized verify+rollback (in-loop timers).

**xctrace 12s attach:** both arms GPU Performance-state present; leaf interval sums are dominated by `blit_to_cpu` labels and under-count long matmuls (tooling limit). Do not invert the in-loop verify attribution from leaf sums.

**Verdict (strengthened):** naive `--tree` KILL stands. Any future tree must (1) cut rollback path cost (B7 / per-position state) **and** (2) raise alt-win rate enough that accept-leg alone exceeds ~+7% wall tax. Current alt_wins≪1/round fails both.

---

## 3. PLD — mechanism

| | m1+PLD | m1 no PLD | Δ |
|---|---:|---:|---:|
| tps | 14.53 | 18.18 | **−20.1%** |
| accept | 2.556 | 2.969 | −0.413 |
| verify_s | +1757 ms | | **99.3% of Δdecode** |
| med_wall | 220.1 | 219.0 | +1.1 (median hides tail) |
| identity vs measured gap | **−10.1%** | +0.3% | skew |

**Long walls:** 15/108 PLD rounds >300 ms (up to **381 ms**), **0/96** without PLD.  
Those long walls are reject-heavy PLD attempts (propose width 8 filter) that still pay a fat verify.  
pld_accepted=4 over 5 pld_rounds — almost no successful copy.

**Verdict (strengthened):** PLD KILL. Not a small constant tax — a **skewed tail of ~380 ms reject rounds** that destroys mean tok/s while barely moving median wall. Any future PLD needs (a) conf/scheduler gate so PLD never fires unless copy likely, and (b) capped verify width on copy attempts.

---

## 4. Fused verify argmax

Δtps **−0.16%**, accept identical, verify +6 ms noise.  
Host CPU not measured separately; component timers show no real lever.  
**KILL** for ship bar ≥+1% stands.

---

## 5. Oracle / scheduler — corrected economics

Under flat mm2d:

| policy | oracle lift |
|---|---:|
| Prefix-width truncate to exact_prefix | **0%** (same tokens, same wall) |
| Conf-threshold width | **negative** (cuts good tails) |
| Skip draft when exact_prefix==0 | **+9–22%** (real walls) |

Conf head **does** separate means (accepted conf ~2.4 vs rejected ~0.1–0.6) — useful for **zero-round prediction**, not for width.

**Rescope confirmed by deep analysis:** implement skip/chain-handoff, not Appendix-A width scan.

---

## 6. What this implies for the program (priority update)

| Priority | Item | Why analysis raised/lowered it |
|---|---|---|
| **P0 done** | v3 baseline | identity closes; flat cost holds |
| **↑↑ P1** | skip-on-zero / chain-handoff scheduler | +9–22% real-wall EV; m3 has 0% zeros so win is on exact/m1 and weak classes |
| **↓ tree default** | naive TW=3 | 38% of tax is rollback; 55% verify m=7; alt_wins insufficient |
| **↓ PLD default** | ungated | 99% tax in verify tail; need gate before retry |
| **→ P8 tree** | only after B7 + better placement | must attack rollback leg first |
| **→ bitplane decode** | still live | plain floor 14.47; independent of spec flags |
| **→ early-exit verify** | still live | only way to cut the 83% verify share |

---

## 7. Method notes (for next runs)

1. Always report **component Δms and share_of_Δdecode**, not only tok/s%.  
2. Always report **identity gap** — large negative gap ⇒ skewed walls (PLD lesson).  
3. Factorize accept vs wall before declaring “cost” or “quality”.  
4. Live host CPU distinguishes host-bound vs GPU-bound **deltas**; here both ~89% so delta is GPU-sync path.  
5. xctrace leaf sums are **not** a substitute for in-loop verify_seconds on this stack; use xctrace for gaps/DVFS, gpudebug embed for isolated kernels.  
6. For skip-on-zero claims, simulate on **empirical** `round_wall_ms` + plain ms, not only analytic flat model.

## 8. Live skip-policy A/B (follow-up)

Implemented `--skip-low-conf` and `--skip-after-reject`. **Both lose live** (see scheduler ticket). Corrects §5: offline skip-on-zero EV is **not** a ship lever without a high-precision causal zero predictor. The +9–22% figures remain valid only as an *oracle upper bound* on replacing zero rounds, not as a policy EV.

## 9. Accept-probe AUC (P5)

Checkpoint hidden scalars (L15/31/47/63 rms/mean/maxabs) AUC ≤0.63 for on_accept_prefix. Drafter conf AUC 0.79 (best) still &lt;0.85 bar. Early-exit verify not unparked on simple features.
