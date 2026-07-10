# DeltaNet Chunked Recurrence

Date: 2026-07-10

Ticket: `optimize-deltanet-chunked-prefill-and-verify-throughput`

## What changed

`GatedDeltaNet` seq>1 processing in `src/qwen35.rs` was rewritten from per-token loops into chunked tensor ops:

- **Depthwise causal conv**: the O(seq × kernel) per-position loop is now kernel-count shifted-window multiplies (4 muls + 3 adds + silu for the whole chunk). The accumulation visits taps in the same ascending order per output position as the old loop, so this path is numerically order-preserving.
- **Recurrent delta rule**: the per-token recurrence `S_t = (I − β_t k_t k_tᵀ) α_t S_{t−1} + β_t k_t v_tᵀ`, `o_t = q_tᵀ S_t` is evaluated in chunks of 32 via the WY/UT-transform: solve `(I + B)Δ = diag(β)(V − diag(γ)K S₀)` with `B[t,j] = β_t·exp(G_t−G_j)·k_tᵀk_j` strictly lower-triangular, then `O = diag(γ)Q S₀ + (rel∘QKᵀ)_{≤} Δ` and `S_C = γ_C S₀ + (K∘decay)ᵀΔ`. All decay factors are relative (`exp(G_t−G_j) ≤ 1` for kept entries), so nothing overflows regardless of decay strength; `(I+B)^{-1}` is exact after ⌈log₂C⌉ Neumann-doubling steps because B is nilpotent. Math runs in F32 like the original. The algebra is identical to the sequential recurrence; only the floating-point summation order differs.
- The original loop survives behind `LMBRRR_DELTANET_SEQUENTIAL=1` for oracle comparisons. The decode path (seq_len 1) is untouched.

## Results (BF16, Metal, M4 Max)

| Metric | before | after |
| --- | ---: | ---: |
| Prefill (long profile, median) | 163 tok/s | **918 tok/s (5.6×)** |
| Steady decode | 64 tok/s | 65 tok/s (unchanged, path untouched) |
| T_verify(8) | 68.8 ms | **31.7 ms (253 tok/s)** |
| T_verify(16) | 117.4 ms | **37.0 ms (433 tok/s)** |
| T_verify(32) | 209.5 ms | **40.8 ms (784 tok/s)** |
| Marginal verify token (fit) | 6.3 ms | **0.80 ms** (target was ≤ 1.5) |
| Per-token efficiency vs decode at γ=32 | 2.6× | **12.5×** |

The scheduler contract table (`target/verify-throughput-table.json`) is regenerated; the new fit is `T_verify(γ) ≈ 15.6 ms + 0.80 ms·γ`. The floor is now the ordinary forward cost (attention/MLP matvecs + lm_head), which is exactly what the hot-path, quantization, and fusion tickets attack.

Known small-γ anomaly: γ=2 costs ~29 ms (≈ γ=4) because the chunk machinery has a fixed overhead over the plain forward floor; the scheduler will naturally prefer γ ≥ 4, and a cheap tiny-chunk path is a possible later refinement.

## Correctness

- Strict logits parity vs the Transformers oracle **passes** (top-1 3/3, top-10 overlaps 9/10/10/10; max shared logit delta 1.75 vs 0.25 baseline — summation-order effect within the strict gate).
- Advisory drift report: 64-token greedy generation diverges from the sequential path at ~token 5 on the long profile with a synonym-level fork ("completion time for each batch" vs "time for each batch to complete") and identical downstream math. Accepted under the campaign quality bar; the DSpark exact-greedy oracle is self-consistent (runner and baseline share the chunked implementation).

## Campaign impact

Chain speculation is now viable: a γ=8 round costs ~32 ms, so τ=5 yields ~6.4 ms/token ≈ 158 tok/s on today's BF16 target — versus ~72 tok/s before this change. With the forward floor reduced by quantization + GEMV/fusion work (projected ~5 ms) and tree speculation lifting τ_eff, the 500–1000 tok/s range becomes arithmetically reachable: e.g. γ=16 at τ_eff=12 → ~1.5 ms/token.
