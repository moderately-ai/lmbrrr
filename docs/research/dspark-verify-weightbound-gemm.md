# DSpark verify: the weight-bound small-m ternary GEMM, and the Metal 4.1 gate

Deliverable of [[dspark-bonsai-integration]] (ticket `dspark-bonsai-integration`), work-item 5 (the verify GEMM). Records why DSpark on Ternary-Bonsai-27B runs at **parity** on the M3 today, the exact structural cause (proven across five kernel variants + the Metal spec), and the concrete path to the 2–3× multiplier. Env: M3 Pro, macOS 27.0 beta (build 26A5378n), Metal toolchain `32023.883` (MetalToolchain-v17.6.109), SDK MacOSX26.5.

## The problem

DSpark spec decode only pays off if the target's **verify** (a forward at m = block_size + 1 = 5) costs about **1× a single decode**, not 5×. At high acceptance (block_size 4 ⇒ ≤5 committed tokens/round), the round is `verify + propose + overhead`; it beats plain decode only if `verify ≈ weight-bound` (one weight read serves all 5 columns).

Measured round breakdown (`gguf-run --spec-drafter` prints propose/verify/overhead seconds), quantum-computing prompt, mean_accepted 3.4–4.0/4:

| segment | share | note |
|---|---|---|
| **verify** | **81%** | target forward at m=5 |
| propose | 10% | the 3.6B drafter is cheap |
| overhead | 8% | snapshot + append_context |

So the entire question is the cost of the target's Q2_0 matmuls at m≈5. NOT the DeltaNet (it uses the chunked kernel `call_gated_delta_chunk` for 2≤l≤12, efficient). The decode baseline is 14.68 tok/s; best spec ≈ 11–12.5 tok/s (parity).

## Five verify-kernel variants — all bounded by one root cause

All numerically correct (rel_err 0.0014, or 0.0000 for the exact planar repack). Shape 17408×5120 (ffn_up), the bulk of per-layer verify traffic. Weight-bound target = the m=1 mv's ~106 GB/s (0.22 ms).

| # | kernel | m=4 | eff GB/s | bounded by |
|---|---|---|---|---|
| 1 | **mc** (`kernel_mul_mv_q2_0_mc_t`, weight shared over NC cols, per-row COALESCED read) | 0.68 ms | 35 | **compute** — the scalar ternary dot × columns; read is fine (106 @ m=1) |
| 2 | generic tile mm (`kernel_mul_mm_q2_0`, BLOCK_SIZE_N=32) | ~2 ms | — | padding waste (~4× compute on 27 dead cols) → spec 7.0 tok/s |
| 3 | from-scratch BM=8, scattered per-element reads | 2.0 ms | 11.6 | non-coalesced byte reads |
| 4 | from-scratch BM=8, coalesced per-row 32-byte bursts | 1.5 ms | 15.9 | BM=8 doesn't amortize shared-staging (16 matrix-muls per 128-block) |
| 5 | from-scratch **BM=64/BN=8** (amortized staging, no waste) | 1.1 ms | 21.6 | **non-coalesced cross-row read** ([row][block] layout: 64 rows are `nb` blocks apart) |
| 6 | **planar mm2d** (`kernel_mul_mm2d_q2_0_smallm`, [k][n_pad] repack, coalesced) | 0.78 ms | 38 | **software staging** — unpack 2-bit→half + shared-write + barrier per tile |

Kernel #6 uses the `q2_0_mm2d_planes` repack (candle `candle-core/src/quantized/metal.rs`): `codes [k, n_pad]` 2-bit n-innermost + `d [k/128, n_pad]` fp16, so reading all rows for a fixed k is contiguous. That fixed the coalescing (21.6 → 38 GB/s, and it's exactly flat m=1..8 = weight-bound-STRUCTURED), but the **hand-rolled software unpack/stage is the new ceiling** — even vectorized (uint32 code load + half4 scale loads) it plateaus at ~38 GB/s.

### The physics

A weight-bound small-m GEMM needs **coalesced weight reads AND matrix-unit tiles at once**. The Q2_0 `[row][block]` layout can't give both:
- `mc` reads **per-row coalesced** (tpb threads over one 128-block) → 106 GB/s — but the per-row scalar dot can't feed the matrix units, so it's compute-bound at m>1.
- A **matrix tile** (8×8) needs 8 rows at the same k; in `[row][block]` those rows are `nb` blocks apart → **strided, non-coalesced** (variants 2–5, capped ~21 GB/s).
- A **planar `[k][row]`** repack makes cross-row reads coalesced (variant 6), but then the *dequant/stage* must happen in MSL software → capped ~38 GB/s. The hardware `matmul2d` avoids this by decoding the packed format in silicon — which is exactly what q4_K's mm2d does (142 GB/s) and what we can't hand-roll.

## The Metal 4.1 root cause (the real gate)

candle's q4_K mm2d (`kernel_mul_mm2d_q4k`, prebuilt into `mm2d_q4k.metallib`) hits 142 GB/s because it feeds the packed weight **directly to the hardware `tensor_ops::matmul2d`** primitive as `tensor<device uint4b_format>`. The tensor/matrix unit unpacks the 4-bit lanes in hardware. That is the ONLY quantized matmul in candle/lmbrrr on `matmul2d`; every other quantized path is hand-rolled `simdgroup_matrix`.

Two research passes (M3 header inspection + test-compile; MSL spec `Metal-Shading-Language-Specification` 2026-06-04):

- **`matmul2d` weight (B) operand types:** `uint8`, `int8`, `uint4b_format`, `int4b_format` — and, **in Metal 4.1**, `uint2b_format`/`int2b_format` (2-bit). Whitelist: `MPPTensorOpsMatMul2dImpl.h:2502-2528`. The packed operand is **always B (the weight)** — no A=sub-byte row exists. That is *exactly* a weight-bound layout: activations in half/bfloat (A), ternary weights in `int2b_format` (B), accumulate to float. Ternary {−1,0,+1} encodes into `int2b_format` (2-bit signed, uses 3 of 4 codes). Constraints (MSL §2.21–2.22, §7.2): K axis in dimension 0, K a multiple of 32 (Fblock), each weight row padded to a **128-byte stride**, per-128-block scales via `tensor_blockwise<tensor_plane_scales>`, tile via `matmul2d_descriptor(M,N,K)` scoped `execution_simdgroups<simdgroups_per_threadgroup>`, C as a `cooperative_tensor` (no device barrier), apply `d·(P − rowsum)` ternary fold in the epilogue via `get_multidimensional_index`.

- **BUT 2-bit is not in the M3's shipped toolchain.** `int2b_format`/`uint2b_format` are **Metal 4.1**; the M3's toolchain `32023.883` only defines the 4-bit types in `metal_packed_numeric` (lines 15/20). A `-std=metal4.0` compile of `uint2b_format` fails ("unknown type name; did you mean 'uint4b_format'"); `uint4b_format` compiles cleanly. So the 2-bit hardware path is **spec'd (June 2026) but not landed in this toolchain build**.

**Conclusion: the 2–3× is a Metal-4.1-toolchain gate, not a design failure or a hardware wall.** When the M3 gets the Metal 4.1 toolchain, the fix is a near-mechanical mirror of the q4_K mm2d (below).

### Why not 4-bit today

`uint4b_format` matmul2d works on the M3 now. But encoding one ternary weight per 4-bit lane **doubles** the target's linears to ~13.5 GB device; with the drafter (~2.4 GB) that overflows the M3's ~13 GB GPU working set → OOM. (You can't pack 2 ternary codes into one 4-bit lane: each B-lane multiplies one activation, so packed codes would multiply the wrong activations.) A per-verify transient repack moves more bytes than it saves. So 4-bit is memory-blocked on the M3.

## Paths to the multiplier (ranked)

1. **Newer M3 Metal toolchain (Metal 4.1).** Most actionable — the types are spec'd; a newer macOS 27 / Xcode beta may already ship them. Check `xcodebuild -downloadComponent MetalToolchain` / a newer Xcode, then test-compile `uint2b_format` with `-std=metal4.1`. If it compiles → build `mm2d_q2_0.metallib` and wire (below) → 7 GB, ~142 GB/s → the multiplier. Zero waste.
2. **CUDA (Modal).** Drafter + spec loop are model-agnostic; on a GPU with headroom the 4-bit up-convert fits and the layout fight disappears — the whitepaper's 1.34×+ lives there.
3. **Larger-block-size drafter (retrain).** block_size=4 caps amortization at 5 tok/round; block_size 8–16 raises the ceiling more than any verify kernel can on the M3. Modal training.

### The mirror (when 2-bit lands or on CUDA)

Mirror the q4_K mm2d stack for Q2_0:
1. `q2_0_mm2d_planes` (candle-core, **already written**): `codes [k,n_pad]` 2-bit + `d [k/128,n_pad]` fp16. For `matmul2d` land it as `tensor<device uint2b_format>` with K in dim-0 and 128-byte row-stride padding; d via `tensor_blockwise`.
2. `mm2d_q2_0.metal`: mirror `kernel_mul_mm2d_q4k_bf16` with `int2b_format` B, half/bfloat A, and the **ternary fold** `acc += d_block·(P − rowsum)` (d is per-128 = per-4 K-tiles, vs q4_K's per-32) — the `P − rowsum` mirrors q4_K's `dsc·P − dmm·rowsum` with `dmm→d`, `dsc→d`, since `Σ(code−1)·d·a = d·(Σcode·a − Σa)`.
3. Build `mm2d_q2_0.metallib`: `xcrun metal -std=metal4.1 -c mm2d_q2_0.metal -o x.air; xcrun metallib x.air -o mm2d_q2_0.metallib` on a Metal-4.1 machine (per `scripts/build_mm2d_q4k.sh`).
4. `Source::Mm2dQ2_0` + `MM2D_Q2_0_LIB` (include_bytes) + the loader in `candle-metal-kernels/src/kernel.rs:166` (mirror the `Mm2dQ4k` `new_library_with_data` branch), + `call_quantized_matmul_mm2d_q2_0`.
5. lmbrrr: load the spec target's Q2_0 linears through the mm2d planes (spec mode only — spec never does m=1, so a planar-only target is fine and fits in one 7 GB copy). Route verify (m∈2..8) to the mm2d.

### Projected result

The hand-rolled planar kernel is already flat/weight-bound-STRUCTURED at 38 GB/s; the hardware `matmul2d` (q4_K reaches 142 GB/s on the same class of shape) removes the software-staging ceiling. Verify(m=5) ≈ weight-bound ≈ 1.3–1.5× a decode ⇒ round ≈ (1.5×68 + 41 + 33) ms ≈ 176 ms for 5 tokens ⇒ **~28–31 tok/s (~2×)** at high acceptance.

## Status of what's committed (candle rev pinned in lmbrrr `Cargo.toml`)

Kernels 1–6 were all A/B'd and the slow ones reverted; the shipped candle keeps **mc for the m∈2..7 verify** (best working), the **generic tile mm for m≥8 prefill** (a real prefill win), and **mv for m=1 decode**. `q2_0_mm2d_planes` + `kernel_mul_mm2d_q2_0_smallm` + `call_quantized_matmul_mm2d_q2_0_smallm` are committed on the candle `lmbrrr` branch (the 38 GB/s hand-rolled planar proof-of-concept, exercised by `gguf-run --bench-gemv`'s `Q2_0 PLANAR` lines). The DSpark e2e is working + byte-correct at parity.

See [[dspark-bonsai-e2e-working]], [[ternary-q2_0-gemv-exhausted]], and `metal_notes.md` §"Metal 4.x tensor/matmul2d quantized formats".
