# Ternary GGUF type-42 (`Q2_0` / Q2_0_g128) — format spec + reference kernels

Deliverable of [[spike-ternary-type42-block-format]] (ticket `spike-ternary-type42-block-format`). Ground truth from prism-ml's published sources (now cloned locally) + confirmed against the actual `Ternary-Bonsai-27B-Q2_0.gguf` bytes.

## The format (fully determined)

`prism-ml/Ternary-Bonsai-27B` stores 498 weight matrices as ggml type code **42** (`file_type 41`). It is **`Q2_0` at QK=128** (the README's "Q2_0_g128"): ternary weights ∈ {−1, 0, +1} in 2-bit slots, one FP16 scale per 128-weight group.

**Block struct** (`ggml-common.h`, prism branch — NOT master, which has QK2_0=64):
```c
#define QK2_0 128
typedef struct {
    ggml_half d;              // fp16 group scale
    uint8_t   qs[QK2_0 / 4];  // 32 bytes: 4 weights/byte, 2 bits each
} block_q2_0;                 // sizeof == 34 bytes for 128 weights
// GGML_TYPE_Q2_0 = 42  (ggml.h)
```
- **34 bytes / 128 weights = 2.125 bpw deployed** (1.71 bpw ideal: log2(3) trit + 16-bit scale / 128). Confirmed independently from the file: every type-42 tensor is exactly 34.00 B/128 elements.
- **Layout**: `d` first (2 B), then 32 B of packed codes. Weight `j` is at byte `j/4`, bit-offset `(j%4)*2`, **LSB-first** within the byte.
- **Code → value** (`dequantize_row_q2_0`): `00→−1, 01→0, 10→+1, 11→+2`; `w = (q − 1) · d`. (Ternary uses 0/1/2; code 3 = +2 is unused by the ternary quantizer.)

**Reference dequant (Python, validated against the struct):**
```python
import numpy as np
def dequant_q2_0(block: bytes) -> np.ndarray:      # 34 bytes -> 128 floats
    d = np.frombuffer(block[:2], dtype=np.float16).astype(np.float32)[0]
    qs = np.frombuffer(block[2:34], dtype=np.uint8)
    q = np.empty(128, np.int32)
    for j in range(128):
        q[j] = (qs[j // 4] >> ((j % 4) * 2)) & 0x3
    return (q - 1).astype(np.float32) * d
```

Sibling packs: `Q2_g64` (7.59 GB) = same scheme at group-64 (36 B/128, scale repeated per 64); `PQ2_0` (7.17 GB) = a repacked g128 variant (confirm same type-42 dot; likely a permuted qs layout for a wider load). `Bonsai-27B` (non-ternary repo) ships a 1-bit `Q1_0` companion.

## Reference Metal kernel (the one to base ours on)

`~/workspace/github.com/PrismML-Eng/llama.cpp` (branch `prism`), `ggml/src/ggml-metal/ggml-metal.metal`. This is a Metal kernel for **exactly our GGUF format** — the most directly portable reference (candle's Metal backend is ggml-style), and closer to us than the MLX path (different, affine, quant scheme; see below).

Core dot identity: `Σ (q−1)·d·y = d·(Σ q·y − Σy)` — precompute `sumy = Σy` once, so each element is a single FMA, no per-element subtract.

- `q2_0_dot_y<SW>` (l.3837) — single-column, **bit-decomposition**: split the 2-bit code into lo/hi bits, `d·(Σ(bit0·y) + 2·Σ(bit1·y) − sumy)`. Fixed ascending accumulation order → deterministic (matches the AR/decode path bit-for-bit).
- `kernel_mul_mv_q2_0_f32_impl<nr0,nr1,tpb>` (l.3858) — the matvec. **nr1>1 reads the bandwidth-dominant weights ONCE and reuses them across nr1 src1 columns** (expand codes to floats once, one FMA per column-element). This is the spec-decode **verify** path (nr1 = draft columns). Host variants `_nr1_{1,2,3,4}` (l.3973–4018); default-on `nr1=2` per commit #64. Register budget: `nr1*SW ≤ 32`.
- `kernel_mul_mv_ext_q2_0_f32_r1_{2..5}` (l.4418) — the mlx `qmv_wide` (`mul_mv_ext`) geometry for wider m; measured ~2.08× the n=1 cost at n=3 on M5 Pro (so the bespoke nr1 kernel wins at small m; ext is the fallback).
- `kernel_mul_mm_q2_0_f32/f16` (l.10758) — GEMM (prefill), only pays off for `ne11 ≳ 32`.
- `dequantize_q2_0` / `_t4` (l.172/191), `kernel_get_rows_q2_0` (l.10694), `quantize_q2_0` (l.255), cpy kernels.

**Porting note for [[metal-ternary-matmul-kernel]]**: this maps 1:1 onto our `mm2d` structure — the nr1 multi-column weight-reuse is the same idea as our uint4b verify path, just at 2-bit ternary. Take `kernel_mul_mv_q2_0_f32_impl` (decode nr1=1, verify nr1=2..4) as the template; the `d·(Σq·y − sumy)` identity + LSB-first unpack is the whole arithmetic.

## MLX path (the user's "mlx equivalent" question)

`PrismML-Eng/mlx` and `PrismML-Eng/mlx-swift` are cloned. MLX's `Ternary-Bonsai-27B-mlx-2bit` uses MLX's **affine group quant** (bits=2), a *different* scheme from GGUF Q2_0 (affine scale+bias per group, not a signed ternary code), so it is a weaker reference for loading this GGUF. Their fork's perf commits (`perf/dspark-qmv-wide`, "Route affine qmv_wide by bit-width and batch size", the upstream `qmv_wide` small-batch matvec #3764) live in `mlx/backend/metal/kernels/quantized.metal` — useful for the *wide-m small-batch* geometry idea, but the llama.cpp Metal Q2_0 kernel above is the one that matches our on-disk format.

## Other prism-fork assets relevant to the broader campaign

The `prism` llama.cpp branch also carries (beyond Bonsai): **DSpark on Metal** — "Metal DSpark Markov resample + quantized markov heads" (#59), unmasked tap-capture path (#63, #67); **GDN on Metal** — rows-indexed state read + snapshot write-fold (#61, #62), `feat/metal-gdn-rows-write-fold`; **`megakernel/rmsnorm-qmv-fuse`** (norm-into-matvec fusion). These are independent witnesses / references for our own MTP/DSpark, GatedDeltaNet, and fusion work — worth mining under [[upstream-fork-kernels]] and the spec-loop tickets, not just the ternary track.
