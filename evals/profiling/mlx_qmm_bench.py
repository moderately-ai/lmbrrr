"""Black-box wall-time sweep of MLX's quantized qmm/qmv vs our matmul2d verify kernel.

Key dispatch fact (mlx/backend/metal/quantized.cpp): MLX picks the memory-bound
QMV path when M < vector_limit and the QMM simdgroup path when M >= vector_limit.
For M3-class GPUs (arch tier 'g', not 'd') on these large shapes
get_qmv_batch_limit(D,O,dev) == 10, so m<=8 measures QMV and m>=10 measures QMM.
=> the batch sweep MUST cross 10 or you never measure the dequant-once qmm path.
(Rigor protocol gate 2: verify you measure what you think you measure.)

Reports ms/call plus three throughput views:
  eff GB/s  = OUR Q2_0 metric (n*k*2.125/8 bytes) -- directly comparable to 43.
  raw GB/s  = packed weight bytes actually moved (n*k*bits/8).
  GFLOP/s   = useful arithmetic (2*m*n*k) -- this is what grows with m on QMV.
bits=2 group_size=128 == Q2_0's 2.125 bpw exactly (2 + 16/128).

Runs on the M3 referee (needs MLX). See README.md for the debuggable-python setup.
"""
import time
import mlx.core as mx

SHAPES = [
    ("o_proj",   5120,  5120),
    ("gate_up",  34816, 5120),
    ("ffn_down", 5120,  17408),
]
BATCHES = (1, 5, 8, 10, 16, 32)          # <-- crosses the qmv/qmm boundary (10)
CONFIGS = [(2, 128), (2, 64), (4, 64), (8, 64)]
Q2_BYTES = lambda n, k: n * k * 2.125 / 8   # our effective-bandwidth denominator

def sync():
    fn = getattr(mx, "synchronize", None)
    (fn() if fn else mx.eval(mx.array(0.0)))

def bench(n, k, m, bits, gs):
    w = mx.random.normal((n, k)).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=gs, bits=bits)
    x = mx.random.normal((m, k)).astype(mx.bfloat16)
    mx.eval(wq, scales, biases, x)
    def call():
        return mx.quantized_matmul(x, wq, scales, biases, transpose=True,
                                   group_size=gs, bits=bits)
    for _ in range(8):
        mx.eval(call())
    sync()
    N = 200
    t = time.perf_counter()
    for _ in range(N):
        mx.eval(call())
    sync()
    dt = (time.perf_counter() - t) / N
    return (dt * 1000.0,
            Q2_BYTES(n, k) / dt / 1e9,          # eff (our metric)
            (n * k * bits / 8) / dt / 1e9,       # raw packed-weight GB/s
            (2 * m * n * k) / dt / 1e9)          # GFLOP/s

print(f"MLX {mx.__version__} on {mx.default_device()}")
try:
    print("device_info:", mx.metal.device_info())   # confirms arch -> vector_limit
except Exception as e:
    print("device_info unavailable:", e)
print(f"{'shape':>10} {'n':>7} {'k':>7} {'m':>3} {'bits':>4} {'gs':>4} "
      f"{'ms/call':>9} {'eff GB/s':>9} {'raw GB/s':>9} {'GFLOP/s':>9}")
for label, n, k in SHAPES:
    for bits, gs in CONFIGS:
        for m in BATCHES:
            try:
                ms, eff, raw, gf = bench(n, k, m, bits, gs)
                print(f"{label:>10} {n:>7} {k:>7} {m:>3} {bits:>4} {gs:>4} "
                      f"{ms:>9.3f} {eff:>9.1f} {raw:>9.1f} {gf:>9.1f}")
            except Exception as e:
                print(f"{label:>10} {n:>7} {k:>7} {m:>3} {bits:>4} {gs:>4}  ERR {e}")
    print()
