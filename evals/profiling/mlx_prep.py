"""Phase 1 (UNCAPTURED) of the two-phase clean capture: synthesize + quantize +
persist the qmm inputs, so the captured phase does ZERO RNG / ZERO quantize --
only the matmul. Without this, a gpudebug capture is ~80% mx.random.normal RNG and
the timeline-global counters read the RNG, not the kernel (rigor protocol gate 2).

Edit n,k,m for the shape/batch under study. Default: gate_up 34816x5120, m=32
(m>=10 => the qmm dequant-once path on M3-tier 'g')."""
import mlx.core as mx

n, k, m = 34816, 5120, 32
w = mx.random.normal((n, k)).astype(mx.bfloat16)
wq, s, b = mx.quantize(w, group_size=128, bits=2)   # 2.125 bpw == Q2_0
x = mx.random.normal((m, k)).astype(mx.bfloat16)
mx.eval(wq, s, b, x)
mx.save_safetensors("/tmp/qmm_inputs.safetensors", {"wq": wq, "s": s, "b": b, "x": x})
print("saved", wq.shape, wq.dtype, "| s", s.shape, "| x", x.shape, x.dtype)
