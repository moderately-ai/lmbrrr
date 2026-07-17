"""Phase 2 (CAPTURED) of the two-phase clean capture: load the pre-quantized inputs
and run ONLY the qmm in a loop. No RNG, no quantize in the captured region -> the
gpudebug timeline is ~all affine_qmm_t, so the (timeline-global) counters describe
the matmul. Confirm purity afterward: gpu_write_bandwidth << gpu_read_bandwidth."""
import mlx.core as mx

d = mx.load("/tmp/qmm_inputs.safetensors")
wq, s, b, x = d["wq"], d["s"], d["b"], d["x"]
mx.eval(wq, s, b, x)

def call():
    return mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=128, bits=2)

for _ in range(5):     # warmup (captured but small vs the 25 below)
    mx.eval(call())
for _ in range(25):    # the bulk: dominates the timeline so counters are qmm
    mx.eval(call())
print("done")
