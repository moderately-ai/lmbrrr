---
id: eval-apples-to-apples-qmv
title: "EVAL: apples-to-apples 4-bit matvec — candle vs llama.cpp vs MLX, same machine/session/shapes"
status: closed
priority: p1
dependencies: []
related: []
scopes: [evals, candle-fork]
shared_scopes: []
paths: []
tags: [eval-wave, kernels]
---
WHY: the 'their kernels hit 250-270 GB/s' evidence is DERIVED (published tok/s x model bytes, other machines, other models, secondary sources +/-15%). Our 196 GB/s is from our own trace. With +/-35% ambient drift demonstrated on this box, a kernel-rewrite decision needs a same-machine, same-session, same-shape measurement. This eval also cleanly separates KERNEL gap from FORMAT gap: llama.cpp runs the SAME q4_K format (pure kernel comparison), MLX runs its affine-4bit format (kernel+format-as-a-unit comparison — the 'replace both' decision input).

PRECONDITIONS: eval-protocol-ambient-control protocol (load < 4, no builds/agents, session control at start AND end, all comparisons within one session, interleave arms — do NOT run all-candle then all-llama.cpp; alternate per shape).

SHAPES (the deployed decode set, batch 1 / m=1): 248094x1024 (lm_head), 8192x1024 (dn_qkvz), 7168x1024 (mlp_gate_up), 1024x2048 (out_proj), 1024x3584 (mlp_down). Metric: effective WEIGHT bandwidth GB/s = weight_bytes / kernel_time (q4_K = 144B per 256 elements = 0.5625 B/elt; MLX affine-4b gs=64 = 4 bits + fp16 scale+bias per 64 = ~0.5625 B/elt too — comparable).

ARM 1 — CANDLE (ours): cd ~/workspace/github.com/huggingface/candle/candle-metal-kernels && cargo build --release --example metal_benchmarks && ./target/release/examples/metal_benchmarks nsg-sweep (records baseline + rt variants on all 5 shapes). Record the nsg=1 rows as 'candle'.

ARM 2 — LLAMA.CPP (same q4_K format = pure kernel test): git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llamacpp && cd /tmp/llamacpp && cmake -B build -DGGML_METAL=ON && cmake --build build -j --target test-backend-ops. Their per-op bench: ./build/bin/test-backend-ops perf -o MUL_MAT -b Metal. IMPORTANT: the default case list does NOT include our shapes — edit tests/test-backend-ops.cpp: find the perf section that constructs test_mul_mat cases (search 'perf' and 'test_mul_mat') and add cases: type_a=GGML_TYPE_Q4_K, type_b=GGML_TYPE_F32, (m,n,k) mapped to their convention — THEIR mul_mat is (ne00=k, ne01=n_rows, ne11=1): add {Q4_K, F32, k=1024, n=248094, batch 1} and the other four shapes. Rebuild, rerun, record us/run and convert to GB/s (weight_bytes = n*k*0.5625). CAVEAT for the executor: their src1 is F32 (not bf16) — note it; the weight-read side dominates so the comparison stands, but record the difference. Also record their kernel's geometry from the build (N_R0_Q4_K/N_SG_Q4_K in ggml/src/ggml-metal/ggml-metal-impl.h) so the number is tied to a known kernel config.

ARM 3 — MLX (format+kernel as a unit): pip install mlx (or uv). Python bench, run per shape:
import mlx.core as mx, time
def bench(n, k, its=200):
    w = mx.random.normal((n, k)).astype(mx.float16)
    wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
    x = mx.random.normal((1, k)).astype(mx.float16)
    def f():
        y = mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=64, bits=4)
        mx.eval(y)
    for _ in range(20): f()  # warmup incl. shader compile
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(its): f()
    mx.synchronize(); dt = time.perf_counter() - t0
    wb = n*k*0.5625
    print(n, k, f'{1e3*dt/its:.3f} ms', f'{wb*its/dt/1e9:.1f} GB/s')
CAVEAT: per-call mx.eval pays MLX's dispatch+sync overhead per iteration (like our isolated protocol); ALSO run a fused-chain variant (its matmuls chained on the same stream, one eval at the end) and report both — the chained number is the fair one vs our many-dispatches-per-buffer protocol. MLX activations fp16 (not bf16) — note it.

ANALYSIS + DECISION: build the 5-shape x 3-stack table, ratios vs candle within-session. (a) llama.cpp q4_K >= 1.25x candle on any deployed shape -> the kernel gap is REAL and portable; promote porting their geometry (N_R0=2/N_SG=2 + their y-staging) into q4k-mv-round3 as the primary variant, with their measured number as the target. (b) llama.cpp ~= candle at our shapes -> the 250-270 GB/s literature numbers were a K=4096 shape effect, the short-K tax is universal, and round-3 proceeds on rt2/N_DST=2 expectations only (+10-15%). (c) MLX >= 1.4x candle where llama.cpp is not -> the FORMAT (unpacked fp16 scales, no 6-bit decode, linear nibbles) is the lever -> file a format-migration eval (requantize at load into MLX affine-4b; needs a quant-quality ladder since grouping changes, gs=64 vs q4_K superblocks — NOT bit-preserving, margin+quality gates). RECORD: full table + verdict as a comment here, cross-link to q4k-mv-round3-production-arbitrated, campaign log if it changes the roadmap.
