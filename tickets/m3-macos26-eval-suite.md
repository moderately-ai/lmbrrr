---
id: m3-macos26-eval-suite
title: "EVAL: M3/macOS-26 box battery — packed_numeric 4-bit unpack, Metal 4.1 probes, drift cross-check"
status: in-progress
priority: p1
dependencies: []
related: []
scopes: [candle-fork]
shared_scopes: []
paths: []
tags: [eval-wave, kernels]
---
BLOCKED ON: user provisioning access to the M3 Mac with latest macOS (offered 2026-07-13). WHY: MSL 4.1's `packed_numeric_type<uint4b_format, N>::unpack<half>` (spec section 2.21, Tables 2.14/2.19) converts 8 or 16 packed 4-bit values to half/float IN ONE CONSTRUCT — the only primitive found (5-agent sweep) that structurally breaks the ~3-ALU-ops-per-4-bit-element floor of our q4_K matvec (75% of the decode step). The header `metal_packed_numeric` is ABSENT on macOS 15.7 (verified: `xcrun metal -std=metal4.1` invalid; header not in toolchain 17.1). It requires the macOS 26-era toolchain.

PROCEDURE (each step has exact commands; run in order):
1. TOOLCHAIN PROBE: `xcrun metal --version` (expect Metal 4.1-capable); `echo 'kernel void t(){}' > /tmp/t.metal && xcrun metal -std=metal4.1 -c /tmp/t.metal -o /tmp/t.air && echo OK`. Then compile this probe (must succeed):
```
#include <metal_stdlib>
#include <metal_packed_numeric>
using namespace metal;
kernel void probe(device const packed_numeric_type<uint4b_format, 8>* w, device half* o, uint tid [[thread_position_in_grid]]) { vec<half,8> v = unpack<half>(w[tid]); o[tid] = v[0] + v[7]; }
```
2. LOWERING CHECK: compile with `-frecord-sources`, then inspect the AIR/native code: `xcrun metal-objdump --disassemble /tmp/p.air` (or compile a .metallib and use Xcode GPU debugger disassembly on-device). PASS = unpack lowers to O(1-2) instructions per 8 values (a real widening op); FAIL = a scalarized 8-iteration loop (then this whole line is dead — record and close).
3. MICRO-BENCH: clone the fork (github.com/tomsanbear/candle branch lmbrrr) on the M3; build `cd candle-metal-kernels && cargo build --release --example metal_benchmarks`; add an `unpack`-based variant of kernel_mul_mv_q4_K_bf16_bf16 (replace the 8 masked-fma lines with unpack<half,uint4b,16> + vectorized fma; note q4_K's nibble ordering is NOT linear — elements i,i+1,i+8,i+9 per u16 — so a load-time repack into linear nibble order may be required for unpack to apply; if so, repack in the bench's q_weight_bytes generator first and note that production adoption needs the same repack at load). Gate: outputs numerically EXACT vs baseline (uint4->half of 0..15 is exact); measure GB/s on the lm_head row (248094x1024) and body rows, within one session per the ambient-control protocol (eval-protocol-ambient-control).
4. BASELINE CROSS-CHECK: run the unmodified `nsg-sweep` task on the M3 and record all rates — tests whether the M4 box's +/-35% drift reproduces on other hardware and gives an independent baseline for the 121-vs-196-vs-267 GB/s triangulation.
5. BONUS PROBES while on the box: `-std=metal4.0` tensor APIs availability; `MTLDevice.supportsFamily` output via a 5-line Swift/objc probe.

DECISION RULE: unpack variant >= 1.3x baseline on the head shape within-session -> file the production port + plan the M4 macOS-26 upgrade path (the fork then carries BOTH kernels, runtime-selected by OS capability). < 1.15x -> close the packed_numeric line with the disassembly as receipt. RECORD everything as a comment here + campaign log.
