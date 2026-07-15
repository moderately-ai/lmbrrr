---
id: ternary-type42-dequant
title: "FEATURE: ternary type-42 (+ type-41) dequant/quant in the candle fork"
status: in-progress
priority: p2
dependencies: [spike-ternary-type42-block-format]
related: [ternary-bonsai-27b-support, metal-ternary-matmul-kernel, gguf-loader-qwen35-hybrid]
scopes: [candle-fork]
shared_scopes: []
paths: []
tags: [ternary-bonsai, model-compat, fork]
---
## WHY

Once the type-42 block layout is known ([[spike-ternary-type42-block-format]]), the fork must be able to READ it: register the ternary type so `gguf_file`/`ggml_file` ingest doesn't reject code 42, and provide a correct CPU dequant (type42 → f32/bf16). This is the reference path that gates the loader and is the correctness oracle for the Metal kernel.

## WORK ITEMS

1. Add the ternary type to the fork's `GgmlDType` (or a fork-local extension) with its block size / type-size, so `qtensor_from_ggml` and the GGUF reader accept it. Fail loudly on unknown variants.
2. Implement `dequantize` (block → f32/bf16) matching the Python reference from the spike bit-for-bit (or within documented rounding).
3. A fork test: dequant a fixture type-42 block == reference vector; and a whole-tensor test vs the F16 reference for one Bonsai weight.
4. Decide the QTensor representation: keep ternary packed (for the Metal kernel to consume directly) vs eager-dequant-to-bf16 at load. Packed is required for the memory/bandwidth win; expose both so the loader can start on dequant-to-bf16 while the kernel lands.

## DONE-WHEN

The fork loads a type-42 GGUF tensor and produces a bf16 tensor matching the F16 reference within ternary error; `cargo nextest` fixture + whole-tensor tests pass. Feeds [[gguf-loader-qwen35-hybrid]] (functional path) and [[metal-ternary-matmul-kernel]] (oracle).

## IMPLEMENTED (2026-07-15) — candle fork `tomsanbear/candle` branch `ternary-q2_0` (commit e1197ca9)

Added BOTH prism-ml types (Q2_0 type 42 ternary AND its sibling Q1_0 type 41 binary — the Bonsai 1-bit phone companion), full `GgmlType`, not a stub:
- `BlockQ2_0 {f16 d; u8 qs[32]}` = 34 B/128 (2.125 bpw); `BlockQ1_0 {f16 d; u8 qs[16]}` = 18 B/128 (1.125 bpw).
- `to_float` (dequant) — Q2_0 `w=(q-1)·d` codes 00→-1 01→0 10→+1 11→+2; Q1_0 sign→±d. Validated cosine 1.0 vs the real F16 weights (spike) + unit tests on exact bit patterns.
- `from_float` (quantize) — faithfully mirrors the fork's `quantize_row_q2_0_ref` (d = max|w|, `q = clamp(round(w/d)+1, 0, 3)`) and `quantize_row_q1_0_ref` (d = mean|w|, sign bit). NOT stubbed — implemented + round-trip unit-tested. (It's a standard round-to-grid quantizer; prism's *training-aware* pipeline is a separate thing, but the quantize step itself is well-defined and now supported.)
- `GgmlDType::{Q1_0,Q2_0}` variants + `from_u32(41/42)`/`to_u32`/`type_size`/`block_size`/`cpu_zeros`/`from_data` + Cpu/Metal/Cuda load arms + the Metal dequant read-back path.
- Quantized-*matmul* dispatch panics with a clear message (no packed Metal kernel yet — that's [[metal-ternary-matmul-kernel]]); deployment path is dequant-to-bf16 first. 128-block types are excluded from `verify_block_sizes!` and the CPU matmul-error set (their 32-block Q8_0 VecDotType mismatch — not the deployment path).
- `cargo check` (metal + non-metal + tests) clean; 4 new nextest tests pass (to_float bit-patterns + from_float round-trips, both types).

REMAINING: repin lmbrrr `Cargo.toml` candle rev to this fork commit once the branch is pushed (integration), then a whole-tensor load test through lmbrrr's GGUF path ([[gguf-loader-qwen35-hybrid]]).
