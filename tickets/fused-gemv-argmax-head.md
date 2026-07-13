---
id: fused-gemv-argmax-head
title: "EVAL: fused GEMV->argmax head kernel (exact; never materialize 248k logits)"
status: todo
priority: p2
dependencies: []
related: []
scopes: [candle-fork, runtime/candle]
shared_scopes: []
paths: []
tags: [eval-wave, kernels, campaign-1000]
---
WHY: greedy decode needs only argmax(logits); we currently (1) run the head GEMV writing 248094 bf16 logits (~0.5MB), (2) barrier, (3) run a separate argmax kernel reading them back. Literature (FlashSampling arXiv 2603.15854; From-Projection-to-Prediction 2511.17599) fuses projection + reduction, never materializing logits. EXACT by construction. Expected: small (+1-3% bench greedy — saves one dispatch + barrier + ~1MB traffic per token) but free of quality risk and stacks with everything. Also applies per-position to the spec-verify argmax path (argmax_tokens), NOT to paths needing full logits (recycle top-k harvest, --accept-margin, dense-logit diagnostics) — those keep the two-pass path.

DESIGN (two dispatches, still a win — the SECOND is tiny): kernel A tiles the 248094 rows across TGs exactly like the existing mv kernel, but each simdgroup keeps a running (max_val, max_idx) in registers instead of storing logits; per-TG winner written to a [nTG x (float,uint)] buffer. Kernel B: one TG argmax-reduces the [nTG] winners. TIE-BREAKING MUST MATCH candle's argmax semantics exactly (verify what candle's Metal argmax returns on ties — read the fast_argmax kernel in reduce.metal FIRST and replicate: lowest index wins ties, or whatever it actually does; a tie-break mismatch silently changes committed text on near-ties and will show up as text-divergence in the gate).

PROCEDURE:
1. Read candle fast_argmax semantics (candle-metal-kernels/src/metal_src/reduce.metal, impl_arg_reduce) and record the tie rule in a comment here BEFORE writing code.
2. Implement kernel A as a variant of kernel_mul_mv_q4_K_bf16_bf16 (same 4-rows-per-simdgroup arithmetic, bit-identical dot products) + kernel B. Host fn call_quantized_matmul_mv_q4k_argmax in kernels/quantized.rs returning the winning u32 index in a 4-byte buffer.
3. GATE 1 (bitwise argmax): bench-task compare — for 100 random bf16 activation vectors, fused argmax id == argmax over the baseline kernel's logits, ALL cases, including manufactured ties (duplicate rows in the synthetic weights).
4. Wire into the greedy device-chain only (generate.rs device_chain branch): model.forward_argmax() path behind env LMBRRR_FUSED_ARGMAX=1; the remap_head_id + pending/readback flow is unchanged (the fused kernel outputs the same u32 id tensor shape).
5. GATE 2: nextest 59/59; stub oracle invariance (max dev <= 0.75); committed text BYTE-IDENTICAL to the flag-off run on 'Explain how tides work.' 128 tokens (this change is exact; any text diff = tie-break bug, go back to step 1).
6. MEASURE: rotated bench A/B per the ambient-control protocol; ship if >= +1% non-overlapping (low bar because zero quality risk and negative dispatch count), else record + keep behind env flag.
RECORD: comment here + campaign log.
