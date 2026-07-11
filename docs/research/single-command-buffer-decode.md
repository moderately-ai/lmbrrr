# Single-command-buffer decode — measured verdict

Measured 2026-07-11 on the post-fusion, quantized-target operating point (greedy q4k ~200 tok/s, BF16 ~147). Conclusion up front: **command-buffer and sync discipline is no longer a lever on this model.** The dispatch war was won by kernel fusion (18 DeltaNet layers at ~95 dispatches each → 1 each); what remains per token is GPU kernel time, not launch or commit overhead.

## Command-buffer granularity: falsified

`CANDLE_METAL_COMPUTE_PER_BUFFER` sweep on greedy q4k (128 tokens, math prompt, same session):

| commands/buffer | 16 | 32 | 50 (default) | 100 | 200 | 400 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tok/s | 197.6 | 196.9 | 200.0 | 202.1 | 198.0 | 192.7 | 181.7 |

Flat within noise from 16–200, degrading beyond. One giant command buffer per token (4096) is **9% slower**, not faster: commit granularity is what lets the host encode buffer N+1 while the GPU executes N. Serializing them (big CB) exposes host-encode time that pipelining otherwise hides. BF16 shows the same shape (147.3 default → 137.1 at 4096). The ticket's recorded quick win is falsified with data; the default of 50 stays.

The serialized-vs-pipelined pair also brackets host encode cost: serialized ≈ host + GPU, pipelined ≈ max(host, GPU); from 137.1 vs 147.3 BF16, host encode is well under half the token budget. That bounds the upside of indirect-command-buffer replay (which would eliminate host encode) at a few percent — **ICB feasibility: candle's Metal backend has no ICB support, the surgery is large, and the measured ceiling doesn't justify it now.**

## Redundant per-token synchronize: removed, no measured change

`argmax_token`/`argmax_tokens` did a leading `device.synchronize()` before the readback (timing attribution); `to_scalar`/`to_vec1` already wait. Removed (commit this session): q4k 199.8/194.0 vs 200.0 control, BF16 147.7/147.0 vs 147.3 — within noise. Kept as strictly-not-worse cleanup. The per-wait cost floor measured via the component profiler is ~90 µs (sync-dominated layernorm rows), so the ~2 waits/token cost ~4% at most — consistent with no visible change once pipelining hides one of them.

## Where the token time actually goes (post-fusion)

The CPU-sync profiler can only rank, not attribute (every component reads ≥ the ~90 µs sync floor). Subtracting that floor from the q4k decode profile: fused DeltaNet ≈ 0.15 ms × 18 layers ≈ 2.8 ms dominates, MLP ≈ 0.06 ms × 24, everything else is at or below the floor. The 5.0 ms measured token vs the ~1.2 ms weight-read roofline is GPU kernel time — per-kernel efficiency and breadth of fusion (norm+projection boundaries, `reduce-metal-dispatch-layer-overheads`), not command-buffer mechanics. Truthful per-kernel attribution needs an Instruments GPU capture (the standing note from the matmul ticket applies here too).

## Acceptance disposition

- Audit + before/after counts: done (≈42 CB commits/token at default; 1–2 at 4096 — and the 1–2 case is slower; count is not the cost).
- Per-token host work: decode path already passes a None mask and does device argmax with a single scalar readback; the leading sync removed. The remaining 4-byte per-step token upload and device-resident EOS check live in `two-stage-argmax-device-sampling` (open, p3).
- ICB replay feasibility: investigated, bounded, declined (above).
- The ≥150 tok/s BF16 stage gate is long met (147→ shared with fusion tickets; quantized 200). The 250 quantized forwards/s aspiration transfers to kernel-efficiency work (`reduce-metal-dispatch-layer-overheads`, Instruments-first).
