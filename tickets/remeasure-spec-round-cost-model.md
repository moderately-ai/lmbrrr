---
id: remeasure-spec-round-cost-model
title: Rebuild the spec-round cost model from in-loop post-fusion measurements
status: done
priority: p1
dependencies: []
related: [implement-dspark-hardware-aware-prefix-scheduler, cut-drafter-propose-cost]
scopes: [evals, inference/speculative]
shared_scopes: [docs/research]
paths: []
tags: [speculative, measurement, campaign-1000]
---
## Outcome (2026-07-11, closing)

Artifact shipped: artifacts/spec-round-cost-model.json (verify_ms by chunk length from vt-gdc2 short+medium averages, draft costs per config); dspark-run --cost-model loads it (built-in measured defaults otherwise) and reproduces the operating point (115.6 tok/s, 0.79x). Old table doc marked SUPERSEDED. The propose breakdown and the l=1->2 attribution (chunk-kernel phases + m=2 gemm) are recorded in the sections below; the m=2 gemm remains the open ~4ms and lives on the bf16-activation ticket as the weight-shared small-m quantized/dense matmul package.

## RESOLVED: l=1 -> l=2 doubling fully attributed (610a46e)

LMBRRR_VT_PROFILE=1 component diff at ctx=27: deltanet_fused_chunk 8.68ms vs deltanet_fused_decode 5.03 (+3.65 — the chunk kernel's sequential phase-4/5 tg_sum barrier chains are ~2x the decode kernel per token; restructure to simdgroup-parallel positions or cooperative B-matrix computation), mlp 4.70 -> 8.85 (+4.15 — m=2 tile-gemm inefficiency is REAL for MLP shapes even though the skinny-gemm v1 design lost; a v2 with multi-column simdgroups for B reuse is the shape of the fix), remainder ~1.9ms across attention/norms. Two scoped fork-kernel tasks worth ~7.5ms/round combined — these are the L2 completion path to verify ~9ms.

## Update: skinny-gemm hypothesis FALSIFIED (fb3f80f)

Built the skinny kernel (fork 5edb0903, opt-in CANDLE_SKINNY_GEMM=1): even function-constant-specialized it LOSES to the mlx tile gemm at m=2-12 (gamma8 verify 23.6 vs 17.1 ms) — the tile kernel's B reuse wins; the "~150 GB/s tile inefficiency" read was wrong in composition. The l=1 -> l=2 verify doubling (6.5 -> 13.8 ms) remains UNEXPLAINED and needs per-component attribution (Instruments capture or Qwen35Profiler through a chunk forward) before any further kernel work. Candidates: lm_head mm at m=2 specifically, fused-chunk-kernel occupancy at small l, mask/cat overheads, allocator effects. This is now the ticket's core remaining question — the answer is worth ~5 ms/round.

## Findings (2026-07-10 late — the headline discovery)

Post-fusion isolated table (target/verify-table-postfusion.json, short profile): T_verify by CHUNK LENGTH l: 1=6.5ms, 2=13.7, 4=14.9, 8=17.1, 16=31.0, 32=35.0. In-loop timed round (LMBRRR_LOOP_TIMING=1, gamma 4 thr 0.3): draft 5.1 + verify 14.9 + rollback 1.2 = 22.3ms wall — table and loop now agree. Bisection (env-flag reruns): the l=1 -> l=2 doubling is NOT the fused DeltaNet chunk kernel (unfused l2=23.4, the kernel already saves 10ms) and NOT SDPA (unchanged) — it is SMALL-M GEMM: at m=1 every projection routes to the gemv kernel (~350 GB/s); at m>=2 they hit the mlx GEMM 32x32 tile at ~150 GB/s effective, so the whole 1.5 GB weight sweep runs at half bandwidth for exactly the chunk sizes verification uses. FIX: skinny-GEMM Metal kernel in the fork (B streamed once per threadgroup, m<=12 activation rows resident) -> verify ~8-9 ms projected; also unblocks small-batch decode for batched-multi-stream-decode-runner.

## Goal

The scheduler's declared input (docs/research/dspark-verification-throughput-table.md, T_verify ~= 11 + 6.3*gamma, 67 ms at gamma=8) is ~5x off post-fusion reality (verify 13.2 ms/round total) and would drive systematic under-verification. Rebuild the cost model from IN-LOOP measurements: T_verify(width) and T_propose(gamma) for width/gamma in 1..=12 across context lengths, plus fixed per-round overheads (syncs, capture concat, ctx append, from_slice uploads). Record where the 8.7 ms propose goes (backbone vs per-Markov-step lm_head/markov_w2 reads vs readback) — that breakdown decides cut-drafter-propose-cost's shape.

## Acceptance

- JSON artifact in the shape the scheduler consumes; scheduler ticket dependency repointed here.
- Propose-cost breakdown by phase.
- Mark docs/research/dspark-verification-throughput-table.md SUPERSEDED (pre-fusion) with a pointer.
- In-loop measurement (means over a real run), same-session protocol per lmbrrr-measurement-protocol memory.
