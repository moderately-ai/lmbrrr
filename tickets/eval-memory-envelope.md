---
id: eval-memory-envelope
title: "EVAL p2: memory envelope — wired budget, pool high-water, pressure failure modes, the 16GB verdict"
status: todo
priority: p2
dependencies: []
related: []
scopes: [evals]
shared_scopes: []
paths: []
tags: [eval-wave, hardware]
---
WHY: GPU-wired allocations cannot swap; exceeding the wired budget (~2/3 of RAM <= 36GB: 16GB Mac -> ~10.7GB, our M3 box 18GB -> 13.6GB) fails HARD (kIOGPUCommandBufferCallbackErrorOutOfMemory), not gracefully. Our stack stopped purging the Metal buffer pool on every wait (fix-metal-buffer-pool ticket) — the pool high-water mark since then is UNMEASURED. Nobody knows our real envelope: weights + drafter + KV at depth + f32 DeltaNet state + pool + vision tower.

PROCEDURE:
1. INSTRUMENT (tiny): sample MTLDevice.currentAllocatedSize + recommendedMaxWorkingSetSize + getrusage ru_maxrss at generation start/end; add to report JSON (coordinates with eval-harness-validity-fixes observability — do them in one fork/repo pass).
2. ENVELOPE SWEEP: record peak allocated vs prompt length {128, 1k, 4k, 16k} x generation {256, 1000}, greedy and spec arms (drafter adds its own state). Deliverable: the envelope table + 'fits in a 16GB Mac with Chrome open' verdict (needs <= ~8GB peak to be comfortable).
3. LEAK CHECK: 20 back-to-back generations in one process; currentAllocatedSize trend must plateau, not stair-step (pool growth without reuse = leak-equivalent for long-lived chat processes).
4. PRESSURE FAILURE MODE: run the 1000-token smoke under `sudo memory_pressure -S -l warn` then -l critical; assert clean completion or clean error — NOT garbage text (Metal buffer aliasing under allocation failure produced silent corruption before: see the #2037 lesson in the dossier). On the 18GB M3 box also try a deliberately oversized context to hit the real wired ceiling and record the failure signature.
DECISION: if peak > ~10GB at realistic settings, memory becomes a campaign constraint (context caps, pool caps, drafter residency policy) and gets its own tickets; if comfortably under, record the envelope and close.
