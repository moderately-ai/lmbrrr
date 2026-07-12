---
id: metal-wave2-host-encode-rlist
title: "Wave 2: host encode-path reductions (static kernel names, pipeline cache, encoder locks)"
status: todo
priority: p1
dependencies: [metal-wave1-gpu-stream-slimming]
related: []
scopes: [candle-fork]
shared_scopes: [docs/research]
paths: []
tags: [kernels, campaign-1000]
---
## Scope

Host encode costs ~2.8μs/dispatch × ~500 dispatches ≈ 1.4ms/token CPU (bench decode_model wall). Anatomy + R-list with file:line sites: docs/research/metal-decode-utilization.md §5b (full agent report in the session dossier). Est. combined −1.0-1.6μs/op ≈ −0.5-0.8ms/step.

Order (bulk the first two, then one at a time):
- **R1**: static kernel names — replace format!/ToString kernel-name paths (binary mod.rs:2004-2034, gemv mlx_gemm.rs:396) with &'static str matches; call_* signatures take the static name. 2-3 heap Strings/dispatch removed.
- **R2**: pipeline cache read-lock fast path (kernel.rs:158 takes a WRITE lock on every hit) — double-checked read-then-write.
- **R3**: call-site pipeline handle caching (OnceLock per op×dtype) — skips the map+hash+retain entirely.
- **R5 (LAST, ALONE)**: collapse the 4 per-dispatch EncoderState mutex hits (encoder.rs:122/138/105) into the dispatch under the outer Commands guard. Touches hazard bookkeeping — silent-race risk class: oracle + tree-check + barrier-probe re-run mandatory; never bundle with numerics changes.

## Gates
No dispatch-stream change ⇒ the gate is LITERAL text equality on smoke runs + nextest + oracle. Perf measured as one package (bench decode_model wall + steady tok/s, quiet machine, dossier §7 protocol). Upstream PR candidates once proven.
