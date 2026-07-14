---
id: eval-host-loop-budget-census
title: "EVAL: host-loop budget census — where the non-GPU milliseconds go in greedy decode"
status: done
priority: p1
dependencies: []
related: []
scopes: [runtime/candle]
shared_scopes: []
paths: []
tags: [eval-wave, host-path]
---
WHY: the 32k-head experiment proved the bench loop is host/sampling-bound at the margin (a 13.5% device-side saving under-realized in tok/s). We have a GPU-side census (trace) but NO host-side one. Two p1/p2 tickets (greedy-host-path-deferred-readback, fused-gemv-argmax-head) are sized by guesses about this budget. Run this census FIRST; it converts those guesses into measurements. Greedy step is ~3.6 ms; GPU busy is ~2.9-3.0 ms (trace); the ~0.6-0.7 ms residual is unattributed.

IMPLEMENT (lmbrrr, src/generate.rs greedy device-chain loop): behind env `LMBRRR_HOST_TIMERS=1` (OnceLock bool), accumulate std::time::Instant deltas per step into named buckets: (a) encode+submit (model forward call up to the point work is queued), (b) readback wait (the synchronize/to_vec on the committed id), (c) detokenize + stream write, (d) EOS/bookkeeping/other (loop total minus a-c). At end of run/bench, print a table: total ms, ms/token, % of step for each bucket. Zero overhead when env unset.

GATES: env-unset run must be byte-identical text + bench tok/s within noise vs pre-change binary (use evals/run_gate_battery.sh once it exists); nextest 59 -> rebuild (nextest clobbers the metal binary).

MEASURE: `LMBRRR_HOST_TIMERS=1 ./target/release/lmbrrr bench --quantized-manifest artifacts/minicpm-v46-q4k-full-text/manifest.json --quantize-lm-head q4k --warmup 1 --iterations 3 --profile medium` (also --profile long). Timer overhead check: env-on vs env-off tok/s must agree within 1% — if not, buckets are too fine; coarsen. OPTIONAL deeper view: `xctrace record --template 'Time Profiler' --launch -- ./target/release/lmbrrr run ...` and read the top host frames (mach_msg waits = readback; tokenizer frames = detokenize).

DELIVERABLE + ROUTING: the ms/token table as a comment here + campaign log. Routing rules: readback-wait bucket >= 0.3 ms/token -> greedy-host-path-deferred-readback proceeds as specced (its expected value is now measured, not guessed); detokenize bucket >= 0.2 ms/token -> file a detokenize-batching item on that same ticket; encode+submit >= 0.5 ms/token -> the dispatch-layer lane (metal-elementwise-fusion, drain probe) gets the priority bump, and cross-check the number against the wave-2 host-encode measurements in the dossier.
