---
id: gpui-metal-coexistence-verdict
title: "gpui spike 6: Metal coexistence measurement (same-process go/no-go)"
status: todo
priority: p2
dependencies: [gpui-stop-and-think-collapse]
related: []
scopes: []
shared_scopes: []
paths: []
tags: [gui, eval]
---
gpui renders via Metal while inference saturates the same GPU — no known precedent for candle-Metal + gpui in one process. Measure: sustained generation (500+ tokens) while interacting with the UI; record frame pacing / input latency (Instruments 'Metal System Trace' or gpui's frame stats if exposed) and decode tok/s vs the headless number on the same machine. VERDICT: acceptable jank -> same-process stands; unacceptable -> the IPC escape hatch (subprocess lmbrrr engine over stdio/socket behind the existing engine trait) becomes its own ticket. Record numbers here either way; this is the go/no-go input for the polish pass.
