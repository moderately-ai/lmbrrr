---
id: gpui-inference-thread
title: "gpui spike 3: dedicated inference thread + streaming channel"
status: todo
priority: p2
dependencies: [gpui-workspace-scaffold]
related: []
scopes: []
shared_scopes: []
paths: []
tags: [gui]
---
One long-lived std::thread owning the model (load_model_with_optional_quantization once at startup; deployed q4k-full-text + q4k head config) and the candle Device, blocking on a work channel of (prompt_string, Arc<AtomicBool> cancel). Per job: generate_tokens with on_token feeding TokenOutputStream + the TUI's ReasoningTagParser, try_send-ing (TextChannel, String) deltas into an UNBOUNDED async-channel (never block the decode loop), terminating with Done{cancelled}/Err. Cancel = AtomicBool observed in on_token -> return Err (zero engine changes — the callback Result aborts the loop). Testable headless: a small integration test drives one prompt and asserts streamed text equals a run-command reference. NOTE: model must never run on gpui's executors (blocks on Metal waits).
