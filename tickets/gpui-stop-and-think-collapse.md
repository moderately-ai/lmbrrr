---
id: gpui-stop-and-think-collapse
title: "gpui spike 5: stop button + collapsible reasoning sections"
status: todo
priority: p2
dependencies: [gpui-streaming-wire]
related: []
scopes: []
shared_scopes: []
paths: []
tags: [gui]
---
Stop button flips the active generation's AtomicBool (callback errors out at the next token, <=1 decode step latency; up to RUN_AHEAD=32 already-encoded tokens are discarded silently). Reasoning (<think>) renders as a dimmed collapsible section per assistant message, auto-collapsing once the Answer channel opens; disclosure caret toggles. Reuse tui.rs split semantics verbatim (reasoning/answer channel contract).
