---
id: gpui-streaming-wire
title: "gpui spike 4: send flow + per-frame coalesced streaming + multi-turn prompt"
status: todo
priority: p2
dependencies: [gpui-chat-layout, gpui-inference-thread]
related: []
scopes: []
shared_scopes: []
paths: []
tags: [gui]
---
Wire PressEnter: append User message + empty Assistant message, assemble the FULL multi-turn prompt (add chat_prompt_multi(history) to lmbrrr src/prompt.rs — the one engine-side addition; single-turn chat_prompt stays), dispatch to the inference thread. A cx.spawn foreground task drains ALL available deltas per wake (while try_recv), appends to the active message's answer/reasoning buffers, then ONE cx.notify — never per-token (191 tok/s must coalesce to frame cadence). Composer disabled while generating; Done/Err states rendered. NOTE the interaction with eval-multiturn-state-reuse: every turn re-prefills the whole history (~650 tok/s prefill) — long chats will feel it; that ticket's priority rises once this ships.
