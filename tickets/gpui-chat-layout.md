---
id: gpui-chat-layout
title: "gpui spike 2: static chat layout (bubbles, scroll, composer)"
status: todo
priority: p2
dependencies: [gpui-workspace-scaffold]
related: []
scopes: []
shared_scopes: []
paths: []
tags: [gui]
---
Root ChatView entity rendering a hard-coded Vec<Message> (role, answer, reasoning, done) as chat bubbles in a scrollable column (plain scroll for now; VirtualList is the polish pass), gpui-component multi-line Input as the composer at the bottom (PressEnter = send stub, Shift+Enter = newline), dark/light theme applied from gpui-component's registry. No inference yet. DONE = looks like a chat app with fake data, resizes cleanly, input editing/IME/clipboard behave.
