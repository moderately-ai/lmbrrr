---
id: gpui-polish-pass
title: "gpui phase B: polish (markdown, virtualization, autoscroll, UX)"
status: todo
priority: p3
dependencies: [gpui-metal-coexistence-verdict]
related: []
scopes: []
shared_scopes: []
paths: []
tags: [gui]
---
The clean/modern pass, only after the spike measures well: gpui-component Markdown element for assistant replies (code blocks, lists), VirtualList for long transcripts (variable-height rows), auto-scroll-to-bottom with scroll-up-pauses-follow, copy-message, model-load spinner, streaming caret, error toasts, theme picker, app icon/bundle. Also revisit: distribution licensing sign-off (cargo-deny green is necessary but external distribution needs the zed #55470 status confirmed at our pinned rev), binary size with strip/codegen-units per rules/rust.md.
