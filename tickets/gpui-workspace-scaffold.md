---
id: gpui-workspace-scaffold
title: "gpui spike 1: workspace conversion + lmbrrr-gui hello window"
status: todo
priority: p2
dependencies: [cargo-deny-license-gate]
related: []
scopes: []
shared_scopes: []
paths: []
tags: [gui]
---
Convert the repo to a cargo workspace (root lmbrrr package + new member lmbrrr-gui). lmbrrr-gui: bin crate depending on lmbrrr as a path lib + gpui and gpui-component pinned by git rev (longbridge/gpui-component 0.5.x; it pins gpui to zed-industries/zed — record both revs in Cargo.toml comments per rust.md pinning rules) + async-channel. Deliverable: a themed empty window opens; cargo-deny green at the new graph (this is where the GPL chain check bites); the core lmbrrr crate's build/test time unchanged (gpui stays out of its dep graph); gate battery still green. Keep dev-profile settings per rules/rust.md.
