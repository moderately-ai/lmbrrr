---
id: cargo-deny-license-gate
title: "cargo-deny: license/ban/advisory gate for the workspace (gpui prereq)"
status: done
priority: p2
dependencies: []
related: []
scopes: []
shared_scopes: []
paths: []
tags: [gui, infra, licensing]
---
PREREQ for all gpui work. Add cargo-deny to the workspace: deny.toml with (a) licenses: allow Apache-2.0/MIT/BSD/ISC/Zlib/Unicode classes, DENY GPL/AGPL/LGPL-static anywhere in the resolved graph; (b) bans: duplicate-version warnings; (c) advisories. MUST specifically catch the documented zed transitive chain gpui -> sum_tree -> ztracing -> GPL-3.0 (zed issue #55470) when lmbrrr-gui lands — that chain contaminating a shipped binary is the exact failure this gate exists for. Wire `cargo deny check` into evals/run_gate_battery.sh (fast, offline after first fetch) and document the override/exception process in deny.toml comments. DONE = deny green on the current tree AND a demonstrated red on a deliberately-added GPL test dep (then removed).
