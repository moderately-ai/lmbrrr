---
id: metal-fence-scope-tidiness
title: "Per-buffer fence waits at encoder start (completes #3532's stated minimum-fence design)"
status: todo
priority: p3
dependencies: [metal-wave2-host-encode-rlist]
related: []
scopes: [candle-fork]
shared_scopes: [docs/research]
paths: []
tags: [kernels]
---
Upstream #3532 states 'wait for the minimum amount of required fences', but the implementation (commands.rs:169-187, byte-identical upstream) waits on ALL outstanding fences at every new encoder start. The prev_ce_outputs map already keys fence-by-buffer, so waits can defer to first-touch-per-buffer (waitForFence must be encoded before the dispatch that uses the resource — bind-time is legal). Modest local value for a single-queue chain; PR-worthy upstream. SYNC-MACHINERY RISK CLASS: isolate from numerics changes, gate with oracle + tree-check + barrier-probe (dossier §7). Do after waves 1-2.
