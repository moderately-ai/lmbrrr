---
id: metal-wave3-commit-pacing
title: "Wave 3: command-buffer commit pacing (eager enqueue/flush vs the CPB cap)"
status: todo
priority: p2
dependencies: [metal-wave2-host-encode-rlist]
related: []
scopes: [candle-fork]
shared_scopes: [docs/research]
paths: []
tags: [kernels, campaign-1000]
---
## Evidence

One-step gputrace shows a ~0.6-0.7ms GPU-idle hole at a command-buffer boundary (~11 buffers/step at the default CANDLE_METAL_COMPUTE_PER_BUFFER=50). Mechanism: commits are lazy (commit_swap_locked, commands.rs:294-315, no host wait) and the new encoder waits on ALL outstanding fences (commands.rs:169-187). CPB=1000 measured WORSE (287.4±9.8 vs 305.1±1.6 steady, load-suspect but mechanistically explained): a giant buffer starves the GPU until the every-8-token readback forces submission — the 50 cap is an accidental commit pacer.

## Work (inherently iterative — measure between variants, quiet machine only)

Variants to A/B (procedure dossier §7): (a) CPB sweep incl. SMALLER caps (25, 12); (b) explicit flush() per token from the greedy loop; (c) early enqueue() (exists unused at command_buffer.rs:55-57) so buffers claim queue slots at creation. Success metric: bench steady tok/s + the boundary hole gone in a re-capture (one-step gputrace).

Related upstream context: #2037 (encoder reuse, the 2024 attempt), #3511/#3532 (current machinery, already in our base). docs/research/metal-decode-utilization.md §3-§4.
