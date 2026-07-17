---
id: kvbuffer-deferred-linattn-state
title: "KVBuffer: IO-aware deferred linear-attention state update for wide trees"
status: todo
priority: p2
dependencies: []
related: [gdn-rollback-free-masked-solve, fuse-deltanet-decode-step-kernel]
scopes: [runtime/metal, inference/speculative]
shared_scopes: []
paths: []
tags: [route-map, kernel, research]
---
Bucket B / B8 (off-axis enabler). KVBuffer (arXiv 2605.19049, in SGLang for Qwen3-Next): buffer recent draft K/V and update the linear-attention state chunkwise/in-batch instead of recurrently per step; verify draft tokens in parallel from buffered KV without materializing a temporary state per draft token (naive path = +384 MB/request for 4 draft tokens). MEASURED -45% linear-attn decode latency -- but that is the MEMORY part, not the compute wall. Value here: kills the per-draft-token state-materialization blowup when widening trees on M3 unified memory. Enabling memory optimization, not a standalone tok/s lever.
