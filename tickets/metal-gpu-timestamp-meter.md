---
id: metal-gpu-timestamp-meter
title: In-binary GPU timestamp meter (per-command-buffer busy/idle, no Xcode)
status: todo
priority: p2
dependencies: []
related: []
scopes: [candle-fork]
shared_scopes: []
paths: []
tags: [kernels, tooling]
---
Boundary-hole and per-buffer GPU busy/idle currently need a 6GB Xcode .gputrace + manual GUI analysis. Add an in-binary meter so a bench task prints 'buffer k busy X ms, boundary gap Y us' directly. RECIPE (counter-research agent, verified against objc2-metal 0.3.2, already pinned): FALLBACK-FIRST (ship this, ~15 lines): MTLCommandBuffer.GPUStartTime()/GPUEndTime() (CFTimeInterval sec, MTLCommandBuffer.rs:409/415, valid on Apple Silicon). Per-buffer busy = GPUEndTime-GPUStartTime; boundary hole = next.GPUStartTime - prev.GPUEndTime. commands.rs holds both buffers at swap. Env-gated (CANDLE_METAL_TIMING) or bench-only. PHASE 2 (per-encoder): MTLCounterSampleBuffer + MTLCommonCounterSetTimestamp at AtStageBoundary (only point M-series supports), via MTLComputePassDescriptor.sampleBufferAttachments start/endOfEncoderSampleIndex, resolve after waitUntilCompleted; ticks already ns on Apple Silicon. wgpu Metal HAL = line-for-line prior art. Use to SIZE the fusion slice (drain-gap time around the 181 elementwise dispatches). Measurement-only, no correctness gates.
