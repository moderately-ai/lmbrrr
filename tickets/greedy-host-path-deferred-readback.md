---
id: greedy-host-path-deferred-readback
title: "Greedy host path: deferred readback, READBACK_EVERY tuning, id ring buffer"
status: closed
priority: p1
dependencies: []
related: []
scopes: [runtime/candle]
shared_scopes: []
paths: []
tags: [campaign-1000]
---
The 32k-head experiment PROVED the bench loop is host/sampling-bound (head = 41% of device bytes but only ~14% of bench wall; removing 7/8 of head bytes bought +13.5% not +35%). Attack the measured host share, all exact: (1) DEFERRED READBACK: generate.rs's every-8-token cat+to_device(Cpu) is flush_and_wait_current = waits for the ENTIRE queue -> GPU drains -> host re-encodes from empty; every 8 tokens the GPU eats a full drain + refill bubble. Fix: double-buffer — hold batch A's cat tensor, enqueue batch B's forwards, THEN read A (its work long complete, wait ~0, GPU never drains). EOS discovered one batch late = bounded wasted forwards. (2) READBACK_EVERY 8 -> 16/32, or adaptive (grow with generation length; overrun cost is only paid at EOS). (3) Ring buffer: argmax writes slot i of a [K] device buffer (tiny copy kernel or offset write) instead of K rank-1 tensors + K-way cat per flush. GATES: exact (greedy text byte-identical — ordering unchanged, only sync timing), bench A/B rotated; measure with the timestamp meter (metal-gpu-timestamp-meter) if built. Prize estimate: the every-8 drain is ~0.2-1.0ms + encode-refill per 8 tokens = ~5-15% of wall at the current floor.
