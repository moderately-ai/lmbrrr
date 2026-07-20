---
id: dspark-ondevice-chunk-assembly
title: "Kill the dspark proposal drain: on-device chunk assembly + width selection"
status: done
priority: p2
dependencies: []
related: []
scopes: [runtime/candle, inference/speculative]
shared_scopes: [docs/research]
paths: []
tags: [campaign-1000]
---
## Evidence (sync census, dossier §3 + session agent report)

A drafted round has exactly TWO structural full-pipeline drains, each paying ~1-2ms OS wait latency on top of GPU completion: (1) the proposal readback (dspark.rs:597-600 — confidences+tokens to host, because the verify chunk is assembled host-side via Tensor::from_slice and schedule_prefix_width runs on host) and (2) the verify-targets readback (FUNDAMENTAL — round r+1 depends on r's acceptance; not removable, only shrinkable to a scalar).

## Work

Eliminate drain (1): keep drafted token ids on-device, assemble the verify chunk on-device (cat anchor + draft-id tensor), and move width selection device-side or restructure so the host decision uses the PREVIOUS round's confidences (one-round-lag scheduling — needs an EV check against the scheduler contract). Flag from the census: width is a data-dependent loop bound, so full on-device scheduling means on-device argmax over the admission rule — prototype the one-round-lag variant first, it is much simpler.

Gates: oracle, drafter parity, rotated suite per protocol. Prize: ~1-2ms × ~30 drafted rounds/128 tokens.
