---
id: drafter-rope-in-place
title: "Drafter rope-in-place: port the target's wave-1 in-place rotary kernel to the drafter backbone"
status: closed
priority: p1
dependencies: [spec-loop-economics-recovery]
related: []
scopes: []
shared_scopes: []
paths: []
tags: [spec-lane, metal, depth-wave]
---
The drafter's Rotary::apply (src/dspark.rs) is the naive 5-dispatch form (narrow views, neg, cat, 2x broadcast_mul, add) applied to q and k in both layers: ~20 mostly-serial dispatches per propose. The target model sheds these via the wave-1 in-place rope kernel (fork 8ebbebd8, bit-preserving); the drafter never got the port. NOTE: the drafter uses FULL-head-dim rotation (training ignores partial_rotary_factor — see the Rotary doc comment), while the target's kernel handles rotary_dim<head_dim; verify the kernel supports rotary_dim == head_dim or add the trivial variant.

GATE TO START: the cpb-ladder calibration on spec-loop-economics-recovery. Depth-x-drain model predicts ~1.5-2.5 ms off the fenced backbone; the per-kernel-us model predicts ~0.3 ms. Only build if the calibrated model prices it above the noise floor.

Ship criteria: bit-identical goldens (rope kernel is bit-preserving on the target), gate battery green, fenced ladder backbone drop consistent with the calibrated prediction, in-loop ms/propose + validation standings on the M3.
