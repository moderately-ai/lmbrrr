---
id: integrate-dspark-block-runner
title: Integrate DSpark block runner
status: todo
priority: p1
dependencies: [train-dspark-semi-autoregressive-drafter, implement-speculative-state-rollback]
related: []
scopes: [inference/speculative, runtime/candle, evals]
shared_scopes: [docs/research]
paths: [src/main.rs, src/minicpm.rs, src/qwen35.rs, evals/dspark/**, docs/research/dspark-block-runner.md]
tags: [speculative, dspark, performance]
---
## Goal

Load the trained DSpark drafter in Rust and run a real speculative cycle: one target anchor, DSpark block draft, scheduled target verification, exact greedy reconstruction.

## Acceptance

- Load the DSpark backbone, Markov head, and confidence head from safetensors.
- Propose a draft block before target verification without computing target hidden states for future positions.
- Verify the scheduled prefix in one target chunk and reconstruct exact greedy output.
- Run multi-round speculative generation: after each verification, accept prefix + bonus token and continue drafting from the new anchor without re-running prompt prefill.
- Add DeltaNet conv/recurrent state and full-attention KV cache snapshot/restore (or restore-then-re-advance over the accepted prefix) so a rejected suffix cannot leave stale layer state; the current runner mutates conv_state/recurrent_state/KvCache in place with no rollback, which limits it to a single cycle.
- Gate state-rollback correctness with an exact greedy oracle over at least 128 generated tokens on multiple prompts.
- Run the drafter as Candle tensors on the target device, sharing the frozen target embedding and LM head; no per-token CPU hidden-state export or scalar Rust matmul loops in the accelerated path.
- Reproduce DeepSpec's draft numerics exactly: bidirectional SDPA with fused context prepended to K/V in every layer, per-head QK-RMSNorm, RoPE position anchor_pos + k per slot (slot 0 = anchor embedding), target-GQA dims (q 1024->2048, kv 1024->512, head_dim 256), and post-Markov logits as the draft distribution p_d.
- Acceptance implements min(1, p_t/p_d) rejection sampling with residual bonus sampling; at temperature 0 this reduces to exact greedy match, which is the v1 oracle gate.
- Report draft latency, verify latency, accepted length, verifier waste, confidence scores, and target calls saved.
- Compare directly against the recurrent EAGLE runner on the same prompts and draft widths.
