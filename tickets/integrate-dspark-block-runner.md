---
id: integrate-dspark-block-runner
title: Integrate DSpark block runner
status: done
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
- Multi-round loop, rollback, and the corruption-invariance oracle are DONE in `implement-speculative-state-rollback` (dspark-run stub mode, docs/research/speculative-state-rollback.md); this ticket replaces the stub with the trained drafter inside that verified loop.
- Load the DeepSpec checkpoint (standard HF safetensors from save_pretrained + config.json) into Candle: backbone layers, fc context fusion + hidden_norm, mask embedding row, Markov W1/W2, confidence head; embedding/lm_head shared with the already-loaded target tensors.
- Mirror the checkpoint config exactly: rope_parameters (theta 1e7, partial_rotary_factor 0.25 -> rotary_dim 64), per-head QK-RMSNorm, q 1024->2048 / kv 1024->512 dims, bidirectional block attention with the fused context prepended to K/V in every layer, draft RoPE positions anchor_pos + k.
- Capture-layer hidden states for the fused context stay on-device (no per-token CPU export or scalar Rust matmul loops in the accelerated path).
- Drafter proposes [d1..dgamma] from [anchor emb, mask emb x (gamma-1)] with left-to-right Markov sampling (post-Markov logits are the draft distribution); confidence logits reported per position.
- Reproduce DeepSpec's draft numerics exactly: bidirectional SDPA with fused context prepended to K/V in every layer, per-head QK-RMSNorm, RoPE position anchor_pos + k per slot (slot 0 = anchor embedding), target-GQA dims (q 1024->2048, kv 1024->512, head_dim 256), and post-Markov logits as the draft distribution p_d.
- Acceptance implements min(1, p_t/p_d) rejection sampling with residual bonus sampling; at temperature 0 this reduces to exact greedy match, which is the v1 oracle gate.
- Report draft latency, verify latency, accepted length, verifier waste, confidence scores, and target calls saved.
- Compare directly against the recurrent EAGLE runner on the same prompts and draft widths.
