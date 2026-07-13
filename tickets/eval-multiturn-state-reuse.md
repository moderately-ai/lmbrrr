---
id: eval-multiturn-state-reuse
title: "EVAL p2: multi-turn chat — per-turn TTFT, hybrid state reuse across requests, the re-prefill tax"
status: todo
priority: p2
dependencies: []
related: []
scopes: [evals, runtime/candle]
shared_scopes: []
paths: []
tags: [eval-wave]
---
WHY: all our benches are one-shot. A chat user pays prefill of the ENTIRE growing conversation every turn unless KV + DeltaNet recurrent state + conv state persist across requests — and for hybrid models this is the ecosystem's biggest pain point: llama.cpp had a whole bug family (#19690, #19858, #20225, #22384) around hybrid state checkpoints, because GDN state cannot be trimmed/rolled back to an arbitrary prefix — a prefix edit invalidates it irrecoverably. Correct multi-turn state handling is where hybrids diverge most from dense transformers, and (since upstream keeps getting it wrong) a competitive edge.

PHASE 1 — MEASURE THE TAX (no code changes): script a 6-8 turn conversation replay (fixed turns, ~100-200 token replies) against the current binary: per-turn TTFT and total wall. Today each turn = full re-prefill; the curve should grow ~linearly with conversation length. Record prefill tok/s at each accumulated length. This number IS the eval: 'turn 8 costs X ms before the first token'.
2. AUDIT what state reuse would need in our stack: the runner is one-shot today (read src/main.rs run flow); persisting state across turns within one process = keep the KV cache + per-layer GDN state + conv taps and append-only continue (legal for pure appends — chat is append-only if the template only appends). Write the design note here; DO NOT implement without a product signal that chat matters (ask user).
3. DECISION INPUT for the user: table of per-turn TTFT now vs projected with reuse (prefill of just the new turn). If turn-5+ TTFT exceeds ~1.5s, state reuse is likely the single biggest UX lever we have — bigger than any kernel work — and deserves its own implementation ticket.
CAVEATS: cross-turn reuse interacts with speculation (drafter state must checkpoint identically — see mlx-lm PR #1456's replay-based rollback for the reference design) and with the thinking-flag chat template (OpenBMB: Instruct checkpoint template enables thinking by default; --reasoning off equivalent needed — check our template handling for the same trap and record what we do).
