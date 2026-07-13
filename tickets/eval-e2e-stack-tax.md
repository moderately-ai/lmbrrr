---
id: eval-e2e-stack-tax
title: "EVAL: end-to-end stack tax — same-machine effective-bandwidth vs llama.cpp and MLX full decode"
status: todo
priority: p1
dependencies: []
related: []
scopes: [evals]
shared_scopes: []
paths: []
tags: [eval-wave, kernels]
---
WHY: eval-apples-to-apples-qmv compares ONE kernel at fixed shapes. This ticket compares WHOLE STACKS: model-weight-bytes x decode-tok/s = effective GB/s, each stack running its own 4-bit model end to end on THIS machine. It bounds the total remaining prize (kernels + dispatch layer + host loop + model structure together) and is exactly how the literature's 250-270 GB/s numbers were derived — but with the cross-machine variable removed. Run it in the same quiet session as the qmv ticket if possible; together they decompose gap = kernel part + everything-else part.

ARMS (record every version: llama.cpp commit, mlx/mlx_lm pip versions, macOS build, Xcode):
1. OURS: `./target/release/lmbrrr bench --quantized-manifest artifacts/minicpm-v46-q4k-full-text/manifest.json --quantize-lm-head q4k --warmup 1 --iterations 3 --profile medium` -> median steady_state_tokens_per_second. Weight bytes: sum the tensor byte sizes in the manifest (python one-liner over manifest.json; record the number — approx 2.4 GB expected, verify). Effective GB/s = bytes x tok/s / 1e9.
2. LLAMA.CPP: first check whether the exact model converts (`python convert_hf_to_gguf.py --help`; look for minicpm-v / qwen3.5-hybrid arch support in their repo). If yes: convert text tower to Q4_K_M and run `./build/bin/llama-bench -m <gguf> -p 0 -n 128 -r 5` (tg tok/s). If NOT (likely — hybrid DeltaNet): use a dense proxy, e.g. bartowski Qwen2.5-3B-Instruct-Q4_K_M.gguf, same llama-bench command, and label the row PROXY.
3. MLX: `pip install mlx mlx-lm`; `python -m mlx_lm generate --model mlx-community/Qwen2.5-3B-Instruct-4bit --prompt "Explain how tides work in detail." --max-tokens 500` -> it prints generation tokens-per-sec. Model bytes: du of the snapshot's safetensors.

HONESTY CAVEATS the executor must include in the writeup: (a) proxy models are DENSE — no DeltaNet state work, no 6-layer full-attn hybrid; our effective GB/s is structurally depressed by the ~25% of GPU time that is not weight-bound GEMV (per the trace census in docs/research/metal-decode-utilization.md). Report BOTH raw effective GB/s and GEMV-adjusted (ours / 0.75) alongside. (b) tokenizer/prompt differences are irrelevant at steady-state decode; use tg/generation numbers only, never prompt-processing numbers. (c) all arms in one session per eval-protocol-ambient-control, interleaved (ours, theirs, ours again).

DECISION: their effective GB/s >= 1.3x our GEMV-adjusted number AFTER the qmv ticket showed kernel parity -> the gap is dispatch/host/loop — attribute with metal-gpu-timestamp-meter + a fresh gputrace dispatch census, and promote the winning host-side tickets (drain probe, deferred readback). Rough parity -> our stack is competitive; record the bound ("<=X% total remaining vs best-of-breed on this machine") in docs/research/1000-toks-campaign.md as the campaign's honest ceiling statement.
