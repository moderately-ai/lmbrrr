---
id: imatrix-calibration-pipeline
title: imatrix activation calibration + mixed 3-bit quantization policy
status: todo
priority: p2
dependencies: []
related: []
scopes: [quantization, runtime/candle]
shared_scopes: [docs/research]
paths: []
tags: [quantization, frontier-survey]
---
## Goal
Fork already ships the llama.cpp imatrix loader + from_float_imatrix for k-quants; only per-linear input-activation accumulation hooks are missing (quant-sensitivity harness runs the calibration prefills already). Pipeline: hooks -> imatrix file (~50-100k tokens diverse text) -> quant-convert via from_float_imatrix.

## Acceptance
- Step 1: imatrix-guided q4_K at identical bytes - quality/tau A/B vs current policy (may recover part of the 2.13->1.69 drafter mismatch).
- Step 2: mixed q3_K policy (up/gate/qk/in_proj_qkv at q3; down/v/o/lm_head stay q4k): ~-45 MB/token, expect ~+4-5%; full q3-class ~+10% as a reported experiment.
- Floor is 3-bit-class: q2/IQ2 recorded as collapsed at 1B scale (arXiv 2505.15030) - do not ship.
- Optional stretch: IQ3 plumbing (kernels present in fork, GgmlDType variants missing) ~1-2 weeks for +3-5%.
