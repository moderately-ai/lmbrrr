---
id: metal-icb-decode-replay
title: "DESIGN: Metal indirect command buffers for the decode step (pre-encoded replay)"
status: todo
priority: p3
dependencies: []
related: []
scopes: [candle-fork]
shared_scopes: []
paths: []
tags: [kernels, frontier]
---
Far-frontier host-encode elimination: pre-encode the whole ~340-dispatch decode step into an MTLIndirectCommandBuffer once; per token, bump a device-side position counter and replay. Kills the ~0.47ms/token host encode STRUCTURALLY (the CUDA-graph analog upstream is building in #3669; Metal's equivalent is ICB compute — supported on Apple Silicon, Tier2 argument buffers). Requirements to design against: per-token-varying scalars (position, KV append offset) must move from set_bytes to device-buffer-indirect reads inside kernels; dispatch sizes are static at m=1; hazard tracking/fences interact with replay. Neither llama.cpp nor MLX does this on Metal — genuinely novel. BUILD ONLY IF the cheap host fixes (greedy-host-path-deferred-readback) leave a measurable host share; this is a large candle dispatch-layer rewrite. Design doc first.
