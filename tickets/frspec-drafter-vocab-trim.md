---
id: frspec-drafter-vocab-trim
title: FR-Spec drafter vocabulary trimming (top-32k head for drafting)
status: done
priority: p2
dependencies: []
related: []
scopes: [inference/speculative]
shared_scopes: []
paths: []
tags: [speculative, frontier-survey]
---
## Goal
The DSpark drafter's 248k-vocab head is ~half of draft cost. Restrict DRAFTING to top-32k frequency-ranked tokens (repacked contiguous q8 rows + id remap); verification keeps the full head so output is provably identical. llama.cpp #25187 measured -85% draft-head time, byte-identical at temp 0.

## Acceptance
- Frequency ranking from corpus stats; repacked sub-head artifact; drafter argmax maps through remap table.
- Bit-identical spec output vs untrimmed drafter on the fixed suite; draft_ms and tau reported before/after.
- Composes with drafter quantization and any future MTP head.
