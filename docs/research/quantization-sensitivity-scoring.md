# Quantization Sensitivity Scoring

Date: 2026-07-07

Ticket: `score-minicpm-quantization-sensitivity`

## Command

Run the first MiniCPM-V-4.6 quantization sensitivity pass with:

```sh
cargo run --release --features metal -- quant-sensitivity \
  --calibration evals/calibration/minicpm_v46_quant_calibration.jsonl \
  --output target/minicpm-v46-quant-sensitivity.json
```

For fast smoke checks:

```sh
cargo run --release --features metal -- quant-sensitivity \
  --max-cases 1 \
  --max-modules 4 \
  --output target/minicpm-v46-quant-sensitivity-smoke.json
```

By default the command scores `q4_symmetric`, `q5_symmetric`, and
`q8_symmetric` simulations. It skips protected tensors such as embeddings,
norms, DeltaNet state tensors, the vision tower, and the multimodal merger. Use
`--include-protected` only when deliberately auditing those protected families.

## What It Measures Now

The report has two measured sections:

- `baseline`: BF16/F32 model prefill on text rows from the calibration JSONL,
  including prompt token counts, synchronized forward latency, top-1 token, and
  top-k logits.
- `weights`: per-candidate tensor symmetric quantization simulation, including
  relative MSE, absolute error, per-output-channel worst MSE, estimated packed
  bytes, and a conservative policy recommendation.

Module families are separated into text MLP, full attention, DeltaNet,
DeltaNet conv/state, embeddings, norms, vision, merger, LM head, and other
families. Protected tensors are still reported in `skipped_modules` with their
reason so later policy conversion does not accidentally quantize them.

## Current Limits

This pass does not yet collect true per-module activation reconstruction error
or per-module logit drift. The runner needs activation hooks or a perturbed
quantized-forward path before those numbers are meaningful. The report marks
those fields as `not_collected` for each module instead of filling them with
proxy values.

Latency deltas are also limited to quantization-simulation time and estimated
weight bytes. Real runtime latency requires the follow-up quantized loader and
Metal matmul benchmark tickets.
