#!/usr/bin/env python3
"""Fit per-position Platt STS calibration from suite confidence records.

Usage:
  python3 evals/fit_sts.py --records 'artifacts/suite-<tag>-<arm>-*.json' \
      --out <drafter-dir>/sts.json

Records come from fixed-gamma (unscheduled) dspark-run reports on the
Spec-Bench CALIBRATION split (see evals/run_spec_suite.py --split
calibration); their confidence_records are exact-argmax accept/reject
labels on the deployed target, which is the event the scheduler's
survival probabilities must predict. Validation prompts stay held out.

Flow (established rounds 2-3): collect records -> fit here -> scheduled
validation on the held-out split must beat the incumbent stack before a
drafter becomes primary.
"""
import argparse
import glob
import json
import math
import sys


def platt(xs, ys, iters=300, lr=0.5):
    scale, shift = 1.0, 0.0
    n = len(xs)
    for _ in range(iters):
        gs = gb = 0.0
        for x, y in zip(xs, ys):
            p = 1.0 / (1.0 + math.exp(-(scale * x + shift)))
            gs += (p - y) * x
            gb += p - y
        scale -= lr * gs / n
        shift -= lr * gb / n
    return scale, shift


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True,
                    help="glob of dspark-run report JSONs with confidence_records")
    ap.add_argument("--out", required=True, help="sts.json output path")
    ap.add_argument("--min-samples", type=int, default=20,
                    help="minimum samples for a per-position fit")
    args = ap.parse_args()

    samples = []
    files = sorted(glob.glob(args.records))
    if not files:
        print(f"no files match {args.records}", file=sys.stderr)
        return 1
    for f in files:
        d = json.load(open(f))
        for round_records in d["confidence_records"]:
            for (pos, logit, _prob, accepted) in round_records:
                samples.append((pos, logit, 1.0 if accepted else 0.0))
    print(f"{len(files)} reports, {len(samples)} samples")

    g_scale, g_shift = platt([s[1] for s in samples], [s[2] for s in samples])
    out = {"scale": g_scale, "shift": g_shift, "positions": []}
    for pos in range(max(s[0] for s in samples) + 1):
        xs = [s[1] for s in samples if s[0] == pos]
        ys = [s[2] for s in samples if s[0] == pos]
        if len(xs) < args.min_samples:
            break
        sc, sh = platt(xs, ys)
        out["positions"].append({
            "position": pos, "scale": sc, "shift": sh,
            "n": len(xs), "accept_rate": round(sum(ys) / len(ys), 3),
        })
        print(f"pos {pos}: n={len(xs)} accept={sum(ys)/len(ys):.3f} "
              f"scale={sc:.3f} shift={sh:.3f}")
    json.dump(out, open(args.out, "w"))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
