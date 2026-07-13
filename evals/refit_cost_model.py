#!/usr/bin/env python3
"""Refit the spec-round cost model from in-loop suite reports.

Usage:
  python3 evals/refit_cost_model.py --reports 'artifacts/suite-<tag>-*.json' \
      --out artifacts/spec-round-cost-model-<name>.json [--base <prior.json>]

Contract (docs/research/1000-toks-campaign.md, in-loop refit): the
scheduler's costs must come from realized per-round walls, not synchronized
kernel tables. Reports carry round_residual_ms samples as
[width, drafted, residual_ms, raw_wall_ms] (raw walls since 2026-07-13).

  verify_ms[l]    <- median raw wall of NO-DRAFT rounds with chunk len l
                     (copy/skip rounds verify l tokens with no draft cost);
                     lens without samples interpolate the base table's shape.
  draft_ms        <- median (drafted-round wall - verify_ms[w+1]) over
                     drafted rounds (the drafter's marginal round cost).
  fixed_ms        <- 0 (the width-dependent truth lives in the table).
  greedy_step_ms  <- verify_ms[1].
"""
import argparse
import glob
import json
import statistics as st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None,
                    help="prior cost model for interpolating unseen lens")
    args = ap.parse_args()

    files = sorted(glob.glob(args.reports))
    if not files:
        raise SystemExit(f"no reports match {args.reports}")

    drafted = []          # (width, wall)
    no_draft = {}         # chunk len -> [walls]
    base = None
    for f in files:
        r = json.load(open(f))
        base = base or r.get("cost_model")
        rr = r.get("round_residual_ms") or {}
        for kind in ("drafter_rounds", "copy_rounds"):
            block = rr.get(kind)
            if not block:
                continue
            for sample in block.get("samples", []):
                if len(sample) < 4:
                    raise SystemExit(
                        f"{f}: sample lacks raw wall (pre-2026-07-13 binary?)")
                w, was_drafted, _residual, wall = sample[:4]
                if was_drafted:
                    drafted.append((w, wall))
                else:
                    no_draft.setdefault(w + 1, []).append(wall)

    if args.base:
        base = json.load(open(args.base))
    if base is None:
        raise SystemExit("no base cost model found in reports; pass --base")
    base_verify = base.get("verify_ms_by_chunk_len") or base["verify_ms"]

    max_len = len(base_verify) - 1
    verify_ms = [0.0] * (max_len + 1)
    anchors = {l: st.median(walls) for l, walls in no_draft.items() if walls}
    if not anchors:
        raise SystemExit("no no-draft rounds in reports; cannot anchor the table")
    # Scale the base table's shape through the measured anchors: for each
    # len, use the measured median when present, else base[l] * (median of
    # measured/base ratios at anchored lens).
    ratios = [anchors[l] / base_verify[l] for l in anchors if base_verify[l] > 0]
    scale = st.median(ratios)
    for l in range(1, max_len + 1):
        verify_ms[l] = anchors.get(l, base_verify[l] * scale)

    if drafted:
        draft_ms = st.median(
            wall - verify_ms[min(w + 1, max_len)] for w, wall in drafted)
    else:
        draft_ms = base.get("default_draft_ms", base.get("draft_ms", 5.0))

    model = {
        "fixed_round_ms": 0.0,
        "default_draft_ms": round(draft_ms, 4),
        "greedy_step_ms": round(verify_ms[1], 4),
        "verify_ms_by_chunk_len": [round(v, 4) for v in verify_ms],
    }
    json.dump(model, open(args.out, "w"), indent=1)
    print(f"{len(files)} reports; {len(drafted)} drafted rounds, "
          f"{sum(len(v) for v in no_draft.values())} no-draft rounds "
          f"(anchored lens: {sorted(anchors)}); table scale {scale:.3f}")
    print(f"draft {model['default_draft_ms']} ms, "
          f"greedy {model['greedy_step_ms']} ms -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
