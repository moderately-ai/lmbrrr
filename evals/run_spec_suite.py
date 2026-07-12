#!/usr/bin/env python3
"""Run lmbrrr dspark-run over the Spec-Bench suite split.

Usage:
  python3 evals/run_spec_suite.py --arm name=DRAFTER_DIR[:extra flags] ... \
      [--split validation|calibration] [--reps 3] [--max-new-tokens 128] \
      [--classes math_reasoning,summarization,...] [--per-class 1] [--tag out-tag]

Arms run interleaved per rep (measurement protocol: rotate arms). Reports
land in target/suite-<tag>-<arm>-<qid>-<rep>.json and a summary table prints
at the end. Extra per-arm flags (after ':') are split on whitespace, e.g.
  --arm b=target/dspark-drafter-round2-fresh:--schedule
"""
import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "evals/prompts/spec-suite.json"
QUESTIONS = ROOT / "evals/prompts/spec_bench_question.jsonl"
QMAN = "target/minicpm-v46-q4k-full-text/manifest.json"
COST = "target/spec-round-cost-model-r4q4f.json"


def load_questions():
    return {
        d["question_id"]: d
        for d in (json.loads(l) for l in QUESTIONS.open())
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="name=drafter_dir[:extra cli flags]")
    ap.add_argument("--split", default="validation",
                    choices=["validation", "calibration"])
    ap.add_argument("--classes", default=None)
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    suite = json.loads(SUITE.read_text())
    questions = load_questions()
    classes = (args.classes.split(",") if args.classes
               else list(suite[args.split].keys()))
    qids = [(cls, qid) for cls in classes
            for qid in suite[args.split][cls][: args.per_class]]

    arms = []
    for spec in args.arm:
        name, _, rest = spec.partition("=")
        drafter, _, extra = rest.partition(":")
        arms.append((name, drafter, extra.split() if extra else []))

    for rep in range(1, args.reps + 1):
        for cls, qid in qids:
            prompt = questions[qid]["turns"][0]
            for name, drafter, extra in arms:
                out = ROOT / f"target/suite-{args.tag}-{name}-{qid}-{rep}.json"
                cmd = [str(ROOT / "target/release/lmbrrr"), "dspark-run",
                       "--drafter", drafter, "--prompt", prompt,
                       "--max-new-tokens", str(args.max_new_tokens)]
                # Defaults an arm's extra flags may override (dspark-run
                # rejects duplicate flags).
                for flag, value in [("--quantized-manifest", str(ROOT / QMAN)),
                                    ("--gamma", str(args.gamma)),
                                    ("--drafter-quantize", "q8-0"),
                                    ("--quantize-lm-head", "q4k"),
                                    ("--cost-model", str(ROOT / COST))]:
                    if flag not in extra:
                        cmd += [flag, value]
                cmd += ["--output", str(out), *extra]
                proc = subprocess.run(cmd, capture_output=True, cwd=ROOT,
                                      check=False, text=True)
                if proc.returncode != 0 or not out.exists():
                    tail = (proc.stderr or proc.stdout or "").strip()[-300:]
                    print(f"rep{rep} {cls} q{qid} {name}: FAILED rc="
                          f"{proc.returncode} :: {tail}", flush=True)
                else:
                    print(f"rep{rep} {cls} q{qid} {name}: done", flush=True)

    print(f"\n{'class':>15} {'qid':>4} {'arm':>8} {'tok/s':>12} {'tau':>5}")
    for cls, qid in qids:
        texts = {}
        for name, _, _ in arms:
            reports = []
            for rep in range(1, args.reps + 1):
                p = ROOT / f"target/suite-{args.tag}-{name}-{qid}-{rep}.json"
                if p.exists():
                    reports.append(json.loads(p.read_text()))
            if not reports:
                print(f"{cls:>15} {qid:>4} {name:>8}  (no reports)")
                continue
            texts[name] = reports[0].get("committed_text")
            tps = [r["tokens_per_second"] for r in reports]
            tau = st.mean(r["mean_accepted_length"] for r in reports)
            print(f"{cls:>15} {qid:>4} {name:>8} {st.mean(tps):7.1f}"
                  f"±{st.pstdev(tps):3.1f} {tau:5.2f}")
        # Different scheduling -> different chunk shapes -> kernel-noise
        # tie-flips can diverge the committed text between arms; tok/s is
        # then comparing different generations and must not be read as a
        # scheduling-economics delta on this question.
        if len(set(texts.values())) > 1:
            print(f"{'':>15} {qid:>4} {'!':>8}  DIVERGENT CONTENT across arms "
                  f"(tie-flip): tok/s not comparable on this question")
    return 0


if __name__ == "__main__":
    sys.exit(main())
