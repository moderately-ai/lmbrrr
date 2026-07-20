#!/usr/bin/env python3
"""adapt-margin suite: shorter class prompts to fit M3 18GB envelope."""
import json, os, statistics, subprocess, time
from pathlib import Path

REPO = Path.home() / "lmbrrr-work/lmbrrr"
BIN = REPO / "target/release/lmbrrr"
GGUF = Path.home() / "models/Ternary-Bonsai-27B/Ternary-Bonsai-27B-Q2_0.gguf"
DRAFTER = Path.home() / "models/Ternary-Bonsai-27B/Ternary-Bonsai-27B-dspark-Q8_0.gguf"
OUT = Path("/tmp/adapt_suite2_out")
OUT.mkdir(exist_ok=True)

N = 96
REPS = 3

# Short, class-representative prompts (avoid long-prefill OOM on 18GB)
PROMPTS = [
    ("math", "What is the derivative of x^3 + 2x? Show the steps briefly."),
    ("code", "Write a Python function that merges two sorted lists into one sorted list. Include a short doctest."),
    ("prose", "Explain quantum computing in simple terms for a curious teenager."),
    ("fact", "Who was the first person to walk on the Moon, and in what year? One short paragraph."),
    ("summarize", "Summarize in 3 bullet points: Magpies are highly intelligent birds that use tools, recognize themselves in mirrors, and form complex social groups."),
    ("translate", "Translate to French: The library opens at nine and closes at six on weekdays."),
]

ARMS = {
    "m1": [],
    "m3": ["--fast"],
    "adapt": ["--adapt-margin", "1.0,2.0"],
}

def run_spec(prompt, extra, tag):
    cmd = [
        str(BIN), "gguf", "spec",
        "--gguf", str(GGUF), "--drafter", str(DRAFTER),
        "--prompt", prompt, "--max-new-tokens", str(N), *extra,
    ]
    outp, errp = OUT / f"{tag}.out", OUT / f"{tag}.err"
    with open(outp, "w") as o, open(errp, "w") as e:
        subprocess.check_call(cmd, stdout=o, stderr=e)
    for line in reversed(outp.read_text().splitlines()):
        if line.strip().startswith("{") and "decode_tokens_per_second" in line:
            return json.loads(line.strip())
    raise RuntimeError(f"no json {tag}: {errp.read_text()[-400:]}")

def run_score(prompt, ids):
    cmd = [str(BIN), "gguf", "score", "--gguf", str(GGUF), "--prompt", prompt, "--ids", ids]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
    for line in reversed(out.splitlines()):
        if line.strip().startswith("{") and "ppl" in line:
            return json.loads(line.strip())
    raise RuntimeError("no score")

def main():
    rows = []
    for rep in range(REPS):
        order = list(ARMS.items())
        if rep % 2:
            order = list(reversed(order))
        for cls, prompt in PROMPTS:
            for aname, extra in order:
                tag = f"{cls}_{aname}_r{rep}"
                print("RUN", tag, flush=True)
                j = run_spec(prompt, extra, tag)
                s = run_score(prompt, j["ids"])
                row = dict(
                    cls=cls, arm=aname, rep=rep,
                    tps=j["decode_tokens_per_second"],
                    acc=j["mean_accepted_per_round"],
                    ppl=s["ppl"], min_lp=s["min_logprob"],
                    mean_lp=s["mean_logprob"],
                    rounds=j["rounds"],
                    ex=j.get("adapt_exact_rounds"),
                    base=j.get("adapt_base_rounds"),
                    fast=j.get("adapt_fast_rounds"),
                )
                rows.append(row)
                print(
                    f"  tps={row['tps']:.2f} acc={row['acc']:.3f} ppl={row['ppl']:.3f} minlp={row['min_lp']:.2f} "
                    f"sched={row['ex']}/{row['base']}/{row['fast']}",
                    flush=True,
                )
    Path("/tmp/adapt_suite2_rows.json").write_text(json.dumps(rows, indent=2))

    classes = [c for c, _ in PROMPTS]
    arms = ["m1", "m3", "adapt"]
    print("\n======== PER-CLASS MEDIAN (3 reps) ========")
    hdr = f"{'class':12} {'m1_tps':8} {'m3_tps':8} {'ad_tps':8} {'m1_ppl':8} {'m3_ppl':8} {'ad_ppl':8} {'d_tps_m1':9} {'d_ppl_m3':9}"
    print(hdr)
    stats = {}
    for cls in classes:
        st = {}
        for arm in arms:
            rs = [r for r in rows if r["cls"] == cls and r["arm"] == arm]
            st[arm] = {
                "tps": statistics.median(r["tps"] for r in rs),
                "ppl": statistics.median(r["ppl"] for r in rs),
                "acc": statistics.median(r["acc"] for r in rs),
                "min_lp": statistics.median(r["min_lp"] for r in rs),
            }
        stats[cls] = st
        dt = 100 * (st["adapt"]["tps"] - st["m1"]["tps"]) / st["m1"]["tps"]
        dp = 100 * (st["adapt"]["ppl"] - st["m3"]["ppl"]) / st["m3"]["ppl"]
        print(
            f"{cls:12} {st['m1']['tps']:8.2f} {st['m3']['tps']:8.2f} {st['adapt']['tps']:8.2f} "
            f"{st['m1']['ppl']:8.3f} {st['m3']['ppl']:8.3f} {st['adapt']['ppl']:8.3f} "
            f"{dt:+8.1f}% {dp:+8.1f}%"
        )

    print("\n======== OVERALL ========")
    def mean_arm(arm, key):
        return statistics.mean(stats[c][arm][key] for c in classes)
    for arm in arms:
        print(f"{arm:6} mean_tps={mean_arm(arm,'tps'):.2f} mean_ppl={mean_arm(arm,'ppl'):.3f} mean_acc={mean_arm(arm,'acc'):.3f}")
    m1t, m3t, adt = mean_arm("m1","tps"), mean_arm("m3","tps"), mean_arm("adapt","tps")
    m1p, m3p, adp = mean_arm("m1","ppl"), mean_arm("m3","ppl"), mean_arm("adapt","ppl")
    print(f"adapt vs m1 tps {100*(adt-m1t)/m1t:+.2f}%")
    print(f"adapt vs m3 tps {100*(adt-m3t)/m3t:+.2f}%")
    print(f"adapt vs m1 ppl {100*(adp-m1p)/m1p:+.2f}%")
    print(f"adapt vs m3 ppl {100*(adp-m3p)/m3p:+.2f}%")

    tps_ok = adt >= m1t * 1.02
    worse_m3 = [c for c in classes if stats[c]["adapt"]["ppl"] > stats[c]["m3"]["ppl"] * 1.02]
    worse_m1 = [c for c in classes if stats[c]["adapt"]["ppl"] > stats[c]["m1"]["ppl"] * 1.05]
    print("\n======== GATES ========")
    print(f"tps +2% vs m1: {'PASS' if tps_ok else 'FAIL'} ({100*(adt-m1t)/m1t:+.2f}%)")
    print(f"no class ppl > m3+2%: {'PASS' if not worse_m3 else 'FAIL'} worse={worse_m3}")
    print(f"no class ppl > m1+5%: {'PASS' if not worse_m1 else 'FAIL'} worse={worse_m1}")
    print("VERDICT:", "DEFAULT-ON CANDIDATE" if tps_ok and not worse_m3 else "KEEP OPT-IN")

if __name__ == "__main__":
    main()
