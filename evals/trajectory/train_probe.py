#!/usr/bin/env python3
"""Train a simple accept-prefix probe from harvest JSON (oracle + accept_probe rows).

Kill bar: AUC >= 0.85 on held-out positions for on_accept_prefix label.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
from collections import defaultdict

def load_rows(paths):
    probe, oracle = [], []
    for p in paths:
        text = Path(p).read_text(errors="replace")
        # last JSON object with keys
        for line in reversed(text.splitlines()):
            s=line.strip()
            if s.startswith("{") and "decode_tokens_per_second" in s:
                try:
                    j=json.loads(s)
                except Exception:
                    continue
                if j.get("accept_probe_rows"):
                    for r in j["accept_probe_rows"]:
                        r["_file"]=str(p)
                        probe.append(r)
                if j.get("oracle_rounds"):
                    for r in j["oracle_rounds"]:
                        r["_file"]=str(p)
                        oracle.append(r)
                break
    return probe, oracle

def auc(scores, labels):
    # Mann-Whitney AUC
    pos=[s for s,y in zip(scores,labels) if y==1]
    neg=[s for s,y in zip(scores,labels) if y==0]
    if not pos or not neg:
        return float("nan")
    pair=0.0
    for p in pos:
        for n in neg:
            if p>n: pair+=1
            elif p==n: pair+=0.5
    return pair/(len(pos)*len(neg))

def main(argv):
    paths=sorted(Path(argv[1]).glob("*.json")) if len(argv)>1 else []
    if not paths:
        print("usage: train_probe.py HARVEST_DIR"); sys.exit(2)
    probe, oracle = load_rows(paths)
    print(f"files={len(paths)} probe_rows={len(probe)} oracle_rounds={len(oracle)}")

    # --- conf-only AUC on exact_mask positions from oracle ---
    conf_s, conf_y = [], []
    for r in oracle:
        conf=r.get("conf") or []
        mask=r.get("exact_mask") or []
        if not conf or not mask: continue
        for i,ok in enumerate(mask):
            if i < len(conf):
                conf_s.append(float(conf[i])); conf_y.append(1 if ok else 0)
    print(f"oracle position conf AUC={auc(conf_s,conf_y):.4f} n={len(conf_y)} pos={sum(conf_y)}")

    # prefix length regression features
    ep=[r.get("exact_prefix",0) for r in oracle if r.get("mean_conf") is not None]
    mc=[float(r["mean_conf"]) for r in oracle if r.get("mean_conf") is not None]
    if ep and mc:
        # corr
        n=len(ep); me=sum(ep)/n; mm=sum(mc)/n
        num=sum((a-me)*(b-mm) for a,b in zip(ep,mc))
        de=math.sqrt(sum((a-me)**2 for a in ep)); dm=math.sqrt(sum((b-mm)**2 for b in mc))
        print(f"corr(mean_conf, exact_prefix)={num/(de*dm+1e-12):.4f}")

    # accept_probe layer features
    if probe:
        label_key="on_accept_prefix" if "on_accept_prefix" in probe[0] else "exact"
        # conf
        cs, cy = [], []
        for r in probe:
            if "conf" not in r: continue
            cs.append(float(r["conf"])); cy.append(1 if r.get(label_key) else 0)
        print(f"probe conf AUC={auc(cs,cy):.4f} n={len(cy)}")
        # each L*_rms
        keys=sorted({k for r in probe for k in r if k.endswith("_rms")})
        for k in keys:
            s,y=[],[]
            for r in probe:
                if k not in r: continue
                s.append(float(r[k])); y.append(1 if r.get(label_key) else 0)
            a=auc(s,y)
            print(f"  {k} AUC={a:.4f} n={len(y)}")
        # logistic-ish: score = conf + w*pos (grid)
        best=0; best_w=0
        for w in [i/10 for i in range(-20,21)]:
            s=[float(r.get("conf",0)) + w*float(r.get("pos",0)) for r in probe if "conf" in r]
            y=[1 if r.get(label_key) else 0 for r in probe if "conf" in r]
            a=auc(s,y)
            if a>best: best, best_w = a, w
        print(f"best conf+w*pos AUC={best:.4f} w={best_w}")
        print(f"KILL bar 0.85: {'PASS' if best>=0.85 else 'FAIL'}")

if __name__=="__main__":
    main(sys.argv)
