#!/usr/bin/env python3
"""Build a task-balanced PROMPT corpus for drafter distillation.

The MTP-head drafter is distilled to match the fakequant MiniCPM-V-4.6 target on
its OWN generations (self-distillation), so this script emits only PROMPTS (user
turns) in the ShareGPT-style schema the regen pipeline consumes — the target
regenerates every assistant answer downstream. Answer quality in the source
datasets is irrelevant; only prompt distribution matters.

Motivation (2026-07-15 corpus audit): the inherited Open-PerfectBlend corpus is
~67% math / ~20% code / ~1% translation / ~2% summarization, which starves the
Spec-Bench weak classes (translation, summarization, qa, writing) that hold the
decode mean under greedy. This blend rebalances to give every class real signal,
per the multilingual-specialized-drafters result (arXiv 2406.16758): a general
chat/math blend is weak on translation, and in-domain self-distilled data fixes
it. Sources chosen for permissive licensing + eval-domain match, and every
prompt is decontaminated against the Spec-Bench eval prompts + GSM8K test.

Each source is STREAMED with early-stop at its target count, so no full dataset
download is needed. Prompt surface forms mirror the eval (e.g. "Translate German
to English: ..."), so the target generates in the same style the drafter meets
at decode time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re

from datasets import load_dataset


# ---- per-class source plan (counts sum to ~500k) ---------------------------
# Each entry: (class, source-key, kwargs). Adapters below turn a streamed row
# into a prompt string (or None to skip). Counts are the post-filter targets.
def default_plan() -> list[dict]:
    return [
        # math (downsampled from the 67% dominance; still ~5x r1's whole corpus)
        {"cls": "math", "src": "metamathqa", "count": 70000},
        {"cls": "math", "src": "orca_math", "count": 40000},
        # coding
        {"cls": "coding", "src": "evol_codealpaca", "count": 65000},
        # translation: WMT19 DE-EN matches the eval domain + tokenized surface
        # form; opus-100 adds other high-resource pairs for robustness.
        {"cls": "translation", "src": "wmt19", "lang": "de", "count": 45000},
        {"cls": "translation", "src": "opus100", "lang": "fr", "count": 10000},
        {"cls": "translation", "src": "opus100", "lang": "es", "count": 8000},
        {"cls": "translation", "src": "opus100", "lang": "ru", "count": 7000},
        {"cls": "translation", "src": "opus100", "lang": "zh", "count": 5000},
        # summarization: XSum (BBC news, != CNN/DM eval), BillSum (CC0, long-doc)
        {"cls": "summarization", "src": "xsum", "count": 50000},
        {"cls": "summarization", "src": "billsum", "count": 15000},
        # open-domain QA (Natural Questions open, the eval's qa source family)
        {"cls": "qa", "src": "nq_open", "count": 45000},
        # RAG: context passage + question (Spec-Bench's 7th class)
        {"cls": "rag", "src": "squad", "count": 40000},
        # writing / general chat: WildChat real user turns (creative writing,
        # brainstorm, open chat — the deployment distribution) + smoltalk
        # rewriting/system-chat subsets for instruction diversity. (Dropped
        # everyday-conversations: ~2.2k rows of repetitive greeting openers.)
        {"cls": "writing", "src": "wildchat", "count": 70000},
        {"cls": "writing", "src": "smoltalk", "cfg": "smol-rewrite", "count": 15000},
        {"cls": "writing", "src": "smoltalk", "cfg": "explore-instruct-rewriting", "count": 8000},
        {"cls": "writing", "src": "smoltalk", "cfg": "systemchats-30k", "count": 7000},
    ]


LANG_NAME = {"de": "German", "fr": "French", "es": "Spanish", "ru": "Russian", "zh": "Chinese"}
MAX_PROMPT_CHARS = 6000  # bound sequence length so prompt+answer fits the trainer


def _first_user(messages) -> str | None:
    for m in messages:
        if m.get("role") == "user" and m.get("content", "").strip():
            return m["content"]
    return None


def stream_prompts(spec: dict):
    """Yield prompt strings for one source spec (already class-tagged)."""
    src = spec["src"]
    if src == "metamathqa":
        for r in load_dataset("meta-math/MetaMathQA", split="train", streaming=True):
            yield r.get("query")
    elif src == "orca_math":
        for r in load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True):
            yield r.get("question")
    elif src == "evol_codealpaca":
        for r in load_dataset("theblackcat102/evol-codealpaca-v1", split="train", streaming=True):
            yield r.get("instruction")
    elif src == "wmt19":
        for r in load_dataset("wmt/wmt19", "de-en", split="train", streaming=True):
            de = r["translation"].get("de")
            yield f"Translate German to English: {de}" if de else None
    elif src == "opus100":
        lang = spec["lang"]
        # opus-100 orders high-resource pairs English-first (en-fr, en-es, ...);
        # de-en is the exception but we source German from WMT19.
        cfg = f"{lang}-en" if lang == "de" else f"en-{lang}"
        for r in load_dataset("Helsinki-NLP/opus-100", cfg, split="train", streaming=True):
            src_text = r["translation"].get(lang)
            yield f"Translate {LANG_NAME[lang]} to English: {src_text}" if src_text else None
    elif src == "xsum":
        for r in load_dataset("EdinburghNLP/xsum", split="train", streaming=True):
            doc = r.get("document")
            yield f"Summarize: {doc}" if doc else None
    elif src == "billsum":
        for r in load_dataset("FiscalNote/billsum", split="train", streaming=True):
            text = r.get("text")
            yield f"Summarize: {text}" if text else None
    elif src == "nq_open":
        for r in load_dataset("google-research-datasets/nq_open", split="train", streaming=True):
            yield r.get("question")
    elif src == "squad":
        for r in load_dataset("rajpurkar/squad", split="train", streaming=True):
            ctx, q = r.get("context"), r.get("question")
            yield f"{ctx}\n\nQuestion: {q}" if ctx and q else None
    elif src == "wildchat":
        for r in load_dataset("allenai/WildChat-1M", split="train", streaming=True):
            if r.get("language") != "English" or r.get("toxic") or r.get("redacted"):
                continue
            yield _first_user(r.get("conversation", []))
    elif src == "smoltalk":
        for r in load_dataset("HuggingFaceTB/smoltalk", spec["cfg"], split="train", streaming=True):
            yield _first_user(r.get("messages", []))
    else:
        raise ValueError(f"unknown source {src}")


# ---- decontamination (word-13-gram vs the eval prompts, mirrors round4) ------
def norm_ngrams(text: str, n: int = 13) -> set:
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def build_eval_grams(spec_bench_path: str) -> set:
    grams: set = set()
    # Spec-Bench prompts (contain the exact WMT14 / CNN-DM / NQ eval items)
    with open(spec_bench_path, encoding="utf-8") as f:
        for line in f:
            for turn in json.loads(line).get("turns", []):
                grams |= norm_ngrams(turn)
    # GSM8K test
    try:
        for row in load_dataset("openai/gsm8k", "main", split="test"):
            grams |= norm_ngrams(row["question"])
    except Exception as e:
        print(f"WARN: gsm8k test unavailable ({e}); eval grams = spec-bench only", flush=True)
    return grams


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, help="blended prompt JSONL")
    p.add_argument("--spec-bench", required=True, help="spec_bench_question.jsonl for decontam")
    p.add_argument("--scale", type=float, default=1.0, help="multiply every target count")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-source-cap", type=int, default=0,
                   help="override every count (smoke/dry-run); 0 = use plan")
    args = p.parse_args()

    rng = random.Random(args.seed)
    print("building eval decontam n-grams...", flush=True)
    eval_grams = build_eval_grams(args.spec_bench)
    print(f"eval grams: {len(eval_grams)}", flush=True)

    plan = default_plan()
    rows: list[dict] = []
    seen: set[str] = set()  # cross-source prompt dedup (exact)
    stats: dict[str, dict] = {}

    for spec in plan:
        target = args.per_source_cap or int(spec["count"] * args.scale)
        variant = spec.get("lang") or spec.get("cfg")
        label = f"{spec['cls']}/{spec['src']}" + (f":{variant}" if variant else "")
        kept = scanned = dropped_contam = dropped_dup = 0
        for prompt in stream_prompts(spec):
            scanned += 1
            if kept >= target:
                break
            if not prompt or not prompt.strip():
                continue
            prompt = prompt.strip()[:MAX_PROMPT_CHARS]
            key = hashlib.md5(prompt.encode("utf-8")).hexdigest()
            if key in seen:
                dropped_dup += 1
                continue
            if norm_ngrams(prompt) & eval_grams:
                dropped_contam += 1
                continue
            seen.add(key)
            rows.append({"cls": spec["cls"], "src": spec["src"], "prompt": prompt})
            kept += 1
            if scanned > target * 40 + 200000:  # safety: stop if source too sparse
                break
        stats[label] = {"kept": kept, "target": target, "scanned": scanned,
                        "contam": dropped_contam, "dup": dropped_dup}
        print(f"{label}: kept {kept}/{target} (scanned {scanned}, "
              f"contam {dropped_contam}, dup {dropped_dup})", flush=True)

    rng.shuffle(rows)
    with open(args.output, "w", encoding="utf-8") as out:
        for i, r in enumerate(rows):
            out.write(json.dumps({
                "id": i,
                "cls": r["cls"],
                "src": r["src"],
                "conversations": [{"role": "user", "content": r["prompt"]}],
            }, ensure_ascii=False) + "\n")

    by_cls: dict[str, int] = {}
    for r in rows:
        by_cls[r["cls"]] = by_cls.get(r["cls"], 0) + 1
    print(f"\nWROTE {len(rows)} prompts -> {args.output}", flush=True)
    print("by class:", json.dumps({k: f"{v} ({100*v/max(len(rows),1):.1f}%)" for k, v in sorted(by_cls.items())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
