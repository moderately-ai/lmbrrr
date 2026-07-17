#!/usr/bin/env python3
"""Regenerate assistant answers with the target model via transformers.

Reads DeepSpec-format JSONL ({"id", "conversations": [{role, content}]}),
replaces every assistant turn with a fresh target-model generation in
non-thinking mode, and writes the same schema back. This replaces DeepSpec's
sglang-based generate_train_data.py with a plain transformers loop so the
labels come from the exact implementation our parity oracle validated.

Batched across conversations; multi-turn conversations advance one turn per
pass with all active conversations generated together (left padding).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field


THINK_SCAFFOLD_END = "</think>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-context-tokens", type=int, default=3072)
    parser.add_argument(
        "--sort-by-length",
        action="store_true",
        help=(
            "Admit conversations in ascending prompt-length order so batches "
            "are length-homogeneous: model.generate runs every batch to its "
            "slowest member, so mixed batches burn compute on pad-waiting."
        ),
    )
    parser.add_argument(
        "--prefill-token-budget",
        type=int,
        default=32768,
        help=(
            "Cap batch_size * padded_prompt_len: HF generate materializes "
            "full-sequence prefill logits (batch x len x 248k vocab fp32), "
            "so long-prompt batches must shrink. 32768 positions ~= 32 GiB "
            "of prefill logits."
        ),
    )
    parser.add_argument(
        "--resume-glob",
        default=None,
        help=(
            "Glob of prior output files for THIS shard (e.g. "
            "'.../regen-shard03-*.jsonl'). Conversation ids already completed "
            "in them are skipped, so a shard re-run after an infra reschedule "
            "regenerates only the remainder (within-shard incremental resume). "
            "This process writes to its own new file; the merge dedups by id."
        ),
    )
    return parser.parse_args()


@dataclass
class ActiveConversation:
    row_id: object
    user_turns: list[str]
    next_turn: int = 0
    messages: list[dict] = field(default_factory=list)

    def done(self) -> bool:
        return self.next_turn >= len(self.user_turns)


def clean_generated(text: str) -> str:
    # The non-thinking chat template puts the empty think scaffold in the
    # generation prompt, so the model should not re-emit it; strip defensively
    # if it does, since assistant content must stay scaffold-free for the
    # training parser.
    text = text.strip()
    if text.startswith("<think>"):
        end = text.find(THINK_SCAFFOLD_END)
        if end != -1:
            text = text[end + len(THINK_SCAFFOLD_END) :].strip()
    return text


def main() -> int:
    args = parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        assert isinstance(im_end_id, int) and im_end_id >= 0
        tokenizer.pad_token_id = im_end_id
    model = (
        AutoModelForImageTextToText.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]

    pending: list[ActiveConversation] = []
    skipped = 0
    with open(args.input, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line_no < args.skip_samples:
                continue
            if args.num_samples is not None and len(pending) >= args.num_samples:
                break
            row = json.loads(line)
            turns = [
                message["content"]
                for message in row.get("conversations", [])
                if message.get("role") == "user"
            ]
            if not turns:
                skipped += 1
                continue
            pending.append(ActiveConversation(row_id=row.get("id", line_no), user_turns=turns))
    print(f"loaded {len(pending)} conversations ({skipped} skipped)", flush=True)

    # Incremental resume: drop conversations already completed in prior attempts
    # of this shard (each row_id is globally unique — the dataset id, else the
    # absolute input line number). Reads defensively: a prior file may be
    # mid-write from a concurrent duplicate attempt, so malformed tail lines
    # are ignored. This attempt writes to its OWN file; the merge dedups by id.
    if args.resume_glob:
        import glob as _glob

        done_ids: set = set()
        for prior in _glob.glob(args.resume_glob):
            try:
                handle = open(prior, "r", encoding="utf-8")
            except FileNotFoundError:
                continue
            with handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        prior_row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if prior_row.get("status") == "success" and "id" in prior_row:
                        done_ids.add(prior_row["id"])
        before = len(pending)
        pending = [conv for conv in pending if conv.row_id not in done_ids]
        print(
            f"resume: {len(done_ids)} completed in prior files; "
            f"skipping {before - len(pending)} of {before}, "
            f"regenerating {len(pending)}",
            flush=True,
        )

    if args.sort_by_length:
        pending.sort(key=lambda conv: sum(len(turn) for turn in conv.user_turns))

    finished: list[ActiveConversation] = []
    error_rows: list[dict] = []
    total_generated_tokens = 0
    import time as _time

    decode_start = _time.monotonic()

    with open(args.output, "w", encoding="utf-8") as out_handle:
        while pending:
            # Token-budget admission: build prompts for a candidate window,
            # measure real token lengths, and take the longest prefix whose
            # padded footprint (n * max_len) fits the prefill-logits budget.
            window = pending[: args.batch_size]
            window_prompts = []
            for conv in window:
                messages = conv.messages + [
                    {"role": "user", "content": conv.user_turns[conv.next_turn]}
                ]
                window_prompts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
            lengths = [
                min(len(ids), args.max_context_tokens)
                for ids in tokenizer(window_prompts, add_special_tokens=False)["input_ids"]
            ]
            take = 1
            max_len = lengths[0]
            for i in range(1, len(window)):
                candidate_max = max(max_len, lengths[i])
                if (i + 1) * candidate_max > args.prefill_token_budget:
                    break
                max_len = candidate_max
                take = i + 1
            batch = window[:take]
            prompts = window_prompts[:take]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_context_tokens,
                add_special_tokens=False,
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )
            new_tokens = generated[:, encoded.input_ids.shape[1] :]
            total_generated_tokens += int(new_tokens.numel())
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            still_pending: list[ActiveConversation] = []
            for conv, text in zip(batch, texts):
                content = clean_generated(text)
                if not content:
                    error_rows.append(
                        {"id": conv.row_id, "status": "error", "error": "empty generation"}
                    )
                    continue
                conv.messages.append(
                    {"role": "user", "content": conv.user_turns[conv.next_turn]}
                )
                conv.messages.append({"role": "assistant", "content": content})
                conv.next_turn += 1
                if conv.done():
                    finished.append(conv)
                    out_handle.write(
                        json.dumps(
                            {
                                "id": conv.row_id,
                                "conversations": conv.messages,
                                "status": "success",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                else:
                    still_pending.append(conv)
            pending = still_pending + pending[take:]
            out_handle.flush()
            print(
                f"progress: finished={len(finished)} pending={len(pending)} "
                f"errors={len(error_rows)} generated_tokens~={total_generated_tokens}",
                flush=True,
            )

    if error_rows:
        error_path = args.output.replace(".jsonl", "_error.jsonl")
        with open(error_path, "w", encoding="utf-8") as handle:
            for row in error_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    elapsed = _time.monotonic() - decode_start
    print(
        f"done: {len(finished)} success, {len(error_rows)} errors, "
        f"~{total_generated_tokens} generated tokens, "
        f"{elapsed:.1f}s generation wall, "
        f"{total_generated_tokens / max(elapsed, 1e-9):.1f} padded-tok/s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
