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

    finished: list[ActiveConversation] = []
    error_rows: list[dict] = []
    total_generated_tokens = 0

    with open(args.output, "w", encoding="utf-8") as out_handle:
        while pending:
            batch = pending[: args.batch_size]
            prompts = []
            for conv in batch:
                messages = conv.messages + [
                    {"role": "user", "content": conv.user_turns[conv.next_turn]}
                ]
                prompts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
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
            pending = still_pending + pending[args.batch_size :]
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
    print(
        f"done: {len(finished)} success, {len(error_rows)} errors, "
        f"~{total_generated_tokens} generated tokens",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
