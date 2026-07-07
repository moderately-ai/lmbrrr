#!/usr/bin/env python3
"""Generate MiniCPM-V-4.6 Transformers parity fixtures.

The default path only needs Hugging Face Transformers and the local MiniCPM
metadata files. Passing --with-next-token additionally loads the model weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = Path("docs/research/models/minicpm-v-4.6/hf-model")
MODEL_ID = "openbmb/MiniCPM-V-4.6"
REVISION = "main"


@dataclass(frozen=True)
class Case:
    id: str
    user_prompt: str
    image_count: int = 0
    enable_thinking: bool = False
    image_paths: tuple[str, ...] = ()


DEFAULT_CASES = (
    Case(
        id="text_closed_thinking_short",
        user_prompt="What is the capital of France?",
    ),
    Case(
        id="text_open_thinking_math",
        user_prompt="Solve 17 * 23. Think carefully.",
        enable_thinking=True,
    ),
    Case(
        id="text_closed_thinking_long_reasoning",
        user_prompt=(
            "Solve this carefully. A lab runs three model-evaluation batches. "
            "Batch A has 18 prompts and each prompt takes 7 seconds. Batch B "
            "has twice as many prompts, but each prompt takes 5 seconds. "
            "Batch C has 12 prompts, each taking 11 seconds, and can only "
            "start after Batch A finishes. If Batch A and Batch B start "
            "together, what is the earliest time when all three batches are "
            "complete?"
        ),
    ),
    Case(
        id="single_image_closed_thinking",
        user_prompt="What causes this phenomenon?",
        image_count=1,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        help="optional model checkpoint directory for --with-next-token",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image", action="append", default=[], help="image path for the single-image case")
    parser.add_argument("--downsample-mode", default="16x", choices=("16x", "4x"))
    parser.add_argument("--max-slice-nums", type=int, default=36)
    parser.add_argument("--with-next-token", action="store_true")
    parser.add_argument("--top-k-logits", type=int, default=10)
    return parser.parse_args()


def require_transformers() -> Any:
    try:
        from transformers import AutoProcessor, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install a MiniCPM-capable Transformers build, for example "
            '`pip install "transformers[torch]>=5.7.0" torchvision`.'
        ) from exc
    return AutoProcessor, AutoTokenizer


def messages_for(case: Case) -> list[dict[str, Any]]:
    if case.image_count == 0:
        content: str | list[dict[str, Any]] = case.user_prompt
    else:
        content = []
        for index in range(case.image_count):
            item = {"type": "image"}
            if index < len(case.image_paths):
                item["url"] = case.image_paths[index]
            content.append(item)
        content.append({"type": "text", "text": case.user_prompt})
    return [{"role": "user", "content": content}]


def normalize_token_ids(tokenized: Any) -> list[int]:
    if hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return [int(token_id) for token_id in tokenized]


def fixture_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    AutoProcessor, AutoTokenizer = require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    processor = None

    cases = list(DEFAULT_CASES)
    if args.image:
        cases = [
            case
            if case.image_count == 0
            else Case(
                id=case.id,
                user_prompt=case.user_prompt,
                image_count=len(args.image),
                enable_thinking=case.enable_thinking,
                image_paths=tuple(args.image),
            )
            for case in cases
        ]

    rows = []
    for case in cases:
        messages = messages_for(case)
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=case.enable_thinking,
        )
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=case.enable_thinking,
        )
        row = {
            **asdict(case),
            "rendered_prompt": rendered,
            "prompt_token_count": len(normalize_token_ids(tokenized)),
            "token_ids": normalize_token_ids(tokenized),
        }

        if case.image_paths:
            if processor is None:
                processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
            processed = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors=None,
                enable_thinking=case.enable_thinking,
                processor_kwargs={
                    "downsample_mode": args.downsample_mode,
                    "max_slice_nums": args.max_slice_nums,
                },
            )
            row["expanded_prompt_token_count"] = len(normalize_token_ids(processed))
            row["expanded_token_ids"] = normalize_token_ids(processed)
            row["image_paths"] = list(case.image_paths)

        rows.append(row)
    return rows


def attach_next_token_logits(args: argparse.Namespace, fixture: dict[str, Any]) -> None:
    try:
        import torch
        from transformers import AutoModelForImageTextToText
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency for --with-next-token: install torch and Transformers with image-text support."
        ) from exc

    _, AutoTokenizer = require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model_dir = args.weights_dir if args.weights_dir is not None else args.model_dir
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    for row in fixture["cases"]:
        if row["image_count"]:
            row["next_token_logits"] = {
                "skipped": "image next-token logits require processor image inputs in the oracle command"
            }
            continue

        input_ids = torch.tensor([row["token_ids"]], device=model.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits[0, -1].float()
            top = torch.topk(logits, k=args.top_k_logits)
        ids = top.indices.detach().cpu().tolist()
        values = top.values.detach().cpu().tolist()
        row["next_token_logits"] = {
            "top_k": args.top_k_logits,
            "top_token_ids": ids,
            "top_tokens": [tokenizer.decode([token_id]) for token_id in ids],
            "top_logits": values,
        }


def main() -> int:
    args = parse_args()
    fixture = {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "model_dir": str(args.model_dir),
        "weights_dir": str(args.weights_dir) if args.weights_dir is not None else None,
        "downsample_mode": args.downsample_mode,
        "source": "Transformers AutoTokenizer/AutoProcessor apply_chat_template",
        "cases": fixture_cases(args),
    }
    if args.with_next_token:
        attach_next_token_logits(args, fixture)

    payload = json.dumps(fixture, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
