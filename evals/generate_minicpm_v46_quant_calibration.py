#!/usr/bin/env python3
"""Generate MiniCPM-V-4.6 quantization calibration fixtures.

The output is JSONL: one calibration row per prompt. Rows include the rendered
chat-template prompt and token ids so sensitivity scoring can run without
recomputing template expansion by hand.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = Path("docs/research/models/minicpm-v-4.6/hf-model")
DEFAULT_OUTPUT = Path("evals/calibration/minicpm_v46_quant_calibration.jsonl")
MODEL_ID = "openbmb/MiniCPM-V-4.6"
REVISION = "main"


@dataclass(frozen=True)
class MediaDescriptor:
    id: str
    kind: str
    path: str
    source: str
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    duration_seconds: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class CalibrationCase:
    id: str
    category: str
    modality: str
    user_prompt: str
    enable_thinking: bool
    max_new_tokens: int
    expected_behavior: str
    sensitivity_focus: tuple[str, ...]
    media: tuple[MediaDescriptor, ...] = ()
    tools: tuple[dict[str, Any], ...] = ()


WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "weather_lookup",
        "description": "Return a deterministic weather summary for a city and date.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "date": {"type": "string"},
                "units": {"type": "string", "enum": ["metric", "imperial"]},
            },
            "required": ["city", "date"],
        },
    },
}


DEFAULT_CASES: tuple[CalibrationCase, ...] = (
    CalibrationCase(
        id="text_short_factual_closed",
        category="short_factual",
        modality="text",
        user_prompt="Answer in one sentence: what is the capital of France?",
        enable_thinking=False,
        max_new_tokens=32,
        expected_behavior="Short answer should mention Paris.",
        sensitivity_focus=("embedding", "lm_head", "short_prefill", "decode"),
    ),
    CalibrationCase(
        id="text_arithmetic_open_thinking",
        category="arithmetic",
        modality="text",
        user_prompt="Solve 17 * 23. Show the calculation briefly.",
        enable_thinking=True,
        max_new_tokens=96,
        expected_behavior="Final answer should be 391.",
        sensitivity_focus=("deltanet", "mlp", "reasoning_trace", "decode"),
    ),
    CalibrationCase(
        id="text_long_reasoning_closed",
        category="long_reasoning",
        modality="text",
        user_prompt=(
            "Solve this carefully. A lab runs three model-evaluation batches. "
            "Batch A has 18 prompts and each prompt takes 7 seconds. Batch B "
            "has twice as many prompts, but each prompt takes 5 seconds. "
            "Batch C has 12 prompts, each taking 11 seconds, and can only "
            "start after Batch A finishes. If Batch A and Batch B start "
            "together, what is the earliest time when all three batches are "
            "complete?"
        ),
        enable_thinking=False,
        max_new_tokens=160,
        expected_behavior="Answer should account for parallel A/B and dependent C; earliest completion is 258 seconds.",
        sensitivity_focus=("long_prefill", "deltanet", "mlp", "logit_drift"),
    ),
    CalibrationCase(
        id="text_code_completion_closed",
        category="code",
        modality="text",
        user_prompt=(
            "Complete this Rust function and keep the answer concise:\n\n"
            "```rust\n"
            "fn tokens_per_second(tokens: usize, elapsed_seconds: f64) -> f64 {\n"
            "    // return 0.0 when elapsed_seconds is zero\n"
            "}\n"
            "```"
        ),
        enable_thinking=False,
        max_new_tokens=128,
        expected_behavior="Completion should guard zero elapsed time and otherwise divide tokens by seconds.",
        sensitivity_focus=("code_tokens", "mlp", "lm_head"),
    ),
    CalibrationCase(
        id="text_tool_style_closed",
        category="tool_style",
        modality="text",
        user_prompt=(
            "Use the weather_lookup tool for Toronto on 2026-07-07 in metric units, "
            "then summarize whether a light jacket is likely useful."
        ),
        enable_thinking=False,
        max_new_tokens=160,
        expected_behavior="Model should emit the MiniCPM tool-call format for weather_lookup.",
        sensitivity_focus=("chat_template", "tool_tokens", "structured_output"),
        tools=(WEATHER_TOOL,),
    ),
    CalibrationCase(
        id="text_thinking_toggle_closed",
        category="thinking_toggle",
        modality="text",
        user_prompt="Reason about which is larger, 9.11 or 9.9, then give only the final comparison.",
        enable_thinking=False,
        max_new_tokens=80,
        expected_behavior="Closed-thinking prompt should place generated text after an empty think block.",
        sensitivity_focus=("reasoning_control", "decimal_comparison", "decode"),
    ),
    CalibrationCase(
        id="text_thinking_toggle_open",
        category="thinking_toggle",
        modality="text",
        user_prompt="Reason about which is larger, 9.11 or 9.9, then give only the final comparison.",
        enable_thinking=True,
        max_new_tokens=160,
        expected_behavior="Open-thinking prompt should allow visible reasoning before the answer.",
        sensitivity_focus=("reasoning_control", "decimal_comparison", "decode"),
    ),
    CalibrationCase(
        id="vlm_single_small_image_closed",
        category="single_image",
        modality="image",
        user_prompt="Describe the main object and one visible attribute.",
        enable_thinking=False,
        max_new_tokens=96,
        expected_behavior="Image row calibrates one unsliced small image placeholder.",
        sensitivity_focus=("vision_embedding", "merger", "image_placeholder", "short_prefill"),
        media=(
            MediaDescriptor(
                id="synthetic_small_object",
                kind="image",
                path="evals/calibration/media/minicpm_v46/synthetic_small_object.png",
                source="deterministic synthetic image; generated later by multimodal eval tooling",
                width=640,
                height=480,
                notes="Representative small RGB image that should not trigger heavy slicing.",
            ),
        ),
    ),
    CalibrationCase(
        id="vlm_high_resolution_sliced_closed",
        category="high_resolution",
        modality="image",
        user_prompt="List the visible regions from top-left to bottom-right.",
        enable_thinking=False,
        max_new_tokens=128,
        expected_behavior="Image row calibrates high-resolution sliced image placeholder behavior.",
        sensitivity_focus=("vision_slicing", "merger", "long_prefill", "image_placeholder"),
        media=(
            MediaDescriptor(
                id="synthetic_high_res_grid",
                kind="image",
                path="evals/calibration/media/minicpm_v46/synthetic_high_res_grid.png",
                source="deterministic synthetic grid image; generated later by multimodal eval tooling",
                width=1792,
                height=1344,
                notes="Wide high-resolution image chosen to exercise slice expansion.",
            ),
        ),
    ),
    CalibrationCase(
        id="vlm_tall_ocr_closed",
        category="ocr_document",
        modality="image",
        user_prompt="Read the document title and the first total shown in the table.",
        enable_thinking=False,
        max_new_tokens=128,
        expected_behavior="OCR row should preserve small text and table structure once image evals are enabled.",
        sensitivity_focus=("vision_slicing", "ocr", "merger", "logit_drift"),
        media=(
            MediaDescriptor(
                id="synthetic_tall_invoice",
                kind="image",
                path="evals/calibration/media/minicpm_v46/synthetic_tall_invoice.png",
                source="deterministic synthetic OCR document; generated later by multimodal eval tooling",
                width=960,
                height=1920,
                notes="Tall document-like image for OCR and aspect-ratio sensitivity.",
            ),
        ),
    ),
    CalibrationCase(
        id="vlm_multi_image_compare_open",
        category="multi_image",
        modality="image",
        user_prompt="Compare the two images and state the one difference that matters most.",
        enable_thinking=True,
        max_new_tokens=160,
        expected_behavior="Multi-image row calibrates repeated image placeholders and open thinking.",
        sensitivity_focus=("multi_image", "vision_embedding", "merger", "reasoning_trace"),
        media=(
            MediaDescriptor(
                id="synthetic_compare_a",
                kind="image",
                path="evals/calibration/media/minicpm_v46/synthetic_compare_a.png",
                source="deterministic synthetic image pair; generated later by multimodal eval tooling",
                width=768,
                height=768,
                notes="First comparison image.",
            ),
            MediaDescriptor(
                id="synthetic_compare_b",
                kind="image",
                path="evals/calibration/media/minicpm_v46/synthetic_compare_b.png",
                source="deterministic synthetic image pair; generated later by multimodal eval tooling",
                width=768,
                height=768,
                notes="Second comparison image with one controlled difference.",
            ),
        ),
    ),
    CalibrationCase(
        id="video_short_clip_metadata_closed",
        category="video",
        modality="video",
        user_prompt="Summarize the motion in the clip in one sentence.",
        enable_thinking=False,
        max_new_tokens=96,
        expected_behavior="Video row is metadata-only until MiniCPM video processor parity is implemented.",
        sensitivity_focus=("video_placeholder", "future_video_eval"),
        media=(
            MediaDescriptor(
                id="synthetic_short_motion",
                kind="video",
                path="evals/calibration/media/minicpm_v46/synthetic_short_motion.mp4",
                source="deterministic synthetic video; generated later by video eval tooling",
                width=448,
                height=448,
                frames=16,
                duration_seconds=2.0,
                notes="Metadata-only row; no video bytes are committed by this ticket.",
            ),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require_transformers() -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency: run `uv sync` before generating calibration fixtures.") from exc
    return AutoTokenizer


def messages_for(case: CalibrationCase) -> list[dict[str, Any]]:
    if not case.media:
        content: str | list[dict[str, Any]] = case.user_prompt
    else:
        content = [{"type": item.kind} for item in case.media]
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


def apply_chat_template(tokenizer: Any, case: CalibrationCase, tokenize: bool) -> Any:
    kwargs: dict[str, Any] = {
        "tokenize": tokenize,
        "add_generation_prompt": True,
        "enable_thinking": case.enable_thinking,
    }
    if case.tools:
        kwargs["tools"] = list(case.tools)
    return tokenizer.apply_chat_template(messages_for(case), **kwargs)


def row_for(tokenizer: Any, case: CalibrationCase, args: argparse.Namespace) -> dict[str, Any]:
    rendered = apply_chat_template(tokenizer, case, tokenize=False)
    token_ids = normalize_token_ids(apply_chat_template(tokenizer, case, tokenize=True))
    return {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "tokenizer_path": str(args.model_dir / "tokenizer.json"),
        "chat_template_path": str(args.model_dir / "chat_template.jinja"),
        "id": case.id,
        "category": case.category,
        "modality": case.modality,
        "enable_thinking": case.enable_thinking,
        "max_new_tokens": case.max_new_tokens,
        "user_prompt": case.user_prompt,
        "messages": messages_for(case),
        "tools": list(case.tools),
        "media": [asdict(item) for item in case.media],
        "media_status": "metadata_only" if case.media else "none",
        "expected_behavior": case.expected_behavior,
        "sensitivity_focus": list(case.sensitivity_focus),
        "rendered_prompt": rendered,
        "prompt_token_count": len(token_ids),
        "token_ids": token_ids,
    }


def main() -> int:
    args = parse_args()
    AutoTokenizer = require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    rows = [row_for(tokenizer, case, args) for case in DEFAULT_CASES]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    print(f"wrote {len(rows)} calibration rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
