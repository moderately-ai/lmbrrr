#!/usr/bin/env python3
"""Generate MiniCPM-V-4.6 image-processor parity fixtures."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MODEL_DIR = Path("docs/research/models/minicpm-v-4.6/hf-model")
MODEL_ID = "openbmb/MiniCPM-V-4.6"
REVISION = "main"


@dataclass(frozen=True)
class ImageCase:
    id: str
    height: int
    width: int


DEFAULT_CASES = (
    ImageCase(id="synthetic_small_unsliced", height=64, width=96),
    ImageCase(id="synthetic_large_sliced", height=900, width=1200),
    ImageCase(id="synthetic_tall_sliced", height=1200, width=600),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-slice-nums", type=int, default=9)
    parser.add_argument("--scale-resolution", type=int, default=448)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--sample-count", type=int, default=64)
    return parser.parse_args()


def require_transformers() -> Any:
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install a MiniCPM-capable Transformers build, for example "
            '`pip install "transformers[torch]>=5.7.0" torchvision`.'
        ) from exc
    return AutoProcessor


def synthetic_image(height: int, width: int) -> Image.Image:
    y = np.arange(height, dtype=np.uint32)[:, None]
    x = np.arange(width, dtype=np.uint32)[None, :]
    image = np.stack(
        (
            (x * 17 + y * 3) % 256,
            (x * 5 + y * 11 + 37) % 256,
            (x * 13 + y * 7 + 91) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    return Image.fromarray(image, "RGB")


def sample_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    seeds = {
        0,
        1,
        13,
        14,
        15,
        length // 7,
        length // 5,
        length // 3,
        length // 2,
        (length * 2) // 3,
        length - 16,
        length - 3,
        length - 2,
        length - 1,
    }
    cursor = 97
    while len(seeds) < count:
        seeds.add((cursor * 104729 + 17) % length)
        cursor += 1
    return sorted(index for index in seeds if 0 <= index < length)[:count]


def fixture_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    AutoProcessor = require_transformers()
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

    rows = []
    for case in DEFAULT_CASES:
        image = synthetic_image(case.height, case.width)
        processed = processor.image_processor(
            [image],
            return_tensors="pt",
            max_slice_nums=args.max_slice_nums,
            scale_resolution=args.scale_resolution,
            patch_size=args.patch_size,
        )
        pixel_values = processed["pixel_values"].detach().cpu().float().contiguous()
        flat = pixel_values.flatten()
        indices = sample_indices(int(flat.numel()), args.sample_count)
        sample_values = [float(flat[index].item()) for index in indices]
        row = {
            **asdict(case),
            "pixel_values_shape": list(pixel_values.shape),
            "target_sizes": processed["target_sizes"].detach().cpu().tolist(),
            "grid": [int(value) for value in processed["grids"][0]],
            "patch_count": int(processed["num_patches_per_image"][0]),
            "sample_indices": indices,
            "sample_values": sample_values,
            "sample_max_abs": max(abs(value) for value in sample_values),
            "pixel_sum": float(pixel_values.sum().item()),
            "pixel_mean": float(pixel_values.mean().item()),
        }
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    fixture = {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "model_dir": str(args.model_dir),
        "source": "Transformers AutoProcessor.image_processor on deterministic synthetic RGB images",
        "max_slice_nums": args.max_slice_nums,
        "scale_resolution": args.scale_resolution,
        "patch_size": args.patch_size,
        "cases": fixture_cases(args),
    }
    payload = json.dumps(fixture, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
