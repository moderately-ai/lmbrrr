#!/usr/bin/env python3
"""Train a small EAGLE-style direct-token draft head from lmbrrr traces."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


@dataclass(frozen=True)
class Sample:
    trace: str
    prompt: str
    step: int
    phase: str
    context_position: int
    token_id: int
    token_text: str
    eos: bool
    feature: list[float]


class EagleDraftHead(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.net(xs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small direct-token EAGLE draft head from lmbrrr trace JSON."
    )
    parser.add_argument("--trace", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=299792458)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--draft-width", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--include-eos", action="store_true")
    return parser.parse_args()


def load_trace(path: Path, include_eos: bool) -> tuple[list[int], list[Sample]]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("kind") != "lmbrrr_hidden_state_trace":
        raise ValueError(f"{path} is not an lmbrrr trace report")

    capture_layers = list(report["capture_layers"])
    samples: list[Sample] = []
    for step in report["steps"]:
        if step.get("eos") and not include_eos:
            continue
        states = sorted(step["hidden_states"], key=lambda state: state["layer_index"])
        layers = [state["layer_index"] for state in states]
        if layers != capture_layers:
            raise ValueError(
                f"{path} step {step['step']} has hidden layers {layers}, expected {capture_layers}"
            )
        feature: list[float] = []
        for state in states:
            feature.extend(float(value) for value in state["values"])
        samples.append(
            Sample(
                trace=str(path),
                prompt=report["prompt"],
                step=int(step["step"]),
                phase=step["phase"],
                context_position=int(step["context_position"]),
                token_id=int(step["target_token_id"]),
                token_text=step.get("target_token", ""),
                eos=bool(step.get("eos", False)),
                feature=feature,
            )
        )
    return capture_layers, samples


def load_samples(paths: list[Path], include_eos: bool) -> tuple[list[int], list[Sample]]:
    all_samples: list[Sample] = []
    expected_layers: list[int] | None = None
    input_dim: int | None = None
    for path in paths:
        layers, samples = load_trace(path, include_eos)
        if expected_layers is None:
            expected_layers = layers
        elif layers != expected_layers:
            raise ValueError(f"{path} captured layers {layers}, expected {expected_layers}")
        for sample in samples:
            if input_dim is None:
                input_dim = len(sample.feature)
            elif len(sample.feature) != input_dim:
                raise ValueError(
                    f"{sample.trace} step {sample.step} feature dim {len(sample.feature)}, "
                    f"expected {input_dim}"
                )
        all_samples.extend(samples)

    if expected_layers is None or input_dim is None:
        raise ValueError("no training samples found")
    return expected_layers, all_samples


def split_indices(sample_count: int, eval_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(sample_count))
    random.Random(seed).shuffle(indices)
    if sample_count < 8 or eval_fraction <= 0:
        return indices, []
    eval_count = max(1, min(sample_count - 1, round(sample_count * eval_fraction)))
    return indices[eval_count:], indices[:eval_count]


def metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    indices: list[int],
    samples: list[Sample],
    draft_width: int,
) -> dict[str, Any]:
    if not indices:
        return {
            "samples": 0,
            "loss": None,
            "top1_accuracy": None,
            "top5_accuracy": None,
            "mean_accepted_prefix": None,
        }

    subset_logits = logits[indices]
    subset_labels = labels[indices]
    loss = torch.nn.functional.cross_entropy(subset_logits, subset_labels).item()
    top1 = subset_logits.argmax(dim=-1)
    top1_accuracy = (top1 == subset_labels).float().mean().item()
    top_k = min(5, subset_logits.shape[-1])
    topk = subset_logits.topk(top_k, dim=-1).indices
    top5_accuracy = (topk == subset_labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    predictions = {idx: int(top1[row].item()) for row, idx in enumerate(indices)}
    mean_accepted = accepted_prefix_estimate(predictions, labels, indices, samples, draft_width)
    return {
        "samples": len(indices),
        "loss": loss,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "mean_accepted_prefix": mean_accepted,
    }


def accepted_prefix_estimate(
    predictions: dict[int, int],
    labels: torch.Tensor,
    indices: list[int],
    samples: list[Sample],
    draft_width: int,
) -> float | None:
    available = set(indices)
    by_trace_step: dict[str, dict[int, int]] = {}
    for idx, sample in enumerate(samples):
        by_trace_step.setdefault(sample.trace, {})[sample.step] = idx
    accepted_lengths: list[int] = []
    for idx in indices:
        sample = samples[idx]
        accepted = 0
        trace_steps = by_trace_step[sample.trace]
        for delta in range(draft_width):
            next_idx = trace_steps.get(sample.step + delta)
            if next_idx is None or next_idx not in available:
                break
            if predictions.get(next_idx) != int(labels[next_idx].item()):
                break
            accepted += 1
        accepted_lengths.append(accepted)
    if not accepted_lengths:
        return None
    return sum(accepted_lengths) / len(accepted_lengths)


def token_vocabulary(samples: list[Sample]) -> tuple[list[int], dict[int, int]]:
    token_ids = sorted({sample.token_id for sample in samples})
    return token_ids, {token_id: idx for idx, token_id in enumerate(token_ids)}


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    capture_layers, samples = load_samples(args.trace, args.include_eos)
    token_ids, label_by_token = token_vocabulary(samples)
    x = torch.tensor([sample.feature for sample in samples], dtype=torch.float32)
    y = torch.tensor([label_by_token[sample.token_id] for sample in samples], dtype=torch.long)
    feature_mean = x.mean(dim=0)
    feature_std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_norm = (x - feature_mean) / feature_std

    train_indices, eval_indices = split_indices(len(samples), args.eval_fraction, args.seed)
    device = torch.device(args.device)
    model = EagleDraftHead(x.shape[1], args.hidden_dim, len(token_ids)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_x = x_norm[train_indices].to(device)
    train_y = y[train_indices].to(device)

    loss_value = math.nan
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = torch.nn.functional.cross_entropy(logits, train_y)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())

    model_cpu = model.cpu()
    with torch.no_grad():
        all_logits = model_cpu(x_norm)
    train_metrics = metrics(all_logits, y, train_indices, samples, args.draft_width)
    eval_metrics = metrics(all_logits, y, eval_indices, samples, args.draft_width)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "weights.safetensors"
    save_file(
        {
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "net.0.weight": model_cpu.net[0].weight,
            "net.0.bias": model_cpu.net[0].bias,
            "net.2.weight": model_cpu.net[2].weight,
            "net.2.bias": model_cpu.net[2].bias,
        },
        weights_path,
    )

    manifest = {
        "kind": "lmbrrr_eagle_draft_head",
        "schema_version": 1,
        "draft_head_type": "observed-vocabulary-mlp",
        "weights": weights_path.name,
        "capture_layers": capture_layers,
        "input_dim": int(x.shape[1]),
        "hidden_dim": args.hidden_dim,
        "output_dim": len(token_ids),
        "token_ids": token_ids,
        "feature_normalization": "zscore",
        "dataset": {
            "traces": [str(path) for path in args.trace],
            "samples": len(samples),
            "train_samples": len(train_indices),
            "eval_samples": len(eval_indices),
            "include_eos": args.include_eos,
            "unique_target_tokens": len(token_ids),
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "final_train_loss": loss_value,
            "device": args.device,
        },
        "metrics": {
            "draft_width": args.draft_width,
            "train": train_metrics,
            "eval": eval_metrics,
        },
        "samples": [
            {
                "trace": sample.trace,
                "step": sample.step,
                "phase": sample.phase,
                "context_position": sample.context_position,
                "target_token_id": sample.token_id,
                "target_token": sample.token_text,
            }
            for sample in samples
        ],
        "limits": [
            "The output vocabulary is restricted to token ids observed in the trace set.",
            "This smoke artifact is useful for validating training/export plumbing, not for speedup claims.",
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = train(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
