#!/usr/bin/env python3
"""Train a small recurrent EAGLE-style drafter from lmbrrr traces."""

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
class TraceStep:
    trace: str
    prompt: str
    step: int
    target_token_id: int
    target_token: str
    feature: list[float]


@dataclass(frozen=True)
class Sample:
    trace: str
    prompt: str
    anchor_step: int
    draft_position: int
    prev_token_id: int
    target_token_id: int
    target_token: str
    feature: list[float]


class RecurrentDrafter(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.net(xs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a recurrent observed-vocabulary EAGLE drafter from trace JSON."
    )
    parser.add_argument("--trace", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=299792458)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--draft-width", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    return parser.parse_args()


def load_trace(path: Path) -> tuple[list[int], list[int], list[TraceStep]]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("kind") != "lmbrrr_hidden_state_trace":
        raise ValueError(f"{path} is not an lmbrrr trace report")

    capture_layers = list(report["capture_layers"])
    prompt_token_ids = [int(token) for token in report["prompt_token_ids"]]
    steps: list[TraceStep] = []
    for step in report["steps"]:
        if step.get("eos"):
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
        steps.append(
            TraceStep(
                trace=str(path),
                prompt=report["prompt"],
                step=int(step["step"]),
                target_token_id=int(step["target_token_id"]),
                target_token=step.get("target_token", ""),
                feature=feature,
            )
        )
    return capture_layers, prompt_token_ids, steps


def load_samples(paths: list[Path], draft_width: int) -> tuple[list[int], list[Sample]]:
    all_samples: list[Sample] = []
    expected_layers: list[int] | None = None
    input_dim: int | None = None
    for path in paths:
        layers, prompt_token_ids, steps = load_trace(path)
        if expected_layers is None:
            expected_layers = layers
        elif layers != expected_layers:
            raise ValueError(f"{path} captured layers {layers}, expected {expected_layers}")
        if not prompt_token_ids:
            raise ValueError(f"{path} has no prompt_token_ids")
        generated = [step.target_token_id for step in steps]
        for anchor_idx, anchor in enumerate(steps):
            if input_dim is None:
                input_dim = len(anchor.feature)
            elif len(anchor.feature) != input_dim:
                raise ValueError(
                    f"{anchor.trace} step {anchor.step} feature dim {len(anchor.feature)}, "
                    f"expected {input_dim}"
                )
            for draft_position in range(draft_width):
                target_idx = anchor_idx + draft_position
                if target_idx >= len(steps):
                    break
                if draft_position == 0:
                    prev_token_id = (
                        generated[anchor_idx - 1] if anchor_idx > 0 else prompt_token_ids[-1]
                    )
                else:
                    prev_token_id = generated[target_idx - 1]
                target = steps[target_idx]
                all_samples.append(
                    Sample(
                        trace=anchor.trace,
                        prompt=anchor.prompt,
                        anchor_step=anchor.step,
                        draft_position=draft_position,
                        prev_token_id=prev_token_id,
                        target_token_id=target.target_token_id,
                        target_token=target.target_token,
                        feature=anchor.feature,
                    )
                )

    if expected_layers is None or input_dim is None or not all_samples:
        raise ValueError("no recurrent drafter samples found")
    return expected_layers, all_samples


def split_indices(sample_count: int, eval_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(sample_count))
    random.Random(seed).shuffle(indices)
    if sample_count < 8 or eval_fraction <= 0:
        return indices, []
    eval_count = max(1, min(sample_count - 1, round(sample_count * eval_fraction)))
    return indices[eval_count:], indices[:eval_count]


def vocab(values: list[int]) -> tuple[list[int], dict[int, int]]:
    token_ids = sorted(set(values))
    return token_ids, {token_id: idx for idx, token_id in enumerate(token_ids)}


def build_inputs(
    samples: list[Sample],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    prev_token_by_id: dict[int, int],
    max_draft_width: int,
) -> torch.Tensor:
    feature_dim = feature_mean.shape[0]
    prev_vocab_dim = len(prev_token_by_id)
    rows: list[torch.Tensor] = []
    for sample in samples:
        feature = torch.tensor(sample.feature, dtype=torch.float32)
        feature = (feature - feature_mean) / feature_std
        prev = torch.zeros(prev_vocab_dim, dtype=torch.float32)
        prev[prev_token_by_id[sample.prev_token_id]] = 1.0
        position = torch.tensor(
            [sample.draft_position / max(1, max_draft_width - 1)], dtype=torch.float32
        )
        row = torch.cat([feature, prev, position])
        assert row.shape[0] == feature_dim + prev_vocab_dim + 1
        rows.append(row)
    return torch.stack(rows)


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
            "mean_anchor_acceptance": None,
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
    mean_acceptance = accepted_prefix_estimate(predictions, labels, indices, samples, draft_width)
    return {
        "samples": len(indices),
        "loss": loss,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "mean_anchor_acceptance": mean_acceptance,
    }


def accepted_prefix_estimate(
    predictions: dict[int, int],
    labels: torch.Tensor,
    indices: list[int],
    samples: list[Sample],
    draft_width: int,
) -> float | None:
    available = set(indices)
    by_anchor: dict[tuple[str, int], dict[int, int]] = {}
    for idx, sample in enumerate(samples):
        by_anchor.setdefault((sample.trace, sample.anchor_step), {})[sample.draft_position] = idx
    accepted_lengths: list[int] = []
    for idx in indices:
        sample = samples[idx]
        if sample.draft_position != 0:
            continue
        accepted = 0
        positions = by_anchor[(sample.trace, sample.anchor_step)]
        for draft_position in range(draft_width):
            next_idx = positions.get(draft_position)
            if next_idx is None or next_idx not in available:
                break
            if predictions.get(next_idx) != int(labels[next_idx].item()):
                break
            accepted += 1
        accepted_lengths.append(accepted)
    if not accepted_lengths:
        return None
    return sum(accepted_lengths) / len(accepted_lengths)


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    capture_layers, samples = load_samples(args.trace, args.draft_width)
    target_token_ids, target_label_by_id = vocab([sample.target_token_id for sample in samples])
    prev_token_ids, prev_token_by_id = vocab([sample.prev_token_id for sample in samples])
    raw_features = torch.tensor([sample.feature for sample in samples], dtype=torch.float32)
    feature_mean = raw_features.mean(dim=0)
    feature_std = raw_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    x = build_inputs(samples, feature_mean, feature_std, prev_token_by_id, args.draft_width)
    y = torch.tensor(
        [target_label_by_id[sample.target_token_id] for sample in samples], dtype=torch.long
    )

    train_indices, eval_indices = split_indices(len(samples), args.eval_fraction, args.seed)
    device = torch.device(args.device)
    model = RecurrentDrafter(x.shape[1], args.hidden_dim, len(target_token_ids)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_x = x[train_indices].to(device)
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
        all_logits = model_cpu(x)
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
        "kind": "lmbrrr_eagle_recurrent_drafter",
        "schema_version": 1,
        "draft_head_type": "observed-vocabulary-recurrent-mlp",
        "activation": "gelu_tanh",
        "weights": weights_path.name,
        "capture_layers": capture_layers,
        "feature_dim": int(raw_features.shape[1]),
        "input_dim": int(x.shape[1]),
        "hidden_dim": args.hidden_dim,
        "output_dim": len(target_token_ids),
        "target_token_ids": target_token_ids,
        "prev_token_ids": prev_token_ids,
        "max_draft_width": args.draft_width,
        "feature_normalization": "zscore",
        "recurrent_state": "anchor_feature + previous_draft_token_one_hot + normalized_draft_position",
        "dataset": {
            "traces": [str(path) for path in args.trace],
            "samples": len(samples),
            "train_samples": len(train_indices),
            "eval_samples": len(eval_indices),
            "unique_target_tokens": len(target_token_ids),
            "unique_prev_tokens": len(prev_token_ids),
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
        "limits": [
            "The output vocabulary is restricted to target token ids observed in the trace set.",
            "Previous-token state is restricted to previous token ids observed in the trace set.",
            "This smoke drafter proposes a single block from one anchor feature before target verification.",
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
