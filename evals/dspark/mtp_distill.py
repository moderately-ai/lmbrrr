#!/usr/bin/env python3
"""Distill the Qwen3.5-0.8B vendor MTP head onto the MiniCPM-V-4.6 target.

The vendor head (15 mtp.* tensors: fc + three norms + one full-attention
decoder layer, embeddings/lm_head shared with the target) was trained for the
BASE Qwen3.5-0.8B; our tower is a VLM finetune of it, and the mismatch costs
draft acceptance (measured 54% position-1 on the M3 vs 80-100% family numbers
on matched towers). This job aligns ONLY the head: initialize from the vendor
weights and train on (target final-norm hidden at p, token p+1) -> token p+2
over target-generated conversations (the regen corpus IS the target's own
distribution), teacher-forced, cross-entropy.

Design notes:
- batch size 1 + gradient accumulation: no padding, no masks, no admission
  logic — the teacher is 0.8B on an H100, throughput is a non-issue.
- teacher hiddens come from the text model's last_hidden_state (post final
  norm — verified in-job by asserting lm_head(hidden) reproduces argmax of
  the model's own logits on the first sample).
- loss over ALL positions (pilot simplification; assistant-span masking is
  iteration 2 if acceptance disappoints).
- held-out next-token top-1 accuracy is the position-1 acceptance proxy,
  reported for the vendor init (baseline) and at every eval.
- HF's Qwen3_5RMSNorm applies the zero-centred (1 + w) convention itself, so
  vendor tensors load verbatim and the saved checkpoint keeps raw weights —
  lmbrrr's loader applies its own shift exactly as it does for the vendor
  file (drop-in replacement for --drafter-mtp).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="target model (fakequant dir)")
    p.add_argument("--mtp-base", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--input", required=True, help="regen-corpus JSONL")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=40000)
    p.add_argument("--holdout", type=int, default=500)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--min-tokens", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


class MtpHead(nn.Module):
    """Mirror of the checkpoint's mtp.* module using HF's own building
    blocks so semantics match the vendor training exactly."""

    def __init__(self, text_config):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
            Qwen3_5TextRotaryEmbedding,
        )

        h = text_config.hidden_size
        # A layer_idx whose block type is full attention (the MTP block).
        full_idx = text_config.layer_types.index("full_attention")
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, eps=text_config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, eps=text_config.rms_norm_eps)
        self.norm = Qwen3_5RMSNorm(h, eps=text_config.rms_norm_eps)
        self.layer = Qwen3_5DecoderLayer(text_config, layer_idx=full_idx)
        self.rotary = Qwen3_5TextRotaryEmbedding(config=text_config)

    def load_vendor(self, base: str) -> None:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download(base, "model.safetensors-00001-of-00001.safetensors")
        raw = load_file(path)
        remap = {}
        for k, v in raw.items():
            if not k.startswith("mtp."):
                continue
            name = k[len("mtp.") :].replace("layers.0.", "layer.")
            remap[name] = v
        missing, unexpected = self.load_state_dict(remap, strict=False)
        # rotary buffers are non-persistent; everything else must match.
        real_missing = [m for m in missing if not m.startswith("rotary.")]
        assert not real_missing, f"missing vendor tensors: {real_missing}"
        assert not unexpected, f"unexpected vendor tensors: {unexpected}"
        assert len(remap) == 15, f"expected 15 mtp tensors, got {len(remap)}"

    def forward(self, hidden: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        # hidden: [1, S, H] teacher post-norm hiddens at positions p..p+S-1;
        # embeds: [1, S, H] embeddings of tokens p+1..p+S. Returns post-norm
        # hidden whose row k predicts token p+k+2 through the shared lm_head.
        x = self.fc(
            torch.cat(
                [self.pre_fc_norm_embedding(embeds), self.pre_fc_norm_hidden(hidden)],
                dim=-1,
            )
        )
        b, s, _ = x.shape
        position_ids = (
            torch.arange(s, device=x.device).view(1, 1, -1).expand(4, b, -1)
        )
        pos_embeddings = self.rotary(x, position_ids[1:])
        x = self.layer(
            x,
            position_embeddings=pos_embeddings,
            attention_mask=None,
            position_ids=position_ids[0],
        )
        return self.norm(x)

    def export_vendor_names(self) -> dict:
        out = {}
        for k, v in self.state_dict().items():
            if k.startswith("rotary."):
                continue
            name = "mtp." + k.replace("layer.", "layers.0.")
            out[name] = v.detach().to(torch.bfloat16).contiguous().cpu()
        assert len(out) == 15, f"export expected 15 tensors, got {len(out)}"
        return out


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    teacher = (
        AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    for p in teacher.parameters():
        p.requires_grad_(False)
    text_model = teacher.model.language_model
    lm_head = teacher.lm_head
    embed = text_model.embed_tokens

    base_cfg = AutoConfig.from_pretrained(args.mtp_base)
    text_cfg = getattr(base_cfg, "text_config", base_cfg)
    text_cfg._attn_implementation = "sdpa"
    head = MtpHead(text_cfg).to(device=device, dtype=torch.bfloat16)
    head.load_vendor(args.mtp_base)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"mtp head: {n_params/1e6:.1f}M params, vendor init loaded", flush=True)

    # Corpus: templated conversations, tokenized once, filtered by length.
    rows = []
    with open(args.input, "r", encoding="utf-8") as fh:
        for line in fh:
            if len(rows) >= args.num_samples + args.holdout:
                break
            row = json.loads(line)
            conv = row.get("conversations", [])
            if not conv:
                continue
            rows.append(conv)
    print(f"loaded {len(rows)} conversations", flush=True)

    def encode(conv) -> torch.Tensor | None:
        text = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < args.min_tokens:
            return None
        return torch.tensor(ids[: args.max_tokens], device=device).unsqueeze(0)

    holdout = rows[: args.holdout]
    train_rows = rows[args.holdout :]

    def teacher_hidden(ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = text_model(input_ids=ids)
            return out.last_hidden_state

    # Verify the post-norm identity once: lm_head(last_hidden) must reproduce
    # the model's own next-token argmax.
    probe = encode(train_rows[0])
    assert probe is not None
    with torch.no_grad():
        h = teacher_hidden(probe)
        via_head = lm_head(h).argmax(-1)
        full = teacher.model(input_ids=probe)
        direct = lm_head(full.last_hidden_state).argmax(-1)
        assert torch.equal(via_head, direct), "post-norm identity check failed"
    print("teacher post-norm identity verified", flush=True)

    def head_loss_and_acc(ids: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        # pairs: (h_p, t_{p+1}) -> t_{p+2} for p in 0..T-3.
        h = teacher_hidden(ids)  # [1, T, H]
        t = ids.shape[1]
        if t < 3:
            zero = torch.zeros((), device=device, requires_grad=True)
            return zero, 0, 0
        hidden = h[:, : t - 1]
        succ = ids[:, 1:t]
        with torch.no_grad():
            embeds = embed(succ)
        post = head(hidden, embeds)  # [1, T-1, H]
        logits = lm_head(post[:, : t - 2])
        labels = ids[:, 2:t]
        loss = nn.functional.cross_entropy(
            logits.float().squeeze(0), labels.squeeze(0)
        )
        correct = int((logits.argmax(-1) == labels).sum())
        return loss, correct, t - 2

    @torch.no_grad()
    def evaluate() -> float:
        head.eval()
        correct = total = 0
        for conv in holdout:
            ids = encode(conv)
            if ids is None:
                continue
            _, c, n = head_loss_and_acc(ids)
            correct += c
            total += n
        head.train()
        return correct / max(total, 1)

    base_acc = evaluate()
    print(f"VENDOR BASELINE holdout next-token top-1: {base_acc:.4f}", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, args.epochs * len(train_rows) // args.grad_accum)

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    os.makedirs(args.output_dir, exist_ok=True)
    from safetensors.torch import save_file

    best_acc = base_acc
    step = 0
    micro = 0
    running = 0.0
    start = time.monotonic()
    head.train()
    for epoch in range(args.epochs):
        for row_idx, conv in enumerate(train_rows):
            ids = encode(conv)
            if ids is None:
                continue
            loss, _, n = head_loss_and_acc(ids)
            if n == 0:
                continue
            (loss / args.grad_accum).backward()
            running += float(loss)
            micro += 1
            if micro % args.grad_accum == 0:
                step += 1
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if step % 50 == 0:
                    print(
                        f"epoch {epoch} step {step}/{total_steps} "
                        f"loss {running / (50 * args.grad_accum):.4f} "
                        f"lr {lr_at(step):.2e} "
                        f"({(time.monotonic() - start)/60:.1f} min)",
                        flush=True,
                    )
                    running = 0.0
                if step % args.eval_every == 0:
                    acc = evaluate()
                    print(f"EVAL step {step}: holdout top-1 {acc:.4f}", flush=True)
                    if acc > best_acc:
                        best_acc = acc
                        save_file(
                            head.export_vendor_names(),
                            os.path.join(args.output_dir, "mtp.safetensors"),
                        )
                        print(f"saved best ({acc:.4f})", flush=True)

    final_acc = evaluate()
    print(f"FINAL holdout top-1: {final_acc:.4f} (vendor {base_acc:.4f}, best {best_acc:.4f})", flush=True)
    if final_acc > best_acc:
        best_acc = final_acc
        save_file(
            head.export_vendor_names(),
            os.path.join(args.output_dir, "mtp.safetensors"),
        )
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fh:
        json.dump(
            {
                "vendor_baseline_top1": base_acc,
                "best_top1": best_acc,
                "final_top1": final_acc,
                "num_train": len(train_rows),
                "epochs": args.epochs,
                "lr": args.lr,
                "max_tokens": args.max_tokens,
                "loss_positions": "all (assistant-span masking = iteration 2)",
            },
            fh,
            indent=1,
        )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
