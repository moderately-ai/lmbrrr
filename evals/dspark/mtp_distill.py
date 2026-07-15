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
- length-bucketed batching (r2): sequences are right-padded within a batch;
  causal attention means padded tails cannot influence real positions, and
  their losses are masked. Per-sequence mean loss + effective batch of 16
  sequences preserve the r1 (batch-1, grad-accum-16) training dynamics.
- teacher hiddens come from the text model's last_hidden_state (post final
  norm — verified in-job by asserting lm_head(hidden) reproduces argmax of
  the model's own logits on the first sample).
- loss over ALL positions by default (bit-identical to r1-r4); --span-mask
  restricts it to assistant-content tokens (the only tokens the drafter ever
  proposes), located by ChatML marker + char offsets since the template
  carries no {% generation %} markers.
- held-out next-token top-1 accuracy is the position-1 acceptance proxy,
  reported for the init (baseline) and at every eval; under --span-mask it is
  measured over assistant positions only (a tighter proxy, but not comparable
  to the all-position r1-r4 baselines).
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
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--bucket-window", type=int, default=512,
                   help="shuffle window sorted by length before batching")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--init-from", default=None,
                   help="warm-start from a prior run's mtp.safetensors "
                        "(vendor tensor names) instead of the HF vendor head")
    p.add_argument("--qat-q4k", action="store_true",
                   help="straight-through q4_K fake-quant on every head "
                        "linear (QAT); the exported checkpoint keeps the "
                        "full-precision master weights")
    p.add_argument("--span-mask", action="store_true",
                   help="restrict the loss (and the holdout top-1 proxy) to "
                        "assistant-content tokens — the only tokens the "
                        "drafter ever proposes. Off = loss over all positions "
                        "(bit-identical to r1-r4).")
    return p.parse_args()


def fake_quant_q4k(w: torch.Tensor) -> torch.Tensor:
    """Approximate GGML q4_K applied to a [out, in] weight: superblocks of
    256 along the input dim, 8 sub-blocks of 32 quantized to 4-bit against
    6-bit sub-block scale/min pairs and fp16 superblock scales. Min-max grid
    per sub-block — the reference quantizer's search refines this slightly,
    but the noise structure (grid spacing, scale quantization, fp16 supers)
    is what robustness training needs to see."""
    out_dim, in_dim = w.shape
    assert in_dim % 256 == 0, f"q4_K needs in_dim % 256 == 0, got {in_dim}"
    v = w.float().reshape(out_dim, in_dim // 256, 8, 32)
    vmax = v.amax(-1, keepdim=True)
    vmin = v.amin(-1, keepdim=True).clamp(max=0)  # ggml mins are >= 0
    s = (vmax - vmin) / 15.0
    m = -vmin
    d = (s.amax(2, keepdim=True) / 63.0).half().float()  # fp16 super scales
    dm = (m.amax(2, keepdim=True) / 63.0).half().float()
    sc = torch.where(d > 0, (s / d).round().clamp(0, 63), torch.zeros_like(s))
    mq = torch.where(dm > 0, (m / dm).round().clamp(0, 63), torch.zeros_like(m))
    scale = d * sc
    minv = dm * mq
    q = torch.where(
        scale > 0, ((v + minv) / scale).round().clamp(0, 15), torch.zeros_like(v)
    )
    wq = (q * scale - minv).reshape(out_dim, in_dim)
    return wq.to(w.dtype)


def patch_qat_q4k(head: nn.Module) -> int:
    """Route every head linear through straight-through fake-quant: forward
    sees the quantized grid, backward passes gradients to the bf16 masters.
    Mirrors lmbrrr's --mtp-quantize scope (all unbiased head linears; norms
    and the shared embed/lm_head are untouched)."""
    import types

    def qat_forward(self, x):
        w = self.weight
        wq = w + (fake_quant_q4k(w) - w).detach()
        return nn.functional.linear(x, wq, self.bias)

    n = 0
    for mod in head.modules():
        if isinstance(mod, nn.Linear):
            mod.forward = types.MethodType(qat_forward, mod)
            n += 1
    return n


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

        path = hf_hub_download(base, "model.safetensors-00001-of-00001.safetensors")
        self._load_mtp_file(path)

    def load_checkpoint(self, path: str) -> None:
        """Warm-start from a prior run's export (vendor tensor names)."""
        self._load_mtp_file(path)

    def _load_mtp_file(self, path: str) -> None:
        from safetensors.torch import load_file

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
    if args.init_from:
        head.load_checkpoint(args.init_from)
        init_desc = f"warm start {args.init_from}"
    else:
        head.load_vendor(args.mtp_base)
        init_desc = "vendor init"
    n_params = sum(p.numel() for p in head.parameters())
    print(f"mtp head: {n_params/1e6:.1f}M params, {init_desc} loaded", flush=True)
    if args.qat_q4k:
        n_patched = patch_qat_q4k(head)
        # Eval runs through the same patched forwards, so every holdout
        # number (including the baseline) measures the QUANTIZED head — the
        # baseline is the init checkpoint's q4k acceptance-proxy drop.
        print(f"QAT: q4_K straight-through on {n_patched} linears", flush=True)

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

    # Assistant-content spans in the ChatML-templated text run from just after
    # "<|im_start|>assistant\n" to the next "<|im_end|>". The template carries
    # no {% generation %} markers (checked), so return_assistant_tokens_mask is
    # unavailable — locate spans by literal marker and map via char offsets.
    IM_ASSIST = "<|im_start|>assistant\n"
    IM_END = "<|im_end|>"

    def _assistant_spans(text: str) -> list[tuple[int, int]]:
        spans = []
        pos = 0
        while True:
            i = text.find(IM_ASSIST, pos)
            if i < 0:
                break
            s = i + len(IM_ASSIST)
            j = text.find(IM_END, s)
            e = j if j >= 0 else len(text)
            spans.append((s, e))
            pos = e + len(IM_END) if j >= 0 else len(text)
        return spans

    if args.span_mask:
        assert tokenizer.is_fast, "--span-mask needs a fast tokenizer (char offsets)"

    def encode_ids(conv) -> tuple[list[int], list[bool]] | None:
        # (ids, assist): assist[k] marks token k as inside an assistant span.
        # --span-mask off -> assist is all-True (loss over every position, the
        # original r1-r4 behaviour), so the downstream mask is a no-op.
        text = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        if not args.span_mask:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) < args.min_tokens:
                return None
            ids = ids[: args.max_tokens]
            return ids, [True] * len(ids)
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        if len(ids) < args.min_tokens:
            return None
        spans = _assistant_spans(text)
        assist = [any(s <= a < e for s, e in spans) for a, _ in enc["offset_mapping"]]
        return ids[: args.max_tokens], assist[: args.max_tokens]

    def encode(conv) -> torch.Tensor | None:
        enc = encode_ids(conv)
        if enc is None:
            return None
        return torch.tensor(enc[0], device=device).unsqueeze(0)

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

    def head_acc(ids_list: list[int], assist_list: list[bool]) -> tuple[int, int]:
        # pairs: (h_p, t_{p+1}) -> t_{p+2} for p in 0..T-3. Top-1 counted only
        # where the PREDICTED token t_{p+2} is an assistant token (assist[p+2]).
        t = len(ids_list)
        if t < 3:
            return 0, 0
        ids = torch.tensor(ids_list, device=device).unsqueeze(0)
        h = teacher_hidden(ids)  # [1, T, H]
        hidden = h[:, : t - 1]
        succ = ids[:, 1:t]
        with torch.no_grad():
            embeds = embed(succ)
        post = head(hidden, embeds)  # [1, T-1, H]
        logits = lm_head(post[:, : t - 2])
        labels = ids[:, 2:t].squeeze(0)
        assist = torch.tensor(assist_list, device=device, dtype=torch.bool)[2:t]
        hit = (logits.argmax(-1).squeeze(0) == labels) & assist
        return int(hit.sum()), int(assist.sum())

    @torch.no_grad()
    def evaluate() -> float:
        head.eval()
        correct = total = 0
        for conv in holdout:
            enc = encode_ids(conv)
            if enc is None:
                continue
            ids_list, assist_list = enc
            if len(ids_list) < 3 or sum(assist_list[2:]) == 0:
                continue
            c, n = head_acc(ids_list, assist_list)
            correct += c
            total += n
        head.train()
        return correct / max(total, 1)

    base_acc = evaluate()
    base_label = "INIT BASELINE" if args.init_from else "VENDOR BASELINE"
    print(f"{base_label} holdout next-token top-1: {base_acc:.4f}", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = 1  # recomputed from the batch count below, before training

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    os.makedirs(args.output_dir, exist_ok=True)
    from safetensors.torch import save_file

    # Length-bucketed batches: tokenize once, shuffle, sort inside a window
    # (keeps randomness, minimizes padding), batch, shuffle batch order.
    import random

    rng = random.Random(args.seed)
    encoded = []
    n_drop_noassist = 0
    for conv in train_rows:
        enc = encode_ids(conv)
        if enc is None:
            continue
        ids, assist = enc
        if len(ids) < 3:
            continue
        if sum(assist[2:]) == 0:  # nothing to learn (assistant truncated out)
            n_drop_noassist += 1
            continue
        encoded.append((ids, assist))
    drop_note = f" (dropped {n_drop_noassist} with no assistant label)" if args.span_mask else ""
    print(f"encoded {len(encoded)} train sequences{drop_note}", flush=True)
    if args.span_mask and encoded:
        # Correctness gate: a broken marker/offset map collapses the mask to
        # all-off or all-on. Assistant spans should be a clear majority-ish of
        # the regen corpus (user prompts are the minority).
        na = sum(sum(a) for _, a in encoded)
        nt = sum(len(a) for _, a in encoded)
        frac = na / max(nt, 1)
        print(f"assistant-span fraction: {frac:.3f} ({na}/{nt} tokens)", flush=True)
        assert 0.1 <= frac <= 0.98, f"span mask looks broken: assistant frac {frac:.3f}"
    rng.shuffle(encoded)
    batches = []
    for w0 in range(0, len(encoded), args.bucket_window):
        window = sorted(encoded[w0 : w0 + args.bucket_window], key=lambda x: len(x[0]))
        for b0 in range(0, len(window), args.batch_size):
            batches.append(window[b0 : b0 + args.batch_size])
    rng.shuffle(batches)
    pad_waste = 1.0 - (
        sum(len(s[0]) for b in batches for s in b)
        / max(1, sum(len(b) * len(b[-1][0]) for b in batches))
    )
    print(f"{len(batches)} batches of <= {args.batch_size}, padding waste {pad_waste:.1%}", flush=True)

    def batch_loss_and_acc(batch: list[tuple[list[int], list[bool]]]):
        # Right padding: causal attention keeps real positions independent of
        # pads; padded label positions are ignored (-100). Per-sequence mean
        # loss preserves the batch-1 weighting. valid = real position AND (with
        # --span-mask) the predicted token is assistant content.
        bsz = len(batch)
        t = max(len(s) for s, _ in batch)
        ids = torch.zeros((bsz, t), dtype=torch.long, device=device)
        mask = torch.zeros((bsz, t), dtype=torch.long, device=device)
        amask = torch.zeros((bsz, t), dtype=torch.bool, device=device)
        for i, (s, a) in enumerate(batch):
            ids[i, : len(s)] = torch.tensor(s, device=device)
            mask[i, : len(s)] = 1
            amask[i, : len(a)] = torch.tensor(a, device=device, dtype=torch.bool)
        with torch.no_grad():
            h = text_model(input_ids=ids, attention_mask=mask).last_hidden_state
            embeds = embed(ids[:, 1:t])
        post = head(h[:, : t - 1], embeds)
        logits = lm_head(post[:, : t - 2])
        labels = ids[:, 2:t].clone()
        valid = mask[:, 2:t].bool() & amask[:, 2:t]
        labels[~valid] = -100
        raw = nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(bsz, t - 2)
        per_seq = (raw * valid).sum(-1) / valid.sum(-1).clamp(min=1)
        loss = per_seq.mean()
        correct = int(((logits.argmax(-1) == labels) & valid).sum())
        return loss, correct, int(valid.sum())

    best_acc = base_acc
    step = 0
    micro = 0
    running = 0.0
    seen_tokens = 0
    start = time.monotonic()
    head.train()
    total_steps = max(1, args.epochs * len(batches) // args.grad_accum)
    for epoch in range(args.epochs):
        for batch in batches:
            loss, _, n = batch_loss_and_acc(batch)
            if n == 0:
                continue
            (loss / args.grad_accum).backward()
            running += float(loss)
            seen_tokens += sum(len(s) for s, _ in batch)
            micro += 1
            if micro % args.grad_accum == 0:
                step += 1
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if step % 50 == 0:
                    elapsed = time.monotonic() - start
                    print(
                        f"epoch {epoch} step {step}/{total_steps} "
                        f"loss {running / (50 * args.grad_accum):.4f} "
                        f"lr {lr_at(step):.2e} "
                        f"({elapsed/60:.1f} min, {seen_tokens/max(elapsed,1e-9):.0f} tok/s)",
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
                "loss_positions": "assistant-span" if args.span_mask else "all",
                "span_mask": args.span_mask,
                "qat_q4k": args.qat_q4k,
                "init_from": args.init_from,
            },
            fh,
            indent=1,
        )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
