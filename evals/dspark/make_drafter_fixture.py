#!/usr/bin/env python3
"""Generate the drafter-parity fixture from the trained DSpark checkpoint.

Runs one deterministic drafter forward (raw capture-concat context -> backbone
-> base logits -> greedy Markov sampling -> confidence) in the pinned training
environment and saves inputs + every intermediate the Candle port must
reproduce. This is the blocking oracle for the Rust drafter loader.
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deepspec-path", default="/deepspec")
    parser.add_argument("--ctx-len", type=int, default=12)
    parser.add_argument("--anchor-token", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, args.deepspec_path)

    import torch
    from safetensors.torch import save_file

    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel

    torch.manual_seed(args.seed)
    model = (
        Qwen3DSparkModel.from_pretrained(
            args.checkpoint,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .eval()
    )
    cfg = model.config
    gamma = int(cfg.block_size)
    ctx_len = int(args.ctx_len)
    num_capture = len(cfg.target_layer_ids)

    # Raw capture concat, scaled to a realistic hidden magnitude.
    target_hidden_states = (
        torch.randn(1, ctx_len, num_capture * cfg.hidden_size, dtype=torch.float32)
        * 2.0
    ).to(torch.bfloat16)
    draft_input_ids = torch.full((1, gamma), int(cfg.mask_token_id), dtype=torch.long)
    draft_input_ids[0, 0] = int(args.anchor_token)
    # Round-1 inference semantics: ctx occupies positions 0..ctx_len-1, the
    # anchor sits at position ctx_len, drafts continue after it.
    position_ids = torch.arange(ctx_len + gamma, dtype=torch.long).unsqueeze(0)

    with torch.inference_mode():
        block_hidden = model._forward_backbone(
            position_ids=position_ids,
            noise_embedding=model.embed_tokens(draft_input_ids),
            target_hidden_states=target_hidden_states,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )
        base_logits = model.compute_logits(block_hidden)
        sampled, corrected_logits = model.sample_draft_tokens(
            base_logits,
            first_prev_token_ids=draft_input_ids[:, 0],
            temperature=0.0,
            hidden_states=block_hidden,
        )
        prev_token_ids = torch.cat([draft_input_ids[:, :1], sampled[:, :-1]], dim=1)
        confidence_logits = model.predict_confidence_step(
            block_hidden,
            prev_token_ids=prev_token_ids,
        )

    save_file(
        {
            "target_hidden_states": target_hidden_states.contiguous(),
            "draft_input_ids": draft_input_ids.to(torch.int32).contiguous(),
            "position_ids": position_ids.to(torch.int32).contiguous(),
            "block_hidden": block_hidden.to(torch.float32).contiguous(),
            "base_logits": base_logits.to(torch.float32).contiguous(),
            "corrected_logits": corrected_logits.to(torch.float32).contiguous(),
            "sampled_tokens": sampled.to(torch.int32).contiguous(),
            "confidence_logits": confidence_logits.to(torch.float32).contiguous(),
        },
        args.output,
        metadata={
            "checkpoint": args.checkpoint,
            "ctx_len": str(ctx_len),
            "gamma": str(gamma),
            "anchor_token": str(int(args.anchor_token)),
            "seed": str(int(args.seed)),
        },
    )
    print(
        f"fixture written to {args.output}: ctx_len={ctx_len} gamma={gamma} "
        f"sampled={sampled.tolist()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
