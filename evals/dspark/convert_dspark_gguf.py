"""Convert a DeepSpec Qwen3DSparkModel checkpoint (safetensors + config.json)
into the `dspark.*`-namespaced GGUF that lmbrrr's DsparkDrafter::load_gguf
reads. The shipped prism drafter came as GGUF; our in-house rounds train
safetensors, so this is the deployment bridge.

Tensor-name map (source -> GGUF), verified against src/dspark.rs:
    embed_tokens.weight                 -> token_embd.weight
    lm_head.weight                      -> output.weight
    norm.weight                         -> output_norm.weight
    fc.weight                           -> dspark.fc.weight
    hidden_norm.weight                  -> dspark.hidden_norm.weight
    markov_head.markov_w1.weight        -> dspark.markov_head_a.weight
    markov_head.markov_w2.weight        -> dspark.markov_head_b.weight
    confidence_head.proj.weight/bias    -> dspark.confidence_head.weight/bias
    layers.N.self_attn.q_proj.weight    -> blk.N.attn_q.weight
             k_proj / v_proj / o_proj   -> attn_k / attn_v / attn_output
             q_norm / k_norm            -> attn_q_norm / attn_k_norm
    layers.N.mlp.gate_proj/up_proj/down -> blk.N.ffn_gate/ffn_up/ffn_down
    layers.N.input_layernorm            -> blk.N.attn_norm
    layers.N.post_attention_layernorm   -> blk.N.ffn_norm

Writes bf16 (weights are bf16 in the checkpoint); run `lmbrrr gguf requant`
afterward for a Q8_0/Q4_1 deployment drafter. No tokenizer metadata — the
drafter GGUF carries none; the target GGUF provides the tokenizer.
"""

from __future__ import annotations

import argparse
import json
import os

import gguf
import numpy as np
from safetensors import safe_open


def _bf16_to_np(slice_) -> np.ndarray:
    # gguf writes bf16 from a uint16 view (numpy has no native bfloat16).
    import torch

    t = slice_[:]
    return t.view(torch.uint16).cpu().numpy()


def convert(checkpoint_dir: str, out_path: str) -> None:
    cfg = json.load(open(os.path.join(checkpoint_dir, "config.json")))
    st_path = os.path.join(checkpoint_dir, "model.safetensors")

    n_layers = int(cfg["num_hidden_layers"])
    writer = gguf.GGUFWriter(out_path, "dspark")

    # --- Metadata (keys read by src/dspark.rs DsparkConfig::from_gguf) ---
    writer.add_uint32("dspark.vocab_size", int(cfg["vocab_size"]))
    writer.add_uint32("dspark.embedding_length", int(cfg["hidden_size"]))
    writer.add_uint32("dspark.feed_forward_length", int(cfg["intermediate_size"]))
    writer.add_uint32("dspark.block_count", n_layers)
    writer.add_uint32("dspark.attention.head_count", int(cfg["num_attention_heads"]))
    writer.add_uint32("dspark.attention.head_count_kv", int(cfg["num_key_value_heads"]))
    writer.add_uint32("dspark.attention.key_length", int(cfg["head_dim"]))
    writer.add_float32(
        "dspark.attention.layer_norm_rms_epsilon", float(cfg["rms_norm_eps"])
    )
    writer.add_float32("dspark.rope.freq_base", float(cfg["rope_theta"]))
    writer.add_uint32("dspark.dspark.block_size", int(cfg["block_size"]))
    writer.add_uint32("dspark.dspark.mask_token_id", int(cfg["mask_token_id"]))
    writer.add_uint32("dspark.dspark.markov_rank", int(cfg["markov_rank"]))
    writer.add_array(
        "dspark.dspark.target_layers",
        [int(x) for x in cfg["target_layer_ids"]],
    )
    writer.add_bool(
        "dspark.dspark.confidence_head", bool(cfg.get("enable_confidence_head", True))
    )
    writer.add_bool(
        "dspark.dspark.confidence_head_with_markov",
        bool(cfg.get("confidence_head_with_markov", True)),
    )
    # Our in-house rounds have no GIDD/log-SNR conditioning (prism's head did).
    writer.add_bool("dspark.dspark.log_snr_conditioning", False)

    # --- Tensor name map ---
    top = {
        "embed_tokens.weight": "token_embd.weight",
        "lm_head.weight": "output.weight",
        "norm.weight": "output_norm.weight",
        "fc.weight": "dspark.fc.weight",
        "hidden_norm.weight": "dspark.hidden_norm.weight",
        "markov_head.markov_w1.weight": "dspark.markov_head_a.weight",
        "markov_head.markov_w2.weight": "dspark.markov_head_b.weight",
        "confidence_head.proj.weight": "dspark.confidence_head.weight",
        "confidence_head.proj.bias": "dspark.confidence_head.bias",
    }
    per_layer = {
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
    }

    written = 0
    with safe_open(st_path, framework="pt") as f:
        present = set(f.keys())
        name_map = dict(top)
        for i in range(n_layers):
            for src_sub, dst_sub in per_layer.items():
                name_map[f"layers.{i}.{src_sub}"] = f"blk.{i}.{dst_sub}"

        for src, dst in name_map.items():
            if src not in present:
                raise SystemExit(f"checkpoint is missing expected tensor: {src}")
            arr = _bf16_to_np(f.get_slice(src))
            writer.add_tensor(dst, arr, raw_dtype=gguf.GGMLQuantizationType.BF16)
            written += 1

        extra = present - set(name_map)
        if extra:
            print(f"NOTE: {len(extra)} checkpoint tensors not mapped: {sorted(extra)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {written} tensors -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="dir with model.safetensors + config.json")
    ap.add_argument("--out", required=True, help="output .gguf path")
    args = ap.parse_args()
    convert(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
