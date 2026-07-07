# MiniCPM-V-4.6 MTP Weight Audit

Date: 2026-07-07

Ticket: `audit-minicpm-mtp-weights`

## Conclusion

MiniCPM-V-4.6's text config advertises a one-layer MTP setting, but the local
checkpoint does not ship any matching MTP or draft-head tensors.

For this checkpoint, there is no built-in one-token MTP drafter to wire into the
Candle runner. The speculative decoding path should proceed through the verifier
harness and replay drafter first, then move to hidden-state tracing and an
EAGLE-style trained drafter.

## Evidence

The vendored config identifies the text backbone as Qwen3.5 text and exposes the
MTP-related fields:

```sh
jq '{architectures, model_type, transformers_version, text_model_type:.text_config.model_type, mtp_num_hidden_layers:.text_config.mtp_num_hidden_layers, mtp_use_dedicated_embeddings:.text_config.mtp_use_dedicated_embeddings}' docs/research/models/minicpm-v-4.6/hf-model/config.json
```

Observed values:

```json
{
  "architectures": ["MiniCPMV4_6ForConditionalGeneration"],
  "model_type": "minicpmv4_6",
  "transformers_version": "5.7.0",
  "text_model_type": "qwen3_5_text",
  "mtp_num_hidden_layers": 1,
  "mtp_use_dedicated_embeddings": false
}
```

The vendored safetensors header has 779 tensor keys. Searching those keys for
MTP or draft-like modules returns no matches:

```sh
jq -r 'keys[] | select(test("(^|\\.)mtp|draft|speculative|eagle|multi"; "i"))' docs/research/models/minicpm-v-4.6/hf-model/model-safetensors-header.json
```

The checkpoint key families are only the language model, vision tower, and
MiniCPM merger modules:

```text
432 model.vision_tower.encoder
318 model.language_model.layers
 16 model.vision_tower.vit_merger
  6 model.merger.mlp
  3 model.vision_tower.embeddings
  2 model.vision_tower.post_layernorm
  1 model.language_model.norm
  1 model.language_model.embed_tokens
```

The vendored Qwen3.5 Transformers source does contain MTP-specific load
tolerance:

- `docs/research/models/qwen3.5/transformers/modeling_qwen3_5.py` ignores
  unexpected `^mtp.*` keys on the base pretrained class.
- `docs/research/models/qwen3.5/transformers/modeling_qwen3_5.py` and
  `docs/research/models/qwen3.5/transformers/modular_qwen3_5.py` ignore
  unexpected `^mtp.*` keys on the causal LM / conditional generation classes.

That points to architecture-family support for checkpoints that may include MTP
weights, not evidence that this MiniCPM-V-4.6 checkpoint includes them.

## Runner Impact

Do not add an MTP loader path to `src/weights.rs` for MiniCPM-V-4.6 unless a
future checkpoint actually includes `mtp.*` tensors. The current strict weight
validation should stay focused on the tensors present in this checkpoint.

For speculative decoding, the next useful runtime work is:

1. Implement the greedy verifier harness against the target model.
2. Add a replay drafter that proposes known baseline tokens and proves the
   verifier/output-reconstruction loop.
3. Add hidden-state trace recording for a trainable EAGLE-style chain drafter.

This keeps the experiment grounded in measurable verifier behavior without
pretending that the current checkpoint has an already-trained draft head.
