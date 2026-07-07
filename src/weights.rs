use std::{collections::HashMap, fs::File, path::PathBuf};

use anyhow::{bail, Context, Result};
use memmap2::MmapOptions;
use safetensors::SafeTensors;

use crate::config::{LayerType, MiniCpmConfig};

#[derive(Clone, Debug)]
pub struct WeightReport {
    pub tensor_count: usize,
    pub has_lm_head: bool,
}

pub fn validate_minicpm_header(files: &[PathBuf], cfg: &MiniCpmConfig) -> Result<WeightReport> {
    let mut shapes = HashMap::<String, Vec<usize>>::new();
    for path in files {
        let file = File::open(path).with_context(|| format!("open weights {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("mmap {}", path.display()))?;
        let safetensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("read safetensors header {}", path.display()))?;
        for name in safetensors.names() {
            let view = safetensors
                .tensor(name)
                .with_context(|| format!("read tensor metadata {name}"))?;
            shapes.insert(name.to_string(), view.shape().to_vec());
        }
    }

    expect(
        &shapes,
        "model.language_model.embed_tokens.weight",
        &[cfg.text_config.vocab_size, cfg.text_config.hidden_size],
    )?;
    expect(
        &shapes,
        "model.language_model.norm.weight",
        &[cfg.text_config.hidden_size],
    )?;
    expect(
        &shapes,
        "model.vision_tower.embeddings.patch_embedding.weight",
        &[
            cfg.vision_config.hidden_size,
            cfg.vision_config.num_channels,
            cfg.vision_config.patch_size,
            cfg.vision_config.patch_size,
        ],
    )?;
    expect(
        &shapes,
        "model.vision_tower.embeddings.position_embedding.weight",
        &[
            (cfg.vision_config.image_size / cfg.vision_config.patch_size).pow(2),
            cfg.vision_config.hidden_size,
        ],
    )?;

    for (idx, layer_type) in cfg.text_config.layer_types.iter().enumerate() {
        let prefix = format!("model.language_model.layers.{idx}");
        expect(
            &shapes,
            &format!("{prefix}.input_layernorm.weight"),
            &[cfg.text_config.hidden_size],
        )?;
        expect(
            &shapes,
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[
                cfg.text_config.intermediate_size,
                cfg.text_config.hidden_size,
            ],
        )?;
        match layer_type {
            LayerType::FullAttention => {
                expect(
                    &shapes,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    &[
                        cfg.text_config.num_attention_heads * cfg.text_config.head_dim * 2,
                        cfg.text_config.hidden_size,
                    ],
                )?;
                expect(
                    &shapes,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    &[
                        cfg.text_config.num_key_value_heads * cfg.text_config.head_dim,
                        cfg.text_config.hidden_size,
                    ],
                )?;
            }
            LayerType::LinearAttention => {
                let conv_dim =
                    cfg.text_config.linear_key_head_dim * cfg.text_config.linear_num_key_heads * 2
                        + cfg.text_config.linear_value_head_dim
                            * cfg.text_config.linear_num_value_heads;
                expect(
                    &shapes,
                    &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    &[conv_dim, cfg.text_config.hidden_size],
                )?;
                expect(
                    &shapes,
                    &format!("{prefix}.linear_attn.conv1d.weight"),
                    &[conv_dim, 1, cfg.text_config.linear_conv_kernel_dim],
                )?;
            }
        }
    }

    Ok(WeightReport {
        tensor_count: shapes.len(),
        has_lm_head: shapes.contains_key("lm_head.weight"),
    })
}

fn expect(shapes: &HashMap<String, Vec<usize>>, name: &str, expected: &[usize]) -> Result<()> {
    let Some(actual) = shapes.get(name) else {
        bail!("missing required tensor {name}");
    };
    if actual.as_slice() != expected {
        bail!("tensor {name} has shape {actual:?}, expected {expected:?}");
    }
    Ok(())
}
