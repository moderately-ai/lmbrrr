use std::{fs::File, path::Path};

use anyhow::{Context, Result};
use candle_nn::Activation;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct MiniCpmConfig {
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub vision_config: VisionConfig,
    pub text_config: TextConfig,
    #[serde(default = "default_insert_layer_id")]
    pub insert_layer_id: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default)]
    pub drop_vision_last_layer: bool,
    pub image_token_id: u32,
    pub video_token_id: u32,
    #[serde(default = "default_downsample_mode")]
    pub downsample_mode: String,
    #[serde(default = "default_merge_kernel_size")]
    pub merge_kernel_size: [usize; 2],
    #[serde(default = "default_merger_times")]
    pub merger_times: usize,
}

impl MiniCpmConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("open config {}", path.display()))?;
        let mut cfg: Self =
            serde_json::from_reader(std::io::BufReader::new(file)).with_context(|| format!("parse {}", path.display()))?;
        cfg.vision_config.insert_layer_id = cfg.insert_layer_id;
        Ok(cfg)
    }

    pub fn eos_ids(&self, generation: Option<&GenerationConfig>) -> Vec<u32> {
        let mut ids = generation
            .map(|g| g.eos_token_id.clone())
            .unwrap_or_default();
        if let Some(id) = self.eos_token_id {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        ids
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_type: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub hidden_act: Activation,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub layer_types: Vec<LayerType>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub rope_parameters: RopeParameters,
}

#[derive(Clone, Debug, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_act: Activation,
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub attention_dropout: f64,
    #[serde(default = "default_insert_layer_id")]
    pub insert_layer_id: usize,
    #[serde(default = "default_window_kernel_size")]
    pub window_kernel_size: [usize; 2],
}

#[derive(Clone, Debug, Deserialize)]
pub struct PreprocessorConfig {
    #[serde(default = "default_max_slice_nums")]
    pub max_slice_nums: usize,
    #[serde(default = "default_scale_resolution")]
    pub scale_resolution: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_true")]
    pub use_image_id: bool,
    #[serde(default = "default_true")]
    pub slice_mode: bool,
    #[serde(default = "default_image_mean")]
    pub image_mean: [f32; 3],
    #[serde(default = "default_image_std")]
    pub image_std: [f32; 3],
}

impl PreprocessorConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file =
            File::open(path).with_context(|| format!("open preprocessor {}", path.display()))?;
        serde_json::from_reader(std::io::BufReader::new(file)).with_context(|| format!("parse {}", path.display()))
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct GenerationConfig {
    #[serde(deserialize_with = "deserialize_token_ids")]
    pub eos_token_id: Vec<u32>,
}

impl GenerationConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("open generation config {}", path.display()))?;
        serde_json::from_reader(std::io::BufReader::new(file)).with_context(|| format!("parse {}", path.display()))
    }
}

fn deserialize_token_ids<'de, D>(deserializer: D) -> std::result::Result<Vec<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(u32),
        Many(Vec<u32>),
    }

    Ok(match OneOrMany::deserialize(deserializer)? {
        OneOrMany::One(id) => vec![id],
        OneOrMany::Many(ids) => ids,
    })
}

fn default_insert_layer_id() -> usize {
    6
}

fn default_image_size() -> usize {
    448
}

fn default_downsample_mode() -> String {
    "16x".to_string()
}

fn default_merge_kernel_size() -> [usize; 2] {
    [2, 2]
}

fn default_window_kernel_size() -> [usize; 2] {
    [2, 2]
}

fn default_merger_times() -> usize {
    1
}

fn default_partial_rotary_factor() -> f64 {
    1.0
}

fn default_rope_theta() -> f64 {
    10_000.0
}

fn default_max_slice_nums() -> usize {
    9
}

fn default_scale_resolution() -> usize {
    448
}

fn default_patch_size() -> usize {
    14
}

fn default_true() -> bool {
    true
}

fn default_image_mean() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}

fn default_image_std() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}
