use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use candle::{safetensors::Load, DType, Device};
use memmap2::MmapOptions;
use safetensors::{tensor::Dtype as SafeDtype, SafeTensors};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize)]
pub struct CalibrationRow {
    pub id: String,
    pub category: String,
    pub modality: String,
    pub enable_thinking: bool,
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
    #[serde(default)]
    pub expected_behavior: Option<String>,
    pub prompt_token_count: usize,
    pub token_ids: Vec<u32>,
    #[serde(default)]
    pub sensitivity_focus: Vec<String>,
    #[serde(default)]
    pub media_status: Option<String>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum QuantFormat {
    SymmetricInt4,
    SymmetricInt5,
    SymmetricInt8,
}

impl QuantFormat {
    pub fn name(self) -> &'static str {
        match self {
            Self::SymmetricInt4 => "q4_symmetric",
            Self::SymmetricInt5 => "q5_symmetric",
            Self::SymmetricInt8 => "q8_symmetric",
        }
    }

    pub fn bits(self) -> u8 {
        match self {
            Self::SymmetricInt4 => 4,
            Self::SymmetricInt5 => 5,
            Self::SymmetricInt8 => 8,
        }
    }

    fn qmax(self) -> f32 {
        ((1u32 << (self.bits() - 1)) - 1) as f32
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct WeightTensorInfo {
    pub name: String,
    pub family: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub num_elements: usize,
    pub original_bytes: usize,
    pub quantizable: bool,
    pub protected: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct WeightError {
    pub relative_mse: f64,
    pub mse: f64,
    pub mean_square: f64,
    pub max_abs_error: f32,
    pub max_abs_weight: f32,
    pub scale: f32,
    pub worst_output_channel: Option<usize>,
    pub worst_output_channel_mse: Option<f64>,
    pub mean_output_channel_mse: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct QuantModuleScore {
    pub name: String,
    pub family: String,
    pub candidate_quant: String,
    pub bits: u8,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub num_elements: usize,
    pub original_bytes: usize,
    pub estimated_quantized_bytes: usize,
    pub estimated_compression_ratio: f64,
    pub protected: bool,
    pub protection_reason: Option<String>,
    pub weight_error: WeightError,
    pub activation_error: serde_json::Value,
    pub logit_drift: serde_json::Value,
    pub top1_flip_rate: Option<f64>,
    pub latency_delta: serde_json::Value,
    pub recommended_policy: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct WeightSensitivityReport {
    pub modules: Vec<QuantModuleScore>,
    pub skipped_modules: Vec<WeightTensorInfo>,
    pub family_counts: BTreeMap<String, usize>,
    pub scan_seconds: f64,
}

pub fn read_calibration_jsonl(path: &Path) -> Result<Vec<CalibrationRow>> {
    let file = File::open(path).with_context(|| format!("open calibration {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read calibration line {}", idx + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        rows.push(
            serde_json::from_str::<CalibrationRow>(&line)
                .with_context(|| format!("parse calibration line {}", idx + 1))?,
        );
    }
    if rows.is_empty() {
        anyhow::bail!("calibration file {} contains no rows", path.display());
    }
    Ok(rows)
}

pub fn score_weight_sensitivity(
    files: &[PathBuf],
    formats: &[QuantFormat],
    max_modules: Option<usize>,
    include_protected: bool,
) -> Result<WeightSensitivityReport> {
    if formats.is_empty() {
        anyhow::bail!("at least one candidate quantization format is required");
    }

    let started = Instant::now();
    let mut modules = Vec::new();
    let mut skipped_modules = Vec::new();
    let mut family_counts = BTreeMap::<String, usize>::new();
    let mut scored_tensors = 0usize;

    for path in files {
        let file = File::open(path).with_context(|| format!("open weights {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("mmap {}", path.display()))?;
        let safetensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("read safetensors {}", path.display()))?;

        let mut names = safetensors.names();
        names.sort_unstable();
        for name in names {
            let view = safetensors
                .tensor(name)
                .with_context(|| format!("read tensor metadata {name}"))?;
            let shape = view.shape().to_vec();
            let num_elements = shape_num_elements(&shape);
            let dtype = format!("{:?}", view.dtype());
            let original_bytes = num_elements * dtype_size_bytes(view.dtype());
            let family = tensor_family(name).to_string();
            *family_counts.entry(family.clone()).or_insert(0) += 1;
            let candidate = candidate_reason(name, &shape);

            if !candidate.quantizable || (candidate.protected && !include_protected) {
                skipped_modules.push(WeightTensorInfo {
                    name: name.to_string(),
                    family,
                    dtype,
                    shape,
                    num_elements,
                    original_bytes,
                    quantizable: candidate.quantizable,
                    protected: candidate.protected,
                    reason: if candidate.protected && !include_protected {
                        format!("protected_not_scored:{}", candidate.reason)
                    } else {
                        candidate.reason
                    },
                });
                continue;
            }

            if max_modules.is_some_and(|limit| scored_tensors >= limit) {
                skipped_modules.push(WeightTensorInfo {
                    name: name.to_string(),
                    family,
                    dtype,
                    shape,
                    num_elements,
                    original_bytes,
                    quantizable: true,
                    protected: candidate.protected,
                    reason: "not_scored_max_modules_reached".to_string(),
                });
                continue;
            }

            let values = load_tensor_values(&view)
                .with_context(|| format!("load tensor values for quant sensitivity {name}"))?;
            for format in formats {
                let quantize_started = Instant::now();
                let weight_error = score_symmetric_quantization(&values, &shape, *format);
                let quantize_elapsed = quantize_started.elapsed();
                let estimated_quantized_bytes = estimated_quantized_bytes(num_elements, *format);
                modules.push(QuantModuleScore {
                    name: name.to_string(),
                    family: family.clone(),
                    candidate_quant: format.name().to_string(),
                    bits: format.bits(),
                    dtype: dtype.clone(),
                    shape: shape.clone(),
                    num_elements,
                    original_bytes,
                    estimated_quantized_bytes,
                    estimated_compression_ratio: ratio(original_bytes, estimated_quantized_bytes),
                    protected: candidate.protected,
                    protection_reason: candidate.protected.then_some(candidate.reason.clone()),
                    recommended_policy: recommended_policy(&family, *format, candidate.protected, &weight_error),
                    activation_error: serde_json::json!({
                        "status": "not_collected",
                        "reason": "module activation hooks are not implemented yet; this pass records weight error and baseline calibration logits"
                    }),
                    logit_drift: serde_json::json!({
                        "status": "not_collected",
                        "reason": "per-module quantized-forward perturbation is not implemented yet"
                    }),
                    top1_flip_rate: None,
                    latency_delta: serde_json::json!({
                        "status": "weight_scan_only",
                        "quant_simulation_seconds": secs(quantize_elapsed),
                        "runtime_delta_seconds": null
                    }),
                    weight_error,
                });
            }
            scored_tensors += 1;
        }
    }

    Ok(WeightSensitivityReport {
        modules,
        skipped_modules,
        family_counts,
        scan_seconds: secs(started.elapsed()),
    })
}

pub fn aggregate_calibration(rows: &[CalibrationRow]) -> serde_json::Value {
    let mut modality_counts = BTreeMap::<String, usize>::new();
    let mut category_counts = BTreeMap::<String, usize>::new();
    let mut thinking_counts = BTreeMap::<String, usize>::new();
    let mut total_prompt_tokens = 0usize;
    for row in rows {
        *modality_counts.entry(row.modality.clone()).or_insert(0) += 1;
        *category_counts.entry(row.category.clone()).or_insert(0) += 1;
        *thinking_counts
            .entry(row.enable_thinking.to_string())
            .or_insert(0) += 1;
        total_prompt_tokens += row.prompt_token_count;
    }
    serde_json::json!({
        "rows": rows.len(),
        "modality_counts": modality_counts,
        "category_counts": category_counts,
        "thinking_counts": thinking_counts,
        "total_prompt_tokens": total_prompt_tokens,
    })
}

#[derive(Clone, Debug)]
struct CandidateReason {
    quantizable: bool,
    protected: bool,
    reason: String,
}

fn candidate_reason(name: &str, shape: &[usize]) -> CandidateReason {
    if shape.len() < 2 {
        return CandidateReason {
            quantizable: false,
            protected: is_protected_tensor(name),
            reason: "rank_lt_2".to_string(),
        };
    }
    if !name.ends_with(".weight") {
        return CandidateReason {
            quantizable: false,
            protected: is_protected_tensor(name),
            reason: "non_weight_tensor".to_string(),
        };
    }
    CandidateReason {
        quantizable: true,
        protected: is_protected_tensor(name),
        reason: tensor_protection_reason(name)
            .unwrap_or("candidate_weight")
            .to_string(),
    }
}

pub fn is_protected_tensor(name: &str) -> bool {
    tensor_protection_reason(name).is_some()
}

pub fn tensor_protection_reason(name: &str) -> Option<&'static str> {
    if name.contains("embed_tokens") {
        Some("token_embedding_protected")
    } else if name == "lm_head.weight" {
        Some("lm_head_protected")
    } else if name.contains("layernorm")
        || name.ends_with(".norm.weight")
        || name.contains(".q_norm.")
        || name.contains(".k_norm.")
    {
        Some("norm_protected")
    } else if name.contains(".linear_attn.A_log") || name.contains(".linear_attn.dt_bias") {
        Some("deltanet_state_protected")
    } else if name.contains(".linear_attn.conv1d.") {
        Some("deltanet_conv_state_path_protected")
    } else if name.contains("vision_tower") {
        Some("vision_tower_protected_initially")
    } else if name.contains("model.merger") {
        Some("multimodal_merger_protected_initially")
    } else {
        None
    }
}

pub fn tensor_family(name: &str) -> &'static str {
    if name == "lm_head.weight" {
        "text.lm_head"
    } else if name.contains("embed_tokens") {
        "text.embedding"
    } else if name.contains("vision_tower.embeddings") {
        "vision.embedding"
    } else if name.contains("vision_tower") && name.contains("self_attn") {
        "vision.attention"
    } else if name.contains("vision_tower") && name.contains("mlp") {
        "vision.mlp"
    } else if name.contains("vision_tower") && name.contains("layernorm") {
        "vision.norm"
    } else if name.contains("model.merger") {
        "vision.merger"
    } else if name.contains(".linear_attn.A_log") || name.contains(".linear_attn.dt_bias") {
        "text.deltanet_state"
    } else if name.contains(".linear_attn.conv1d.") {
        "text.deltanet_conv"
    } else if name.contains(".linear_attn.") {
        "text.deltanet"
    } else if name.contains(".self_attn.") {
        "text.full_attention"
    } else if name.contains(".mlp.") {
        "text.mlp"
    } else if name.contains("layernorm") || name.ends_with(".norm.weight") {
        "text.norm"
    } else if name.starts_with("model.language_model") {
        "text.other"
    } else {
        "other"
    }
}

fn load_tensor_values(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    let tensor = view.load(&Device::Cpu)?;
    Ok(tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

fn score_symmetric_quantization(
    values: &[f32],
    shape: &[usize],
    format: QuantFormat,
) -> WeightError {
    let max_abs_weight = values.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    let qmax = format.qmax();
    let scale = if max_abs_weight > 0.0 {
        max_abs_weight / qmax
    } else {
        1.0
    };

    let mut squared_error = 0.0f64;
    let mut squared_weight = 0.0f64;
    let mut max_abs_error = 0.0f32;
    let mut channel_error = channel_accumulator(shape);

    for (index, value) in values.iter().copied().enumerate() {
        let quantized = if max_abs_weight > 0.0 {
            (value / scale).round().clamp(-qmax, qmax)
        } else {
            0.0
        };
        let dequantized = quantized * scale;
        let error = value - dequantized;
        let abs_error = error.abs();
        max_abs_error = max_abs_error.max(abs_error);
        let sq_error = (error as f64) * (error as f64);
        squared_error += sq_error;
        squared_weight += (value as f64) * (value as f64);
        if let Some(acc) = channel_error.as_mut() {
            let channel = index / acc.channel_width;
            acc.squared_error[channel] += sq_error;
            acc.counts[channel] += 1;
        }
    }

    let count = values.len().max(1) as f64;
    let mse = squared_error / count;
    let mean_square = squared_weight / count;
    let relative_mse = if mean_square > 0.0 {
        mse / mean_square
    } else {
        0.0
    };
    let (worst_output_channel, worst_output_channel_mse, mean_output_channel_mse) =
        summarize_channels(channel_error);

    WeightError {
        relative_mse,
        mse,
        mean_square,
        max_abs_error,
        max_abs_weight,
        scale,
        worst_output_channel,
        worst_output_channel_mse,
        mean_output_channel_mse,
    }
}

#[derive(Clone, Debug)]
struct ChannelAccumulator {
    channel_width: usize,
    squared_error: Vec<f64>,
    counts: Vec<usize>,
}

fn channel_accumulator(shape: &[usize]) -> Option<ChannelAccumulator> {
    let channels = *shape.first()?;
    if channels == 0 {
        return None;
    }
    let num_elements = shape_num_elements(shape);
    let channel_width = num_elements / channels;
    if channel_width == 0 || channel_width * channels != num_elements {
        return None;
    }
    Some(ChannelAccumulator {
        channel_width,
        squared_error: vec![0.0; channels],
        counts: vec![0; channels],
    })
}

fn summarize_channels(
    channel_error: Option<ChannelAccumulator>,
) -> (Option<usize>, Option<f64>, Option<f64>) {
    let Some(channel_error) = channel_error else {
        return (None, None, None);
    };
    let mut worst = None::<(usize, f64)>;
    let mut total = 0.0f64;
    let mut count = 0usize;
    for (idx, (squared_error, samples)) in channel_error
        .squared_error
        .iter()
        .zip(channel_error.counts.iter())
        .enumerate()
    {
        if *samples == 0 {
            continue;
        }
        let mse = *squared_error / *samples as f64;
        total += mse;
        count += 1;
        if worst.map(|(_, current)| mse > current).unwrap_or(true) {
            worst = Some((idx, mse));
        }
    }
    (
        worst.map(|(idx, _)| idx),
        worst.map(|(_, mse)| mse),
        (count > 0).then_some(total / count as f64),
    )
}

fn recommended_policy(
    family: &str,
    format: QuantFormat,
    protected: bool,
    weight_error: &WeightError,
) -> String {
    if protected {
        return "protect".to_string();
    }
    if format == QuantFormat::SymmetricInt8 {
        return "candidate_q8".to_string();
    }
    let threshold = if family == "text.mlp" { 0.002 } else { 0.00075 };
    if weight_error.relative_mse <= threshold {
        format!("candidate_{}", format.name())
    } else {
        "prefer_higher_precision".to_string()
    }
}

fn estimated_quantized_bytes(num_elements: usize, format: QuantFormat) -> usize {
    let packed = (num_elements * format.bits() as usize).div_ceil(8);
    packed + std::mem::size_of::<f32>()
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn shape_num_elements(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn dtype_size_bytes(dtype: SafeDtype) -> usize {
    match dtype {
        SafeDtype::BOOL
        | SafeDtype::U8
        | SafeDtype::I8
        | SafeDtype::F8_E5M2
        | SafeDtype::F8_E4M3 => 1,
        SafeDtype::I16 | SafeDtype::U16 | SafeDtype::F16 | SafeDtype::BF16 => 2,
        SafeDtype::I32 | SafeDtype::U32 | SafeDtype::F32 => 4,
        SafeDtype::I64 | SafeDtype::U64 | SafeDtype::F64 => 8,
        SafeDtype::F6_E2M3 | SafeDtype::F6_E3M2 | SafeDtype::F4 | SafeDtype::F8_E8M0 => 1,
        _ => 1,
    }
}

fn secs(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_minicpm_families() {
        assert_eq!(
            tensor_family("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
            "text.deltanet"
        );
        assert_eq!(
            tensor_family("model.language_model.layers.11.self_attn.q_proj.weight"),
            "text.full_attention"
        );
        assert_eq!(
            tensor_family("model.vision_tower.encoder.layers.0.mlp.fc1.weight"),
            "vision.mlp"
        );
        assert_eq!(
            tensor_family("model.merger.mlp.0.linear_1.weight"),
            "vision.merger"
        );
    }

    #[test]
    fn symmetric_quantization_error_is_zero_for_exact_grid() {
        let values = [-1.0, 0.0, 1.0];
        let score = score_symmetric_quantization(&values, &[3, 1], QuantFormat::SymmetricInt4);
        assert_eq!(score.mse, 0.0);
        assert_eq!(score.relative_mse, 0.0);
        assert_eq!(score.worst_output_channel, Some(0));
    }

    #[test]
    fn calibration_aggregation_counts_rows() {
        let rows = vec![
            CalibrationRow {
                id: "a".to_string(),
                category: "short".to_string(),
                modality: "text".to_string(),
                enable_thinking: false,
                max_new_tokens: None,
                expected_behavior: None,
                prompt_token_count: 3,
                token_ids: vec![1, 2, 3],
                sensitivity_focus: vec![],
                media_status: None,
            },
            CalibrationRow {
                id: "b".to_string(),
                category: "ocr".to_string(),
                modality: "image".to_string(),
                enable_thinking: true,
                max_new_tokens: None,
                expected_behavior: None,
                prompt_token_count: 4,
                token_ids: vec![1, 2, 3, 4],
                sensitivity_focus: vec![],
                media_status: Some("metadata_only".to_string()),
            },
        ];
        let value = aggregate_calibration(&rows);
        assert_eq!(value["rows"], 2);
        assert_eq!(value["total_prompt_tokens"], 7);
        assert_eq!(value["modality_counts"]["text"], 1);
        assert_eq!(value["modality_counts"]["image"], 1);
    }
}
