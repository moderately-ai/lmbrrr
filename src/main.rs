#![recursion_limit = "256"]

use std::{
    collections::HashMap,
    fs::{self, OpenOptions},
    io::{BufWriter, IsTerminal, Stdout, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use candle::{
    quantized::{GgmlDType, QMatMul, QTensor},
    safetensors::Load,
    DType, Device, Module, Tensor, D,
};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    utils::apply_repeat_penalty,
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use crossterm::{
    cursor::{Hide, Show},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use lmbrrr::{
    artifacts::{resolve_model_artifacts, ArtifactOverrides, Artifacts},
    config::{GenerationConfig, MiniCpmConfig, PreprocessorConfig},
    image_processor::{preprocess_paths, ProcessedImages},
    minicpm::MiniCpmForConditionalGeneration,
    prompt::{chat_prompt, expand_image_placeholders},
    quant_convert::{convert_mixed_precision, ConversionOptions, MixedPrecisionPolicy},
    quant_sensitivity::{
        aggregate_calibration, read_calibration_jsonl, score_weight_sensitivity, CalibrationRow,
        QuantFormat,
    },
    quantized_linear::QuantizedTextArtifact,
    qwen35::{Qwen35HiddenStateTrace, Qwen35ProfileEvent, Qwen35Profiler, Qwen35TraceRecorder},
    token_stream::TokenOutputStream,
    weights::{validate_minicpm_header, WeightReport},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    text::Line,
    widgets::{Block, BorderType, Borders, Paragraph, Wrap},
    Terminal,
};
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(name = "lmbrrr")]
#[command(about = "Small-model inference experiments on Candle")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Run(RunArgs),
    Bench(BenchArgs),
    Logits(LogitsArgs),
    Profile(ProfileArgs),
    SpecVerify(SpecVerifyArgs),
    Trace(TraceArgs),
    QuantSensitivity(QuantSensitivityArgs),
    QuantConvert(QuantConvertArgs),
    QuantMatmulBench(QuantMatmulBenchArgs),
    QuantQuality(QuantQualityArgs),
    EagleChainDraft(EagleChainDraftArgs),
    EagleLiveProbe(EagleLiveProbeArgs),
    EagleRecurrentDraft(EagleRecurrentDraftArgs),
    Roofline(RooflineArgs),
    VerifyTable(VerifyTableArgs),
}

#[derive(Args, Clone, Debug)]
struct ModelArgs {
    #[arg(long, default_value = "openbmb/MiniCPM-V-4.6")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long, default_value = "16x")]
    downsample_mode: String,

    #[arg(long)]
    cpu: bool,

    #[arg(long, value_enum, default_value_t = DTypeArg::Auto)]
    dtype: DTypeArg,

    #[arg(long)]
    config: Option<PathBuf>,

    #[arg(long)]
    tokenizer: Option<PathBuf>,

    #[arg(long)]
    generation_config: Option<PathBuf>,

    #[arg(long)]
    preprocessor: Option<PathBuf>,

    #[arg(long = "weights")]
    weights: Vec<PathBuf>,

    #[arg(long)]
    quantized_manifest: Option<PathBuf>,
}

#[derive(Args, Clone, Debug)]
struct GenerationArgs {
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long)]
    top_k: Option<usize>,

    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long)]
    enable_thinking: bool,
}

#[derive(Parser, Debug)]
struct RunArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(long)]
    prompt: String,

    #[arg(long = "image")]
    images: Vec<PathBuf>,

    #[command(flatten)]
    generation: GenerationArgs,

    #[arg(long)]
    dry_run: bool,

    #[arg(long)]
    no_progress: bool,
}

#[derive(Parser, Debug)]
struct BenchArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[command(flatten)]
    generation: GenerationArgs,

    #[arg(long = "profile", value_enum)]
    profiles: Vec<BenchProfile>,

    #[arg(long = "prompt")]
    prompts: Vec<String>,

    #[arg(long, default_value_t = 1)]
    warmup: usize,

    #[arg(long, default_value_t = 3)]
    iterations: usize,

    #[arg(long)]
    output: Option<PathBuf>,

    #[arg(long)]
    append: bool,
}

#[derive(Parser, Debug)]
struct LogitsArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(
        long,
        default_value = "evals/fixtures/minicpm_v46_transformers_text_logits.json"
    )]
    fixture: PathBuf,

    #[arg(long, default_value_t = 10)]
    top_k: usize,

    #[arg(long)]
    output: Option<PathBuf>,

    #[arg(long)]
    fail_on_mismatch: bool,
}

#[derive(Parser, Debug)]
struct ProfileArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(long = "profile", value_enum, default_value_t = BenchProfile::Long)]
    profile: BenchProfile,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct SpecVerifyArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(long)]
    prompt: String,

    #[arg(long)]
    enable_thinking: bool,

    #[arg(long = "draft-token", value_delimiter = ',')]
    draft_tokens: Vec<u32>,

    #[arg(long = "draft-confidence", value_delimiter = ',')]
    draft_confidences: Vec<f64>,

    #[arg(long)]
    schedule_confidence_threshold: Option<f64>,

    #[arg(long)]
    baseline_draft_tokens: Option<usize>,

    #[arg(long)]
    corrupt_draft_at: Option<usize>,

    #[arg(long)]
    output: Option<PathBuf>,

    #[arg(long)]
    fail_on_mismatch: bool,
}

#[derive(Parser, Debug)]
struct TraceArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(long)]
    prompt: String,

    #[arg(long)]
    enable_thinking: bool,

    #[arg(long, default_value_t = 16)]
    max_new_tokens: usize,

    #[arg(long = "capture-layer", value_delimiter = ',')]
    capture_layers: Vec<usize>,

    #[arg(long, default_value_t = 8)]
    top_k_logits: usize,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct QuantSensitivityArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(
        long,
        default_value = "evals/calibration/minicpm_v46_quant_calibration.jsonl"
    )]
    calibration: PathBuf,

    #[arg(long = "candidate-quant", value_enum)]
    candidate_quants: Vec<QuantFormatArg>,

    #[arg(long)]
    max_cases: Option<usize>,

    #[arg(long)]
    max_modules: Option<usize>,

    #[arg(long)]
    include_protected: bool,

    #[arg(long, default_value_t = 8)]
    top_k_logits: usize,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct QuantConvertArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(long, default_value = "target/minicpm-v46-quant-sensitivity.json")]
    sensitivity: PathBuf,

    #[arg(long, value_enum, default_value_t = MixedPrecisionPolicyArg::Q8TextLinears)]
    policy: MixedPrecisionPolicyArg,

    #[arg(long, default_value = "target/minicpm-v46-mixed-precision")]
    output_dir: PathBuf,

    #[arg(long)]
    max_tensors: Option<usize>,

    #[arg(long)]
    manifest_only: bool,
}

#[derive(Parser, Debug)]
struct QuantMatmulBenchArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(long, default_value_t = 128)]
    chunk_tokens: usize,

    #[arg(long, default_value_t = 2)]
    warmup: usize,

    #[arg(long, default_value_t = 5)]
    iterations: usize,

    #[arg(long)]
    include_lm_head: bool,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct QuantQualityArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[command(flatten)]
    generation: GenerationArgs,

    #[arg(
        long,
        default_value = "evals/calibration/minicpm_v46_quant_calibration.jsonl"
    )]
    calibration: PathBuf,

    #[arg(long = "case-id")]
    case_ids: Vec<String>,

    #[arg(long)]
    max_cases: Option<usize>,

    #[arg(long, default_value = "target/minicpm-v46-q8-full/manifest.json")]
    q8_manifest: PathBuf,

    #[arg(long, default_value = "target/minicpm-v46-q4k-mlp-full/manifest.json")]
    q4_mlp_manifest: PathBuf,

    #[arg(
        long,
        default_value = "target/minicpm-v46-q4k-text-safe-full/manifest.json"
    )]
    q4_text_safe_manifest: PathBuf,

    #[arg(
        long,
        default_value = "target/minicpm-v46-q4k-mlp-q8-text-full/manifest.json"
    )]
    mixed_manifest: PathBuf,

    #[arg(long, default_value_t = 0.25)]
    min_prefix_ratio: f64,

    #[arg(long, default_value_t = 0.50)]
    min_token_jaccard: f64,

    #[arg(long, default_value_t = 0.50)]
    min_lexical_jaccard: f64,

    #[arg(long, default_value_t = 0.50)]
    max_length_ratio_delta: f64,

    #[arg(long)]
    fail_on_gate: bool,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct EagleChainDraftArgs {
    #[arg(long)]
    trace: PathBuf,

    #[arg(long, default_value_t = 4)]
    draft_width: usize,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct EagleLiveProbeArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(long)]
    prompt: String,

    #[arg(long)]
    draft_head_manifest: PathBuf,

    #[arg(long, default_value_t = 8)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 4)]
    draft_width: usize,

    #[arg(long)]
    schedule_confidence_threshold: Option<f64>,

    #[arg(long)]
    enable_thinking: bool,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct EagleRecurrentDraftArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(long)]
    prompt: String,

    #[arg(long)]
    drafter_manifest: PathBuf,

    #[arg(long, default_value_t = 4)]
    draft_width: usize,

    #[arg(long)]
    schedule_confidence_threshold: Option<f64>,

    #[arg(long)]
    enable_thinking: bool,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct RooflineArgs {
    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value_t = 2)]
    warmup: usize,

    #[arg(long, default_value_t = 10)]
    iterations: usize,

    #[arg(long, default_value_t = 256)]
    dispatch_chain: usize,

    /// Estimated Metal dispatches per decode forward, used for the derived
    /// dispatch-bound projection. Refine once encoder-level counting exists.
    #[arg(long, default_value_t = 550)]
    assumed_dispatches: usize,

    /// Weight bytes read per decode forward for the bandwidth-bound
    /// projection. Default is the BF16 text decoder (~1.5 GB).
    #[arg(long, default_value_t = 1_500_000_000)]
    assumed_weight_bytes: u64,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct VerifyTableArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(long = "gamma", value_delimiter = ',')]
    gammas: Vec<usize>,

    #[arg(long = "profile", value_enum)]
    profiles: Vec<BenchProfile>,

    #[arg(long, default_value_t = 1)]
    warmup: usize,

    #[arg(long, default_value_t = 5)]
    iterations: usize,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DTypeArg {
    Auto,
    F32,
    F16,
    Bf16,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum QuantFormatArg {
    Q4Symmetric,
    Q5Symmetric,
    Q8Symmetric,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum MixedPrecisionPolicyArg {
    Q8TextLinears,
    Q4kMlpOnly,
    Q4kTextSafe,
    Q4kMlpQ8Text,
}

impl MixedPrecisionPolicyArg {
    fn resolve(self) -> MixedPrecisionPolicy {
        match self {
            Self::Q8TextLinears => MixedPrecisionPolicy::Q8TextLinears,
            Self::Q4kMlpOnly => MixedPrecisionPolicy::Q4KMlpOnly,
            Self::Q4kTextSafe => MixedPrecisionPolicy::Q4KTextSafe,
            Self::Q4kMlpQ8Text => MixedPrecisionPolicy::Q4KMlpQ8Text,
        }
    }
}

impl QuantFormatArg {
    fn resolve(self) -> QuantFormat {
        match self {
            Self::Q4Symmetric => QuantFormat::SymmetricInt4,
            Self::Q5Symmetric => QuantFormat::SymmetricInt5,
            Self::Q8Symmetric => QuantFormat::SymmetricInt8,
        }
    }
}

impl DTypeArg {
    fn resolve(self, device: &Device) -> DType {
        match self {
            Self::Auto if device.is_cpu() => DType::F32,
            Self::Auto => DType::BF16,
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::BF16,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BenchProfile {
    Short,
    Medium,
    Long,
}

impl BenchProfile {
    fn all() -> [Self; 3] {
        [Self::Short, Self::Medium, Self::Long]
    }

    fn name(self) -> &'static str {
        match self {
            Self::Short => "short",
            Self::Medium => "medium",
            Self::Long => "long",
        }
    }

    fn prompt(self) -> &'static str {
        match self {
            Self::Short => "Answer in one sentence: what is 17 * 23?",
            Self::Medium => {
                "A notebook costs $3, a pen costs $2, and a folder costs $5. A student buys 4 notebooks, 6 pens, and 3 folders. What is the total cost?"
            }
            Self::Long => {
                "Solve this carefully. A lab runs three model-evaluation batches. Batch A has 18 prompts and each prompt takes 7 seconds. Batch B has twice as many prompts, but each prompt takes 5 seconds. Batch C has 12 prompts, each taking 11 seconds, and can only start after Batch A finishes. If Batch A and Batch B start together, what is the earliest time when all three batches are complete?"
            }
        }
    }
}

#[derive(Clone, Debug)]
struct BenchPrompt {
    name: String,
    text: String,
}

#[derive(Debug)]
struct ArtifactBundle {
    artifacts: Artifacts,
    config: MiniCpmConfig,
    generation_config: Option<GenerationConfig>,
    weight_report: WeightReport,
    elapsed: Duration,
}

#[derive(Clone, Debug)]
struct QuantizedLoadStats {
    manifest: PathBuf,
    quantized_tensors: usize,
    replaced_text_linears: usize,
    backend: String,
    quantized_data_bytes: u64,
    dense_equivalent_bytes: usize,
}

#[derive(Clone, Debug)]
struct GenerationStats {
    prompt_tokens: usize,
    generated_tokens: usize,
    generated_token_ids: Vec<u32>,
    eos_reached: bool,
    prefill_elapsed: Duration,
    decode_elapsed: Duration,
    decode_model_elapsed: Duration,
    sampling_elapsed: Duration,
    next_input_elapsed: Duration,
    callback_elapsed: Duration,
    first_token_after_prefill: Option<Duration>,
}

#[derive(Clone, Debug)]
struct QuantQualityGeneration {
    stats: GenerationStats,
    raw_text: String,
    reasoning_text: String,
    answer_text: String,
}

#[derive(Clone, Debug)]
struct QuantQualityPolicyRun {
    label: String,
    manifest: Option<PathBuf>,
    load_elapsed: Duration,
    run_elapsed: Duration,
    quantized_load: Option<QuantizedLoadStats>,
    generations: Vec<QuantQualityGeneration>,
}

#[derive(Clone, Debug)]
struct QualityThresholds {
    min_prefix_ratio: f64,
    min_token_jaccard: f64,
    min_lexical_jaccard: f64,
    max_length_ratio_delta: f64,
}

#[derive(Clone, Debug)]
struct QualityComparison {
    exact_token_match: bool,
    common_prefix_tokens: usize,
    divergence_index: Option<usize>,
    prefix_ratio: f64,
    token_jaccard: f64,
    lexical_jaccard: f64,
    length_ratio: f64,
    length_ratio_delta: f64,
    passed_gate: bool,
}

impl GenerationStats {
    fn total_generated_tokens(&self) -> usize {
        self.prompt_tokens + self.generated_tokens
    }

    fn time_to_first_token(&self) -> Option<Duration> {
        self.first_token_after_prefill
            .map(|decode| self.prefill_elapsed + decode)
    }

    fn prefill_tokens_per_second(&self) -> f64 {
        tokens_per_second(self.prompt_tokens, self.prefill_elapsed)
    }

    fn decode_tokens_per_second(&self) -> f64 {
        tokens_per_second(self.generated_tokens, self.decode_elapsed)
    }

    fn decode_model_tokens(&self) -> usize {
        self.generated_tokens.saturating_sub(1)
    }

    fn decode_model_tokens_per_second(&self) -> Option<f64> {
        let tokens = self.decode_model_tokens();
        (tokens > 0).then(|| tokens_per_second(tokens, self.decode_model_elapsed))
    }

    fn sampling_tokens_per_second(&self) -> f64 {
        tokens_per_second(self.generated_tokens, self.sampling_elapsed)
    }

    fn decode_non_model_elapsed(&self) -> Duration {
        self.decode_elapsed
            .saturating_sub(self.decode_model_elapsed)
    }

    fn decode_non_model_share(&self) -> f64 {
        if self.decode_elapsed.is_zero() {
            0.0
        } else {
            self.decode_non_model_elapsed().as_secs_f64() / self.decode_elapsed.as_secs_f64()
        }
    }

    fn decode_bookkeeping_elapsed(&self) -> Duration {
        self.decode_elapsed.saturating_sub(
            self.decode_model_elapsed
                + self.sampling_elapsed
                + self.next_input_elapsed
                + self.callback_elapsed,
        )
    }

    fn steady_state_tokens_per_second(&self) -> Option<f64> {
        let first = self.first_token_after_prefill?;
        let steady_elapsed = self.decode_elapsed.checked_sub(first)?;
        let steady_tokens = self.generated_tokens.checked_sub(1)?;
        if steady_tokens == 0 {
            None
        } else {
            Some(tokens_per_second(steady_tokens, steady_elapsed))
        }
    }
}

#[derive(Clone, Debug)]
struct ReasoningParts {
    raw_text: String,
    reasoning_text: String,
    answer_text: String,
}

#[derive(Debug, Deserialize)]
struct LogitsOracleFixture {
    model_id: String,
    revision: String,
    cases: Vec<LogitsOracleCase>,
}

#[derive(Debug, Deserialize)]
struct LogitsOracleCase {
    id: String,
    user_prompt: String,
    image_count: usize,
    prompt_token_count: usize,
    token_ids: Vec<u32>,
    next_token_logits: Option<OracleTopLogits>,
}

#[derive(Debug, Deserialize)]
struct OracleTopLogits {
    top_token_ids: Option<Vec<u32>>,
    top_logits: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
struct TopLogit {
    token_id: u32,
    token: String,
    logit: f32,
}

#[derive(Clone, Debug)]
struct SpecVerifyPosition {
    index: usize,
    draft_token_id: u32,
    target_token_id: u32,
    token_match: bool,
    accepted: bool,
    first_rejected: bool,
}

#[derive(Clone, Debug)]
struct SpecVerifyAnalysis {
    positions: Vec<SpecVerifyPosition>,
    accepted_tokens: usize,
    first_rejected_index: Option<usize>,
    bonus_token_id: u32,
    reconstructed_token_ids: Vec<u32>,
}

impl SpecVerifyAnalysis {
    fn verified_tokens(&self) -> usize {
        self.positions.len()
    }

    fn bonus_tokens(&self) -> usize {
        1
    }

    fn accepted_length(&self) -> usize {
        self.accepted_tokens + self.bonus_tokens()
    }

    fn acceptance_rate(&self) -> Option<f64> {
        let verified = self.verified_tokens();
        (verified > 0).then(|| self.accepted_tokens as f64 / verified as f64)
    }

    fn verifier_waste_tokens(&self) -> usize {
        self.first_rejected_index
            .map(|idx| self.verified_tokens().saturating_sub(idx + 1))
            .unwrap_or(0)
    }

    fn verifier_waste_share(&self) -> Option<f64> {
        let verified = self.verified_tokens();
        (verified > 0).then(|| self.verifier_waste_tokens() as f64 / verified as f64)
    }
}

#[derive(Clone, Debug)]
struct SpecVerifyStats {
    analysis: SpecVerifyAnalysis,
    target_token_ids: Vec<u32>,
    prefill_elapsed: Duration,
    verify_elapsed: Duration,
    argmax_elapsed: Duration,
}

#[derive(Clone, Debug)]
struct ConfidenceSchedule {
    threshold: f64,
    original_draft_tokens: usize,
    scheduled_draft_tokens: usize,
    dropped_draft_tokens: usize,
    scheduled_cumulative_confidence: f64,
    next_rejected_cumulative_confidence: Option<f64>,
    confidences: Vec<f64>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum TextChannel {
    Reasoning,
    Answer,
}

impl TextChannel {
    fn label(self) -> &'static str {
        match self {
            Self::Reasoning => "Reasoning",
            Self::Answer => "Answer",
        }
    }
}

#[derive(Clone, Debug)]
enum TextEvent {
    Text(TextChannel, String),
}

#[derive(Clone, Debug)]
struct ReasoningTagParser {
    mode: TextChannel,
    pending: String,
}

impl ReasoningTagParser {
    fn new(mode: TextChannel) -> Self {
        Self {
            mode,
            pending: String::new(),
        }
    }

    fn feed(&mut self, text: &str) -> Vec<TextEvent> {
        self.pending.push_str(text);
        self.drain(false)
    }

    fn finish(&mut self) -> Vec<TextEvent> {
        self.drain(true)
    }

    fn drain(&mut self, final_chunk: bool) -> Vec<TextEvent> {
        const THINK_START: &str = "<think>";
        const THINK_END: &str = "</think>";

        let mut events = Vec::new();
        loop {
            let tag = match self.mode {
                TextChannel::Answer => THINK_START,
                TextChannel::Reasoning => THINK_END,
            };
            if let Some(idx) = self.pending.find(tag) {
                if idx > 0 {
                    events.push(TextEvent::Text(self.mode, self.pending[..idx].to_string()));
                }
                self.pending.drain(..idx + tag.len());
                self.mode = match self.mode {
                    TextChannel::Answer => TextChannel::Reasoning,
                    TextChannel::Reasoning => TextChannel::Answer,
                };
                continue;
            }

            if final_chunk {
                if !self.pending.is_empty() {
                    events.push(TextEvent::Text(
                        self.mode,
                        std::mem::take(&mut self.pending),
                    ));
                }
                break;
            }

            let keep = tag.len().saturating_sub(1);
            let emit_len = safe_prefix_len(&self.pending, keep);
            if emit_len == 0 {
                break;
            }
            let text = self.pending[..emit_len].to_string();
            self.pending.drain(..emit_len);
            events.push(TextEvent::Text(self.mode, text));
        }
        events
    }
}

#[derive(Debug)]
struct ReasoningRenderer {
    parser: ReasoningTagParser,
    active: Option<TextChannel>,
}

impl ReasoningRenderer {
    fn new(initial_channel: TextChannel) -> Self {
        Self {
            parser: ReasoningTagParser::new(initial_channel),
            active: None,
        }
    }

    fn write_chunk(&mut self, text: &str) -> Result<()> {
        for event in self.parser.feed(text) {
            self.render(event)?;
        }
        std::io::stdout().flush().ok();
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        for event in self.parser.finish() {
            self.render(event)?;
        }
        if self.active.is_some() {
            println!();
        }
        Ok(())
    }

    fn render(&mut self, event: TextEvent) -> Result<()> {
        match event {
            TextEvent::Text(_, text) if text.is_empty() => {}
            TextEvent::Text(channel, text) => {
                if self.active != Some(channel) {
                    if self.active.is_some() {
                        println!("\n");
                    }
                    println!("{}:", channel.label());
                    self.active = Some(channel);
                }
                print!("{text}");
            }
        }
        Ok(())
    }
}

struct TuiOutput {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    parser: ReasoningTagParser,
    reasoning_text: String,
    answer_text: String,
    prompt_tokens: usize,
    max_new_tokens: usize,
    last_draw: Instant,
}

impl TuiOutput {
    fn new(
        prompt_tokens: usize,
        max_new_tokens: usize,
        initial_channel: TextChannel,
    ) -> Result<Self> {
        enable_raw_mode().context("enable terminal raw mode")?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen, Hide).context("enter terminal UI")?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend).context("create terminal UI")?;
        terminal.clear().context("clear terminal UI")?;

        let mut output = Self {
            terminal,
            parser: ReasoningTagParser::new(initial_channel),
            reasoning_text: String::new(),
            answer_text: String::new(),
            prompt_tokens,
            max_new_tokens,
            last_draw: Instant::now(),
        };
        output.draw(0, Duration::ZERO, None)?;
        Ok(output)
    }

    fn write_chunk(
        &mut self,
        text: &str,
        generated: usize,
        decode_elapsed: Duration,
        prefill_elapsed: Duration,
    ) -> Result<()> {
        let events = self.parser.feed(text);
        self.append_events(events);
        if self.last_draw.elapsed() >= Duration::from_millis(80) || generated == self.max_new_tokens
        {
            self.draw(generated, decode_elapsed, Some(prefill_elapsed))?;
        }
        Ok(())
    }

    fn finish(mut self, stats: &GenerationStats) -> Result<ReasoningParts> {
        let events = self.parser.finish();
        self.append_events(events);
        self.draw(
            stats.generated_tokens,
            stats.decode_elapsed,
            Some(stats.prefill_elapsed),
        )?;
        Ok(ReasoningParts {
            raw_text: format!("{}{}", self.reasoning_text, self.answer_text),
            reasoning_text: self.reasoning_text.clone(),
            answer_text: self.answer_text.clone(),
        })
    }

    fn append_events(&mut self, events: Vec<TextEvent>) {
        for event in events {
            match event {
                TextEvent::Text(TextChannel::Reasoning, text) => {
                    self.reasoning_text.push_str(&text)
                }
                TextEvent::Text(TextChannel::Answer, text) => self.answer_text.push_str(&text),
            }
        }
    }

    fn draw(
        &mut self,
        generated: usize,
        decode_elapsed: Duration,
        prefill_elapsed: Option<Duration>,
    ) -> Result<()> {
        let prompt_tokens = self.prompt_tokens;
        let max_output_tokens = self.max_new_tokens;
        let total_tokens = prompt_tokens + generated;
        let max_total_tokens = prompt_tokens + max_output_tokens;
        let prefill_rate = prefill_elapsed
            .map(|elapsed| format!("{:.2}", tokens_per_second(prompt_tokens, elapsed)))
            .unwrap_or_else(|| "prefilling".to_string());
        let output_rate = tokens_per_second(generated, decode_elapsed);
        let metrics = vec![
            Line::from(format!(
                "prefill: {prefill_rate} tok/s | output: {output_rate:.2} tok/s"
            )),
            Line::from(format!(
                "prompt tokens: {prompt_tokens} | output tokens: {generated} / {max_output_tokens} | total tokens: {total_tokens} / {max_total_tokens}"
            )),
        ];
        let reasoning = self.reasoning_text.clone();
        let answer = self.answer_text.clone();

        self.terminal
            .draw(|frame| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(4),
                        Constraint::Percentage(45),
                        Constraint::Percentage(55),
                    ])
                    .split(frame.area());

                let metrics_widget = Paragraph::new(metrics).block(
                    Block::default()
                        .title("Metrics")
                        .borders(Borders::ALL)
                        .border_type(BorderType::Plain),
                );
                frame.render_widget(metrics_widget, chunks[0]);

                let reasoning_scroll = text_scroll(&reasoning, chunks[1].height);
                let reasoning_widget = Paragraph::new(reasoning)
                    .block(
                        Block::default()
                            .title("Reasoning")
                            .borders(Borders::ALL)
                            .border_type(BorderType::Plain),
                    )
                    .wrap(Wrap { trim: false })
                    .scroll((reasoning_scroll, 0));
                frame.render_widget(reasoning_widget, chunks[1]);

                let answer_scroll = text_scroll(&answer, chunks[2].height);
                let answer_widget = Paragraph::new(answer)
                    .block(
                        Block::default()
                            .title("Answer")
                            .borders(Borders::ALL)
                            .border_type(BorderType::Plain),
                    )
                    .wrap(Wrap { trim: false })
                    .scroll((answer_scroll, 0));
                frame.render_widget(answer_widget, chunks[2]);
            })
            .context("draw terminal UI")?;
        self.last_draw = Instant::now();
        Ok(())
    }
}

impl Drop for TuiOutput {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), Show, LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(args),
        Command::Bench(args) => bench(args),
        Command::Logits(args) => logits(args),
        Command::Profile(args) => profile_decode(args),
        Command::SpecVerify(args) => spec_verify(args),
        Command::Trace(args) => trace_hidden_states(args),
        Command::QuantSensitivity(args) => quant_sensitivity(args),
        Command::QuantConvert(args) => quant_convert(args),
        Command::QuantMatmulBench(args) => quant_matmul_bench(args),
        Command::QuantQuality(args) => quant_quality(args),
        Command::EagleChainDraft(args) => eagle_chain_draft(args),
        Command::EagleLiveProbe(args) => eagle_live_probe(args),
        Command::EagleRecurrentDraft(args) => eagle_recurrent_draft(args),
        Command::Roofline(args) => roofline(args),
        Command::VerifyTable(args) => verify_table(args),
    }
}

fn run(args: RunArgs) -> Result<()> {
    let bundle = resolve_artifacts(&args.model)?;

    if args.dry_run {
        println!(
            "{}",
            serde_json::json!({
                "model_id": args.model.model_id.as_str(),
                "revision": args.model.revision.as_str(),
                "config": bundle.artifacts.config,
                "tokenizer": bundle.artifacts.tokenizer,
                "weights": bundle.artifacts.weights,
                "quantized_manifest": args.model.quantized_manifest,
                "tensor_count": bundle.weight_report.tensor_count,
                "has_lm_head": bundle.weight_report.has_lm_head,
                "text_layers": bundle.config.text_config.num_hidden_layers,
                "vision_layers": bundle.config.vision_config.num_hidden_layers,
                "image_inputs": args.images.len(),
            })
        );
        return Ok(());
    }

    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let preprocessor = if args.images.is_empty() {
        None
    } else {
        let path = bundle
            .artifacts
            .preprocessor
            .as_ref()
            .context("image inputs require preprocessor_config.json")?;
        Some(PreprocessorConfig::from_path(path)?)
    };
    let processed_images = match (&preprocessor, args.images.is_empty()) {
        (Some(cfg), false) => Some(preprocess_paths(&args.images, cfg, &device)?),
        _ => None,
    };

    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let prompt_text = prepare_run_prompt(&args, preprocessor.as_ref(), processed_images.as_ref())?;
    let tokens = tokenize_prompt(&tokenizer, prompt_text)?;

    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
    )?;
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());
    let mut stream = TokenOutputStream::new(tokenizer);
    let use_tui = !args.no_progress && std::io::stdout().is_terminal();
    let initial_channel = if args.generation.enable_thinking {
        TextChannel::Reasoning
    } else {
        TextChannel::Answer
    };
    let stats = if use_tui {
        let mut renderer = TuiOutput::new(
            tokens.len(),
            args.generation.max_new_tokens,
            initial_channel,
        )?;
        let stats = generate_tokens(
            &mut model,
            &device,
            &args.generation,
            &tokens,
            processed_images.as_ref(),
            &args.model.downsample_mode,
            &eos_ids,
            |next_token, generated, elapsed, prefill_elapsed| {
                if let Some(text) = stream.next_token(next_token)? {
                    renderer.write_chunk(&text, generated, elapsed, prefill_elapsed)?;
                }
                Ok(())
            },
        )?;

        if let Some(rest) = stream.decode_rest()? {
            renderer.write_chunk(
                &rest,
                stats.generated_tokens,
                stats.decode_elapsed,
                stats.prefill_elapsed,
            )?;
        }
        let final_text = renderer.finish(&stats)?;
        print_reasoning_parts(&final_text)?;
        stats
    } else {
        let mut renderer = ReasoningRenderer::new(initial_channel);
        let stats = generate_tokens(
            &mut model,
            &device,
            &args.generation,
            &tokens,
            processed_images.as_ref(),
            &args.model.downsample_mode,
            &eos_ids,
            |next_token, _, _, _| {
                if let Some(text) = stream.next_token(next_token)? {
                    renderer.write_chunk(&text)?;
                }
                Ok(())
            },
        )?;

        if let Some(rest) = stream.decode_rest()? {
            renderer.write_chunk(&rest)?;
        }
        renderer.finish()?;
        stats
    };

    eprintln!(
        "{}",
        serde_json::json!({
            "artifact_seconds": secs(bundle.elapsed),
            "load_seconds": secs(load_elapsed),
            "prefill_seconds": secs(stats.prefill_elapsed),
            "prefill_tokens_per_second": stats.prefill_tokens_per_second(),
            "time_to_first_token_seconds": stats.time_to_first_token().map(secs),
            "decode_time_to_first_token_seconds": stats.first_token_after_prefill.map(secs),
            "decode_seconds": secs(stats.decode_elapsed),
            "decode_model_input_tokens": stats.decode_model_tokens(),
            "decode_model_seconds": secs(stats.decode_model_elapsed),
            "decode_model_tokens_per_second": stats.decode_model_tokens_per_second(),
            "decode_non_model_seconds": secs(stats.decode_non_model_elapsed()),
            "decode_non_model_share": stats.decode_non_model_share(),
            "sampling_seconds": secs(stats.sampling_elapsed),
            "sampling_tokens_per_second": stats.sampling_tokens_per_second(),
            "next_input_seconds": secs(stats.next_input_elapsed),
            "callback_seconds": secs(stats.callback_elapsed),
            "decode_bookkeeping_seconds": secs(stats.decode_bookkeeping_elapsed()),
            "prompt_tokens": stats.prompt_tokens,
            "generated_tokens": stats.generated_tokens,
            "total_tokens": stats.total_generated_tokens(),
            "max_generated_tokens": args.generation.max_new_tokens,
            "max_total_tokens": stats.prompt_tokens + args.generation.max_new_tokens,
            "eos_reached": stats.eos_reached,
            "output_tokens_per_second": stats.decode_tokens_per_second(),
            "decode_tokens_per_second": stats.decode_tokens_per_second(),
            "steady_state_tokens_per_second": stats.steady_state_tokens_per_second(),
            "device": format!("{device:?}"),
            "dtype": format!("{dtype:?}"),
            "enable_thinking": args.generation.enable_thinking,
            "quantized_load": quantized_load_json(&quantized_load),
        })
    );
    Ok(())
}

fn bench(args: BenchArgs) -> Result<()> {
    if args.iterations == 0 {
        anyhow::bail!("--iterations must be greater than zero");
    }

    let prompts = bench_prompts(&args);
    if prompts.is_empty() {
        anyhow::bail!("no benchmark prompts selected");
    }

    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let tokenized = prompts
        .iter()
        .map(|prompt| {
            let prompt_text = chat_prompt(&prompt.text, 0, args.generation.enable_thinking);
            Ok((prompt.clone(), tokenize_prompt(&tokenizer, prompt_text)?))
        })
        .collect::<Result<Vec<_>>>()?;

    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
    )?;
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());
    let mut writer = benchmark_writer(args.output.as_ref(), args.append)?;

    for (prompt, tokens) in &tokenized {
        for _ in 0..args.warmup {
            let _ = generate_tokens(
                &mut model,
                &device,
                &args.generation,
                tokens,
                None::<&ProcessedImages>,
                &args.model.downsample_mode,
                &eos_ids,
                |_, _, _, _| Ok(()),
            )?;
        }

        for iteration in 0..args.iterations {
            let stats = generate_tokens(
                &mut model,
                &device,
                &args.generation,
                tokens,
                None::<&ProcessedImages>,
                &args.model.downsample_mode,
                &eos_ids,
                |_, _, _, _| Ok(()),
            )?;
            let raw_text = decode_tokens(&tokenizer, &stats.generated_token_ids)?;
            let text = split_reasoning_text(&raw_text, args.generation.enable_thinking);
            serde_json::to_writer(
                &mut writer,
                &serde_json::json!({
                    "kind": "lmbrrr_benchmark",
                    "model_id": args.model.model_id.as_str(),
                    "revision": args.model.revision.as_str(),
                    "device": format!("{device:?}"),
                    "dtype": format!("{dtype:?}"),
                    "downsample_mode": args.model.downsample_mode.as_str(),
                    "profile": prompt.name.as_str(),
                    "iteration": iteration,
                    "warmup_iterations": args.warmup,
                    "prompt_tokens": stats.prompt_tokens,
                    "generated_tokens": stats.generated_tokens,
                    "total_tokens": stats.total_generated_tokens(),
                    "max_generated_tokens": args.generation.max_new_tokens,
                    "max_total_tokens": stats.prompt_tokens + args.generation.max_new_tokens,
                    "eos_reached": stats.eos_reached,
                    "prefill_seconds": secs(stats.prefill_elapsed),
                    "prefill_tokens_per_second": stats.prefill_tokens_per_second(),
                    "time_to_first_token_seconds": stats.time_to_first_token().map(secs),
                    "decode_time_to_first_token_seconds": stats.first_token_after_prefill.map(secs),
                    "decode_seconds": secs(stats.decode_elapsed),
                    "decode_model_input_tokens": stats.decode_model_tokens(),
                    "decode_model_seconds": secs(stats.decode_model_elapsed),
                    "decode_model_tokens_per_second": stats.decode_model_tokens_per_second(),
                    "decode_non_model_seconds": secs(stats.decode_non_model_elapsed()),
                    "decode_non_model_share": stats.decode_non_model_share(),
                    "sampling_seconds": secs(stats.sampling_elapsed),
                    "sampling_tokens_per_second": stats.sampling_tokens_per_second(),
                    "next_input_seconds": secs(stats.next_input_elapsed),
                    "callback_seconds": secs(stats.callback_elapsed),
                    "decode_bookkeeping_seconds": secs(stats.decode_bookkeeping_elapsed()),
                    "output_tokens_per_second": stats.decode_tokens_per_second(),
                    "decode_tokens_per_second": stats.decode_tokens_per_second(),
                    "steady_state_tokens_per_second": stats.steady_state_tokens_per_second(),
                    "artifact_seconds": secs(bundle.elapsed),
                    "load_seconds": secs(load_elapsed),
                    "tensor_count": bundle.weight_report.tensor_count,
                    "has_lm_head": bundle.weight_report.has_lm_head,
                    "quantized_load": quantized_load_json(&quantized_load),
                    "text": {
                        "raw": text.raw_text,
                        "reasoning": text.reasoning_text,
                        "answer": text.answer_text,
                    },
                    "generation": {
                        "max_new_tokens": args.generation.max_new_tokens,
                        "temperature": args.generation.temperature,
                        "top_p": args.generation.top_p,
                        "top_k": args.generation.top_k,
                        "seed": args.generation.seed,
                        "repeat_penalty": args.generation.repeat_penalty,
                        "repeat_last_n": args.generation.repeat_last_n,
                        "enable_thinking": args.generation.enable_thinking,
                    },
                }),
            )?;
            writer.write_all(b"\n")?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn logits(args: LogitsArgs) -> Result<()> {
    if args.top_k == 0 {
        anyhow::bail!("--top-k must be greater than zero");
    }

    let fixture_text = fs::read_to_string(&args.fixture)
        .with_context(|| format!("read fixture {}", args.fixture.display()))?;
    let fixture: LogitsOracleFixture =
        serde_json::from_str(&fixture_text).context("parse logits oracle fixture")?;
    let cases = fixture
        .cases
        .iter()
        .filter(|case| case.image_count == 0 && case.next_token_logits.is_some())
        .collect::<Vec<_>>();
    if cases.is_empty() {
        anyhow::bail!("fixture contains no text-only logits cases");
    }

    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
    )?;

    let mut rows = Vec::with_capacity(cases.len());
    let mut all_passed = true;
    for case in cases {
        model.clear_cache();
        let input = Tensor::from_slice(&case.token_ids, (1, case.token_ids.len()), &device)?;
        let logits = model.forward(
            &input,
            None::<&ProcessedImages>,
            &args.model.downsample_mode,
            0,
        )?;
        let candle_top = top_k_logits(&logits.squeeze(0)?, args.top_k, &tokenizer)?;
        let expected = case
            .next_token_logits
            .as_ref()
            .context("missing oracle logits")?;
        let expected_token_ids = expected
            .top_token_ids
            .as_ref()
            .context("missing oracle top_token_ids")?;
        let expected_logits = expected
            .top_logits
            .as_ref()
            .context("missing oracle top_logits")?;
        let comparison = compare_top_logits(&candle_top, expected_token_ids, expected_logits);
        all_passed &= comparison.passed;

        rows.push(serde_json::json!({
            "id": case.id.as_str(),
            "user_prompt": case.user_prompt.as_str(),
            "prompt_tokens": case.token_ids.len(),
            "oracle_prompt_tokens": case.prompt_token_count,
            "candle": {
                "top_token_ids": candle_top.iter().map(|item| item.token_id).collect::<Vec<_>>(),
                "top_tokens": candle_top.iter().map(|item| item.token.as_str()).collect::<Vec<_>>(),
                "top_logits": candle_top.iter().map(|item| item.logit).collect::<Vec<_>>(),
            },
            "transformers": {
                "top_token_ids": expected_token_ids,
                "top_logits": expected_logits,
            },
            "comparison": {
                "top1_match": comparison.top1_match,
                "top_k_overlap": comparison.top_k_overlap,
                "top_k_overlap_threshold": comparison.top_k_overlap_threshold,
                "max_abs_shared_logit_delta": comparison.max_abs_shared_logit_delta,
                "shared_logit_deltas": comparison.shared_logit_deltas,
                "passed": comparison.passed,
            }
        }));
    }

    let report = serde_json::json!({
        "kind": "lmbrrr_logits_parity",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "fixture_model_id": fixture.model_id,
        "fixture_revision": fixture.revision,
        "fixture": args.fixture,
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "downsample_mode": args.model.downsample_mode.as_str(),
        "top_k": args.top_k,
        "artifact_seconds": secs(bundle.elapsed),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "passed": all_passed,
        "cases": rows,
    });

    write_json_report(args.output.as_ref(), &report)?;
    if args.fail_on_mismatch && !all_passed {
        anyhow::bail!("Candle logits did not match the Transformers oracle");
    }
    Ok(())
}

fn profile_decode(args: ProfileArgs) -> Result<()> {
    if args.max_new_tokens == 0 {
        anyhow::bail!("--max-new-tokens must be greater than zero");
    }

    let prompt = args
        .prompt
        .clone()
        .unwrap_or_else(|| args.profile.prompt().to_string());
    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let prompt_text = chat_prompt(&prompt, 0, false);
    let tokens = tokenize_prompt(&tokenizer, prompt_text)?;
    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
    )?;
    let profiler = Qwen35Profiler::new();
    model.set_text_profiler(Some(profiler.clone()));
    model.clear_cache();

    profiler.clear();
    let prefill_input = Tensor::from_slice(&tokens, (1, tokens.len()), &device)?;
    let prefill_started = Instant::now();
    let logits = model.forward(
        &prefill_input,
        None::<&ProcessedImages>,
        &args.model.downsample_mode,
        0,
    )?;
    device.synchronize()?;
    let prefill_elapsed = prefill_started.elapsed();
    let prefill_events = profiler.events();
    let (mut next_token, prefill_argmax_elapsed) = argmax_token(&logits, &device)?;

    let mut position = tokens.len();
    let mut decode_events = Vec::new();
    let mut decode_steps = Vec::with_capacity(args.max_new_tokens);
    for step in 0..args.max_new_tokens {
        profiler.clear();
        let input = Tensor::from_slice(&[next_token], (1, 1), &device)?;
        let forward_started = Instant::now();
        let logits = model.forward(
            &input,
            None::<&ProcessedImages>,
            &args.model.downsample_mode,
            position,
        )?;
        device.synchronize()?;
        let forward_elapsed = forward_started.elapsed();
        let events = profiler.events();
        let (sampled, argmax_elapsed) = argmax_token(&logits, &device)?;
        let component_seconds = events.iter().map(|event| event.seconds).sum::<f64>();
        decode_steps.push(serde_json::json!({
            "step": step,
            "input_token_id": next_token,
            "next_token_id": sampled,
            "position": position,
            "model_forward_seconds": secs(forward_elapsed),
            "argmax_seconds": secs(argmax_elapsed),
            "profiled_component_seconds": component_seconds,
            "profiled_event_count": events.len(),
        }));
        decode_events.extend(events);
        next_token = sampled;
        position += 1;
    }
    model.set_text_profiler(None);

    let prefill_profile_seconds = prefill_events
        .iter()
        .map(|event| event.seconds)
        .sum::<f64>();
    let decode_model_forward_seconds = decode_steps
        .iter()
        .filter_map(|step| step["model_forward_seconds"].as_f64())
        .sum::<f64>();
    let decode_argmax_seconds = decode_steps
        .iter()
        .filter_map(|step| step["argmax_seconds"].as_f64())
        .sum::<f64>();
    let decode_profile_seconds = decode_events.iter().map(|event| event.seconds).sum::<f64>();

    let report = serde_json::json!({
        "kind": "lmbrrr_decode_profile",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "downsample_mode": args.model.downsample_mode.as_str(),
        "profile": args.profile.name(),
        "prompt": prompt,
        "prompt_tokens": tokens.len(),
        "decode_steps": args.max_new_tokens,
        "artifact_seconds": secs(bundle.elapsed),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "timing_method": "wall-clock with device.synchronize() around profiled components; intrusive but attributable",
        "prefill": {
            "seconds": secs(prefill_elapsed),
            "tokens_per_second": tokens_per_second(tokens.len(), prefill_elapsed),
            "argmax_seconds": secs(prefill_argmax_elapsed),
            "profiled_component_seconds": prefill_profile_seconds,
            "aggregate": aggregate_profile_events(&prefill_events),
        },
        "decode": {
            "model_forward_seconds": decode_model_forward_seconds,
            "argmax_seconds": decode_argmax_seconds,
            "profiled_component_seconds": decode_profile_seconds,
            "model_forward_tokens_per_second": if decode_model_forward_seconds > 0.0 {
                args.max_new_tokens as f64 / decode_model_forward_seconds
            } else {
                0.0
            },
            "argmax_share_of_forward_plus_argmax": if decode_model_forward_seconds + decode_argmax_seconds > 0.0 {
                decode_argmax_seconds / (decode_model_forward_seconds + decode_argmax_seconds)
            } else {
                0.0
            },
            "aggregate": aggregate_profile_events(&decode_events),
            "by_layer_kind": aggregate_profile_events_by_layer_kind(&decode_events),
            "steps": decode_steps,
            "events": decode_events,
        },
        "kernel_launch_note": "This report counts synchronized component scopes, not Metal command-buffer kernel launches. Use it to rank code-path families; use Xcode/Metal capture for exact launch counts.",
    });

    write_json_report(args.output.as_ref(), &report)
}

fn spec_verify(args: SpecVerifyArgs) -> Result<()> {
    if args.baseline_draft_tokens.is_some() && !args.draft_tokens.is_empty() {
        anyhow::bail!("use either --draft-token or --baseline-draft-tokens, not both");
    }
    if let Some(count) = args.baseline_draft_tokens {
        if count == 0 {
            anyhow::bail!("--baseline-draft-tokens must be greater than zero");
        }
    } else if args.draft_tokens.is_empty() {
        anyhow::bail!("provide at least one --draft-token or use --baseline-draft-tokens");
    }

    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let prompt_text = chat_prompt(&args.prompt, 0, args.enable_thinking);
    let prompt_tokens = tokenize_prompt(&tokenizer, prompt_text)?;
    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
    )?;
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());

    let (mut draft_tokens, draft_source, baseline_tokens) =
        if let Some(count) = args.baseline_draft_tokens {
            let baseline_generation = greedy_generation_args(count + 1, args.enable_thinking);
            let baseline = generate_tokens(
                &mut model,
                &device,
                &baseline_generation,
                &prompt_tokens,
                None::<&ProcessedImages>,
                &args.model.downsample_mode,
                &eos_ids,
                |_, _, _, _| Ok(()),
            )?;
            if baseline.generated_token_ids.len() < count + 1 {
                anyhow::bail!(
                    "baseline generation produced {} tokens before EOS; need {}",
                    baseline.generated_token_ids.len(),
                    count + 1
                );
            }
            (
                baseline.generated_token_ids[..count].to_vec(),
                "baseline".to_string(),
                Some(baseline.generated_token_ids),
            )
        } else {
            (args.draft_tokens.clone(), "explicit".to_string(), None)
        };

    let corruption = if let Some(index) = args.corrupt_draft_at {
        Some(corrupt_draft_token(
            &mut draft_tokens,
            index,
            bundle.config.text_config.vocab_size,
        )?)
    } else {
        None
    };

    let confidence_schedule = apply_confidence_schedule(
        &mut draft_tokens,
        &args.draft_confidences,
        args.schedule_confidence_threshold,
    )?;

    let stats = verify_greedy_draft(
        &mut model,
        &device,
        &prompt_tokens,
        &draft_tokens,
        &args.model.downsample_mode,
    )?;
    let analysis = &stats.analysis;
    let accepted_token_ids = draft_tokens[..analysis.accepted_tokens].to_vec();
    let rejected_token_ids = analysis
        .first_rejected_index
        .map(|idx| draft_tokens[idx..].to_vec())
        .unwrap_or_default();
    let baseline_prefix_match = baseline_tokens.as_ref().map(|tokens| {
        tokens
            .get(..analysis.reconstructed_token_ids.len())
            .map(|prefix| prefix == analysis.reconstructed_token_ids.as_slice())
            .unwrap_or(false)
    });
    let expected_rejection_index = corruption
        .as_ref()
        .and_then(|corruption| (draft_source == "baseline").then_some(corruption.index));
    let rejection_matched_expectation =
        expected_rejection_index.map(|expected| analysis.first_rejected_index == Some(expected));

    let positions = analysis
        .positions
        .iter()
        .map(|position| {
            serde_json::json!({
                "index": position.index,
                "draft_token_id": position.draft_token_id,
                "target_token_id": position.target_token_id,
                "draft_token": decode_token_lossy(&tokenizer, position.draft_token_id),
                "target_token": decode_token_lossy(&tokenizer, position.target_token_id),
                "token_match": position.token_match,
                "accepted": position.accepted,
                "first_rejected": position.first_rejected,
            })
        })
        .collect::<Vec<_>>();

    let report = serde_json::json!({
        "kind": "lmbrrr_greedy_spec_verify",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "downsample_mode": args.model.downsample_mode.as_str(),
        "enable_thinking": args.enable_thinking,
        "artifact_seconds": secs(bundle.elapsed),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "draft_source": draft_source,
        "prompt": args.prompt.as_str(),
        "prompt_tokens": prompt_tokens.len(),
        "draft_tokens": draft_tokens.len(),
        "confidence_schedule": confidence_schedule_json(&confidence_schedule),
        "verified_tokens": analysis.verified_tokens(),
        "accepted_tokens": analysis.accepted_tokens,
        "bonus_tokens": analysis.bonus_tokens(),
        "accepted_length": analysis.accepted_length(),
        "acceptance_rate": analysis.acceptance_rate(),
        "first_rejected_index": analysis.first_rejected_index,
        "verifier_waste_tokens": analysis.verifier_waste_tokens(),
        "verifier_waste_share": analysis.verifier_waste_share(),
        "prefill_seconds": secs(stats.prefill_elapsed),
        "prefill_tokens_per_second": tokens_per_second(prompt_tokens.len(), stats.prefill_elapsed),
        "verify_seconds": secs(stats.verify_elapsed),
        "verify_tokens_per_second": tokens_per_second(draft_tokens.len(), stats.verify_elapsed),
        "argmax_seconds": secs(stats.argmax_elapsed),
        "round_seconds": secs(stats.prefill_elapsed + stats.verify_elapsed + stats.argmax_elapsed),
        "draft_token_ids": &draft_tokens,
        "target_token_ids": &stats.target_token_ids,
        "accepted_token_ids": &accepted_token_ids,
        "rejected_token_ids": rejected_token_ids,
        "bonus_token_id": analysis.bonus_token_id,
        "bonus_token": decode_token_lossy(&tokenizer, analysis.bonus_token_id),
        "reconstructed_token_ids": &analysis.reconstructed_token_ids,
        "draft_text": decode_tokens(&tokenizer, &draft_tokens)?,
        "accepted_text": decode_tokens(&tokenizer, &accepted_token_ids)?,
        "reconstructed_text": decode_tokens(&tokenizer, &analysis.reconstructed_token_ids)?,
        "baseline_token_ids": baseline_tokens,
        "baseline_prefix_match": baseline_prefix_match,
        "expected_rejection_index": expected_rejection_index,
        "rejection_matched_expectation": rejection_matched_expectation,
        "corruption": corruption.map(|corruption| serde_json::json!({
            "index": corruption.index,
            "original_token_id": corruption.original_token_id,
            "corrupted_token_id": corruption.corrupted_token_id,
            "original_token": decode_token_lossy(&tokenizer, corruption.original_token_id),
            "corrupted_token": decode_token_lossy(&tokenizer, corruption.corrupted_token_id),
        })),
        "positions": positions,
    });

    let failed_expectation = baseline_prefix_match == Some(false)
        || rejection_matched_expectation == Some(false)
        || (args.fail_on_mismatch && analysis.first_rejected_index.is_some());
    write_json_report(args.output.as_ref(), &report)?;
    if failed_expectation {
        anyhow::bail!("speculative verifier expectation failed");
    }
    Ok(())
}

fn trace_hidden_states(args: TraceArgs) -> Result<()> {
    if args.max_new_tokens == 0 {
        anyhow::bail!("--max-new-tokens must be greater than zero");
    }
    if args.top_k_logits == 0 {
        anyhow::bail!("--top-k-logits must be greater than zero");
    }

    let bundle = resolve_artifacts(&args.model)?;
    let capture_layers = trace_capture_layers(
        &args.capture_layers,
        bundle.config.text_config.num_hidden_layers,
    )?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let prompt_text = chat_prompt(&args.prompt, 0, args.enable_thinking);
    let prompt_tokens = tokenize_prompt(&tokenizer, prompt_text)?;
    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
    )?;
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());
    let trace_recorder = Qwen35TraceRecorder::new(capture_layers.clone());
    model.set_text_trace_recorder(Some(trace_recorder.clone()));
    model.clear_cache();

    let mut generated_token_ids = Vec::with_capacity(args.max_new_tokens);
    let mut steps = Vec::with_capacity(args.max_new_tokens);
    let mut total_forward_elapsed = Duration::ZERO;
    let mut total_argmax_elapsed = Duration::ZERO;
    let mut total_logits_elapsed = Duration::ZERO;
    let mut eos_reached = false;

    trace_recorder.clear();
    let prompt_input = Tensor::from_slice(&prompt_tokens, (1, prompt_tokens.len()), &device)?;
    let forward_start = Instant::now();
    let mut logits = model.forward(
        &prompt_input,
        None::<&ProcessedImages>,
        &args.model.downsample_mode,
        0,
    )?;
    device.synchronize()?;
    let mut forward_elapsed = forward_start.elapsed();
    let mut hidden_states = trace_recorder.take();
    let mut phase = "prefill";
    let mut context_position = prompt_tokens.len() - 1;
    let mut offset = 0usize;
    let mut seq_len = prompt_tokens.len();

    for step_index in 0..args.max_new_tokens {
        total_forward_elapsed += forward_elapsed;
        let (next_token, argmax_elapsed) = argmax_token(&logits, &device)?;
        total_argmax_elapsed += argmax_elapsed;
        let logits_start = Instant::now();
        let top_logits = top_k_logits(&logits.squeeze(0)?, args.top_k_logits, &tokenizer)?;
        let logits_elapsed = logits_start.elapsed();
        total_logits_elapsed += logits_elapsed;

        let step_eos = eos_ids.contains(&next_token);
        steps.push(serde_json::json!({
            "step": step_index,
            "phase": phase,
            "context_position": context_position,
            "offset": offset,
            "seq_len": seq_len,
            "target_token_id": next_token,
            "target_token": decode_token_lossy(&tokenizer, next_token),
            "eos": step_eos,
            "model_forward_seconds": secs(forward_elapsed),
            "argmax_seconds": secs(argmax_elapsed),
            "logits_top_k_seconds": secs(logits_elapsed),
            "top_logits": top_logits_json(&top_logits),
            "hidden_state_count": hidden_states.len(),
            "hidden_states": hidden_states,
        }));

        if step_eos {
            eos_reached = true;
            break;
        }
        generated_token_ids.push(next_token);
        if generated_token_ids.len() == args.max_new_tokens {
            break;
        }

        phase = "decode";
        context_position = prompt_tokens.len() + generated_token_ids.len() - 1;
        offset = context_position;
        seq_len = 1;
        trace_recorder.clear();
        let input = Tensor::from_slice(&[next_token], (1, 1), &device)?;
        let forward_start = Instant::now();
        logits = model.forward(
            &input,
            None::<&ProcessedImages>,
            &args.model.downsample_mode,
            offset,
        )?;
        device.synchronize()?;
        forward_elapsed = forward_start.elapsed();
        hidden_states = trace_recorder.take();
    }
    model.set_text_trace_recorder(None);

    let prompt_token_count = prompt_tokens.len();
    let generated_token_count = generated_token_ids.len();
    let generated_text = decode_tokens(&tokenizer, &generated_token_ids)?;
    let report = serde_json::json!({
        "kind": "lmbrrr_hidden_state_trace",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "downsample_mode": args.model.downsample_mode.as_str(),
        "enable_thinking": args.enable_thinking,
        "artifact_seconds": secs(bundle.elapsed),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "prompt": args.prompt.as_str(),
        "prompt_token_ids": &prompt_tokens,
        "prompt_tokens": prompt_token_count,
        "max_new_tokens": args.max_new_tokens,
        "generated_token_ids": &generated_token_ids,
        "generated_tokens": generated_token_count,
        "generated_text": generated_text,
        "eos_reached": eos_reached,
        "capture_layers": capture_layers,
        "top_k_logits": args.top_k_logits,
        "timing": {
            "model_forward_seconds": secs(total_forward_elapsed),
            "model_forward_tokens_per_second": tokens_per_second(steps.len(), total_forward_elapsed),
            "argmax_seconds": secs(total_argmax_elapsed),
            "logits_top_k_seconds": secs(total_logits_elapsed),
        },
        "steps": steps,
    });

    write_json_report(args.output.as_ref(), &report)
}

fn quant_sensitivity(args: QuantSensitivityArgs) -> Result<()> {
    if args.top_k_logits == 0 {
        anyhow::bail!("--top-k-logits must be greater than zero");
    }
    let formats = if args.candidate_quants.is_empty() {
        vec![
            QuantFormat::SymmetricInt4,
            QuantFormat::SymmetricInt5,
            QuantFormat::SymmetricInt8,
        ]
    } else {
        args.candidate_quants
            .iter()
            .map(|format| format.resolve())
            .collect::<Vec<_>>()
    };

    let calibration_rows = read_calibration_jsonl(&args.calibration)?;
    let text_rows = calibration_rows
        .iter()
        .filter(|row| row.modality == "text")
        .take(args.max_cases.unwrap_or(usize::MAX))
        .collect::<Vec<_>>();
    if text_rows.is_empty() {
        anyhow::bail!(
            "calibration file {} contains no text rows",
            args.calibration.display()
        );
    }

    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
    )?;

    let baseline_started = Instant::now();
    let mut baseline_cases = Vec::with_capacity(text_rows.len());
    for row in text_rows {
        baseline_cases.push(run_quant_baseline_case(
            &mut model,
            &device,
            &tokenizer,
            row,
            &args.model.downsample_mode,
            args.top_k_logits,
        )?);
    }
    let baseline_elapsed = baseline_started.elapsed();

    let weight_report = score_weight_sensitivity(
        &bundle.artifacts.weights,
        &formats,
        args.max_modules,
        args.include_protected,
    )?;

    let report = serde_json::json!({
        "kind": "lmbrrr_quantization_sensitivity",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "downsample_mode": args.model.downsample_mode.as_str(),
        "calibration_set": args.calibration,
        "calibration": aggregate_calibration(&calibration_rows),
        "candidate_quants": formats.iter().map(|format| format.name()).collect::<Vec<_>>(),
        "include_protected": args.include_protected,
        "artifact_seconds": secs(bundle.elapsed),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "tensor_count": bundle.weight_report.tensor_count,
        "has_lm_head": bundle.weight_report.has_lm_head,
        "baseline": {
            "status": "measured",
            "rows_measured": baseline_cases.len(),
            "text_rows_available": calibration_rows.iter().filter(|row| row.modality == "text").count(),
            "seconds": secs(baseline_elapsed),
            "top_k_logits": args.top_k_logits,
            "cases": baseline_cases,
        },
        "weights": weight_report,
        "measurement_limits": {
            "activation_error": "not collected yet because MiniCPM module activation hooks are not implemented",
            "per_module_logit_drift": "not collected yet because this command does not run perturbed quantized module forwards",
            "latency_delta": "weight quantization simulation is timed; runtime matmul latency awaits quantized loader/kernel tickets",
        },
    });

    write_json_report(args.output.as_ref(), &report)
}

fn quant_convert(args: QuantConvertArgs) -> Result<()> {
    let bundle = resolve_artifacts(&args.model)?;
    let manifest = convert_mixed_precision(ConversionOptions {
        model_id: args.model.model_id.clone(),
        revision: args.model.revision.clone(),
        policy: args.policy.resolve(),
        source_weights: bundle.artifacts.weights.clone(),
        sensitivity_artifact: args.sensitivity.clone(),
        output_dir: args.output_dir.clone(),
        max_tensors: args.max_tensors,
        manifest_only: args.manifest_only,
    })?;
    let manifest_path = args.output_dir.join("manifest.json");
    let summary = serde_json::json!({
        "kind": "lmbrrr_quant_convert_complete",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "policy": args.policy.resolve().name(),
        "manifest": manifest_path,
        "artifact_seconds": secs(bundle.elapsed),
        "manifest_only": args.manifest_only,
        "summary": manifest["summary"].clone(),
    });
    write_json_report(None, &summary)
}

fn quant_matmul_bench(args: QuantMatmulBenchArgs) -> Result<()> {
    if args.chunk_tokens == 0 {
        anyhow::bail!("--chunk-tokens must be greater than zero");
    }
    if args.iterations == 0 {
        anyhow::bail!("--iterations must be greater than zero");
    }

    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let shapes = quant_matmul_shapes(&bundle.config, args.include_lm_head);
    let activation_dtypes = [DType::F32, DType::F16, DType::BF16];
    let quant_dtypes = [
        GgmlDType::Q8_0,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ];

    let mut rows = Vec::new();
    for shape in shapes {
        let weight_values = deterministic_values(shape.out_dim * shape.in_dim, 0.013);
        let weight_cpu =
            Tensor::from_vec(weight_values, (shape.out_dim, shape.in_dim), &Device::Cpu)?;
        for mode in [MatmulMode::Decode, MatmulMode::Prefill] {
            let tokens = match mode {
                MatmulMode::Decode => 1,
                MatmulMode::Prefill => args.chunk_tokens,
            };
            let input_values = deterministic_values(tokens * shape.in_dim, 0.017);
            let input_cpu =
                Tensor::from_vec(input_values, (1, tokens, shape.in_dim), &Device::Cpu)?;

            for activation_dtype in activation_dtypes {
                rows.push(bench_dense_matmul(
                    &shape,
                    mode,
                    activation_dtype,
                    &weight_cpu,
                    &input_cpu,
                    &device,
                    args.warmup,
                    args.iterations,
                ));
            }

            for quant_dtype in quant_dtypes {
                for activation_dtype in activation_dtypes {
                    rows.push(bench_quant_matmul(
                        &shape,
                        mode,
                        quant_dtype,
                        activation_dtype,
                        &weight_cpu,
                        &input_cpu,
                        &device,
                        args.warmup,
                        args.iterations,
                    ));
                }
            }
        }
    }

    let report = serde_json::json!({
        "kind": "lmbrrr_quantized_matmul_benchmark",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "chunk_tokens": args.chunk_tokens,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "include_lm_head": args.include_lm_head,
        "artifact_seconds": secs(bundle.elapsed),
        "rows": rows,
        "note": "Dense baselines use generated weights with Candle matmul. Quantized rows use Candle QTensor::quantize_onto and QMatMul::forward; failures are recorded because activation dtype support is part of the measurement.",
    });
    write_json_report(args.output.as_ref(), &report)
}

fn quant_quality(args: QuantQualityArgs) -> Result<()> {
    if !is_greedy_generation(&args.generation) {
        anyhow::bail!("quant-quality requires greedy generation; leave --temperature at 0");
    }
    validate_quality_threshold("min-prefix-ratio", args.min_prefix_ratio, 0.0, 1.0)?;
    validate_quality_threshold("min-token-jaccard", args.min_token_jaccard, 0.0, 1.0)?;
    validate_quality_threshold("min-lexical-jaccard", args.min_lexical_jaccard, 0.0, 1.0)?;
    validate_quality_threshold(
        "max-length-ratio-delta",
        args.max_length_ratio_delta,
        0.0,
        10.0,
    )?;

    let thresholds = QualityThresholds {
        min_prefix_ratio: args.min_prefix_ratio,
        min_token_jaccard: args.min_token_jaccard,
        min_lexical_jaccard: args.min_lexical_jaccard,
        max_length_ratio_delta: args.max_length_ratio_delta,
    };
    let calibration_rows = read_calibration_jsonl(&args.calibration)?;
    let mut text_rows = calibration_rows
        .iter()
        .filter(|row| row.modality == "text")
        .filter(|row| args.case_ids.is_empty() || args.case_ids.contains(&row.id))
        .collect::<Vec<_>>();
    if let Some(max_cases) = args.max_cases {
        text_rows.truncate(max_cases);
    }
    if text_rows.is_empty() {
        anyhow::bail!(
            "calibration file {} contains no selected text rows",
            args.calibration.display()
        );
    }

    for manifest in [
        &args.q8_manifest,
        &args.q4_mlp_manifest,
        &args.q4_text_safe_manifest,
        &args.mixed_manifest,
    ] {
        if !manifest.exists() {
            anyhow::bail!(
                "quantized manifest {} does not exist; run quant-convert for this policy first",
                manifest.display()
            );
        }
    }

    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());
    let policy_specs = [
        ("dense", None::<&PathBuf>),
        ("q8-text-linears", Some(&args.q8_manifest)),
        ("q4k-mlp-only", Some(&args.q4_mlp_manifest)),
        ("q4k-text-safe", Some(&args.q4_text_safe_manifest)),
        ("q4k-mlp-q8-text", Some(&args.mixed_manifest)),
    ];

    let mut policy_runs = Vec::new();
    for (label, manifest) in policy_specs {
        eprintln!("running quant-quality policy {label}");
        let started = Instant::now();
        let (generations, load_elapsed, quantized_load) = run_quant_quality_policy(
            &bundle, &device, dtype, manifest, &text_rows, &tokenizer, &eos_ids, &args,
        )
        .with_context(|| format!("run quant-quality policy {label}"))?;
        policy_runs.push(QuantQualityPolicyRun {
            label: label.to_string(),
            manifest: manifest.cloned(),
            load_elapsed,
            run_elapsed: started.elapsed(),
            quantized_load,
            generations,
        });
    }

    let dense_generations = policy_runs
        .first()
        .context("missing dense quant-quality generations")?;
    let mut cases = Vec::new();
    let mut summaries = HashMap::<String, Vec<QualityComparison>>::new();
    for (case_index, row) in text_rows.iter().enumerate() {
        let dense_generation = dense_generations
            .generations
            .get(case_index)
            .context("dense generation count did not match selected rows")?;
        let mut candidates = Vec::new();
        for policy_run in policy_runs.iter().skip(1) {
            let candidate_generation = policy_run
                .generations
                .get(case_index)
                .context("candidate generation count did not match selected rows")?;
            let comparison = compare_quality_outputs(
                &dense_generation.stats.generated_token_ids,
                &dense_generation.raw_text,
                &candidate_generation.stats.generated_token_ids,
                &candidate_generation.raw_text,
                &thresholds,
            );
            summaries
                .entry(policy_run.label.clone())
                .or_default()
                .push(comparison.clone());
            candidates.push(serde_json::json!({
                "policy": policy_run.label,
                "generation": quant_quality_generation_json(candidate_generation),
                "comparison": quality_comparison_json(&comparison),
            }));
        }
        cases.push(serde_json::json!({
            "id": row.id,
            "category": row.category,
            "expected_behavior": row.expected_behavior,
            "enable_thinking": row.enable_thinking,
            "prompt_tokens": row.token_ids.len(),
            "max_new_tokens": quality_generation_args(&args.generation, row).max_new_tokens,
            "dense": quant_quality_generation_json(dense_generation),
            "candidates": candidates,
        }));
    }

    let mut policy_summaries = summaries
        .iter()
        .map(|(policy, comparisons)| quality_summary_json(policy, comparisons))
        .collect::<Vec<_>>();
    policy_summaries.sort_by(|left, right| {
        left["policy"]
            .as_str()
            .unwrap_or_default()
            .cmp(right["policy"].as_str().unwrap_or_default())
    });
    let passed = policy_summaries
        .iter()
        .all(|summary| summary["passed"].as_bool().unwrap_or(false));

    let report = serde_json::json!({
        "kind": "lmbrrr_quantization_quality_eval",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "downsample_mode": args.model.downsample_mode.as_str(),
        "calibration_set": args.calibration,
        "selected_cases": text_rows.len(),
        "artifact_seconds": secs(bundle.elapsed),
        "thresholds": {
            "min_prefix_ratio": thresholds.min_prefix_ratio,
            "min_token_jaccard": thresholds.min_token_jaccard,
            "min_lexical_jaccard": thresholds.min_lexical_jaccard,
            "max_length_ratio_delta": thresholds.max_length_ratio_delta,
        },
        "generation": {
            "max_new_tokens_cap": args.generation.max_new_tokens,
            "temperature": args.generation.temperature,
            "top_p": args.generation.top_p,
            "top_k": args.generation.top_k,
            "seed": args.generation.seed,
            "repeat_penalty": args.generation.repeat_penalty,
            "repeat_last_n": args.generation.repeat_last_n,
        },
        "policy_runs": policy_runs.iter().map(|run| {
            serde_json::json!({
                "label": run.label,
                "manifest": run.manifest,
                "load_seconds": secs(run.load_elapsed),
                "run_seconds": secs(run.run_elapsed),
                "load": quantized_load_json(&run.quantized_load),
            })
        }).collect::<Vec<_>>(),
        "policy_summaries": policy_summaries,
        "passed": passed,
        "cases": cases,
        "gate_note": "A candidate passes a case when it exactly matches dense tokens or meets every configured prefix, token-overlap, lexical-overlap, and length-delta threshold.",
    });
    if args.fail_on_gate && !passed {
        write_json_report(args.output.as_ref(), &report)?;
        anyhow::bail!("one or more quantization quality gates failed");
    }
    write_json_report(args.output.as_ref(), &report)
}

fn run_quant_quality_policy(
    bundle: &ArtifactBundle,
    device: &Device,
    dtype: DType,
    quantized_manifest: Option<&PathBuf>,
    rows: &[&CalibrationRow],
    tokenizer: &Tokenizer,
    eos_ids: &[u32],
    args: &QuantQualityArgs,
) -> Result<(
    Vec<QuantQualityGeneration>,
    Duration,
    Option<QuantizedLoadStats>,
)> {
    let (mut model, load_elapsed, quantized_load) =
        load_model_with_optional_quantization(bundle, dtype, device, quantized_manifest)?;
    let mut generations = Vec::with_capacity(rows.len());
    for row in rows {
        let generation = quality_generation_args(&args.generation, row);
        let stats = generate_tokens(
            &mut model,
            device,
            &generation,
            &row.token_ids,
            None::<&ProcessedImages>,
            &args.model.downsample_mode,
            eos_ids,
            |_, _, _, _| Ok(()),
        )?;
        let raw_text = decode_tokens(tokenizer, &stats.generated_token_ids)?;
        let parts = split_reasoning_text(&raw_text, row.enable_thinking);
        generations.push(QuantQualityGeneration {
            stats,
            raw_text,
            reasoning_text: parts.reasoning_text,
            answer_text: parts.answer_text,
        });
    }
    Ok((generations, load_elapsed, quantized_load))
}

fn quality_generation_args(base: &GenerationArgs, row: &CalibrationRow) -> GenerationArgs {
    let mut generation = base.clone();
    generation.enable_thinking = row.enable_thinking;
    if let Some(row_max_new_tokens) = row.max_new_tokens {
        generation.max_new_tokens = generation.max_new_tokens.min(row_max_new_tokens);
    }
    generation
}

fn quant_quality_generation_json(generation: &QuantQualityGeneration) -> serde_json::Value {
    serde_json::json!({
        "generated_tokens": generation.stats.generated_tokens,
        "generated_token_ids": generation.stats.generated_token_ids,
        "eos_reached": generation.stats.eos_reached,
        "prefill_seconds": secs(generation.stats.prefill_elapsed),
        "prefill_tokens_per_second": generation.stats.prefill_tokens_per_second(),
        "decode_seconds": secs(generation.stats.decode_elapsed),
        "decode_model_seconds": secs(generation.stats.decode_model_elapsed),
        "decode_tokens_per_second": generation.stats.decode_tokens_per_second(),
        "steady_state_tokens_per_second": generation.stats.steady_state_tokens_per_second(),
        "text": {
            "raw": generation.raw_text,
            "reasoning": generation.reasoning_text,
            "answer": generation.answer_text,
        },
    })
}

fn compare_quality_outputs(
    dense_token_ids: &[u32],
    dense_text: &str,
    candidate_token_ids: &[u32],
    candidate_text: &str,
    thresholds: &QualityThresholds,
) -> QualityComparison {
    let exact_token_match = dense_token_ids == candidate_token_ids;
    let common_prefix_tokens = common_prefix_len(dense_token_ids, candidate_token_ids);
    let divergence_index = (!exact_token_match).then_some(common_prefix_tokens);
    let dense_len = dense_token_ids.len().max(1);
    let prefix_ratio = common_prefix_tokens as f64 / dense_len as f64;
    let token_jaccard = token_multiset_jaccard(dense_token_ids, candidate_token_ids);
    let lexical_jaccard = lexical_multiset_jaccard(dense_text, candidate_text);
    let length_ratio = if dense_token_ids.is_empty() {
        if candidate_token_ids.is_empty() {
            1.0
        } else {
            f64::INFINITY
        }
    } else {
        candidate_token_ids.len() as f64 / dense_token_ids.len() as f64
    };
    let length_ratio_delta = (length_ratio - 1.0).abs();
    let passed_gate = exact_token_match
        || (prefix_ratio >= thresholds.min_prefix_ratio
            && token_jaccard >= thresholds.min_token_jaccard
            && lexical_jaccard >= thresholds.min_lexical_jaccard
            && length_ratio_delta <= thresholds.max_length_ratio_delta);
    QualityComparison {
        exact_token_match,
        common_prefix_tokens,
        divergence_index,
        prefix_ratio,
        token_jaccard,
        lexical_jaccard,
        length_ratio,
        length_ratio_delta,
        passed_gate,
    }
}

fn quality_comparison_json(comparison: &QualityComparison) -> serde_json::Value {
    serde_json::json!({
        "exact_token_match": comparison.exact_token_match,
        "common_prefix_tokens": comparison.common_prefix_tokens,
        "divergence_index": comparison.divergence_index,
        "prefix_ratio": comparison.prefix_ratio,
        "token_jaccard": comparison.token_jaccard,
        "lexical_jaccard": comparison.lexical_jaccard,
        "length_ratio": comparison.length_ratio,
        "length_ratio_delta": comparison.length_ratio_delta,
        "passed_gate": comparison.passed_gate,
    })
}

fn quality_summary_json(policy: &str, comparisons: &[QualityComparison]) -> serde_json::Value {
    let cases = comparisons.len();
    let exact_token_matches = comparisons
        .iter()
        .filter(|comparison| comparison.exact_token_match)
        .count();
    let passed_cases = comparisons
        .iter()
        .filter(|comparison| comparison.passed_gate)
        .count();
    serde_json::json!({
        "policy": policy,
        "cases": cases,
        "exact_token_matches": exact_token_matches,
        "passed_cases": passed_cases,
        "failed_cases": cases.saturating_sub(passed_cases),
        "mean_prefix_ratio": mean(comparisons.iter().map(|comparison| comparison.prefix_ratio)),
        "mean_token_jaccard": mean(comparisons.iter().map(|comparison| comparison.token_jaccard)),
        "mean_lexical_jaccard": mean(comparisons.iter().map(|comparison| comparison.lexical_jaccard)),
        "mean_length_ratio_delta": mean(comparisons.iter().map(|comparison| comparison.length_ratio_delta)),
        "passed": cases > 0 && passed_cases == cases,
    })
}

fn validate_quality_threshold(name: &str, value: f64, min: f64, max: f64) -> Result<()> {
    if !value.is_finite() || value < min || value > max {
        anyhow::bail!("{name} must be finite and within [{min}, {max}], got {value}");
    }
    Ok(())
}

fn common_prefix_len<T: Eq>(left: &[T], right: &[T]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(left, right)| left == right)
        .count()
}

fn token_multiset_jaccard(left: &[u32], right: &[u32]) -> f64 {
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    let mut left_counts = HashMap::<u32, usize>::new();
    let mut right_counts = HashMap::<u32, usize>::new();
    for token in left {
        *left_counts.entry(*token).or_default() += 1;
    }
    for token in right {
        *right_counts.entry(*token).or_default() += 1;
    }
    let mut intersection = 0usize;
    let mut union = 0usize;
    for (token, left_count) in &left_counts {
        let right_count = right_counts.get(token).copied().unwrap_or(0);
        intersection += (*left_count).min(right_count);
        union += (*left_count).max(right_count);
    }
    for (token, right_count) in &right_counts {
        if !left_counts.contains_key(token) {
            union += *right_count;
        }
    }
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

fn lexical_multiset_jaccard(left: &str, right: &str) -> f64 {
    let left_terms = lexical_terms(left);
    let right_terms = lexical_terms(right);
    if left_terms.is_empty() && right_terms.is_empty() {
        return 1.0;
    }
    let mut left_counts = HashMap::<String, usize>::new();
    let mut right_counts = HashMap::<String, usize>::new();
    for term in left_terms {
        *left_counts.entry(term).or_default() += 1;
    }
    for term in right_terms {
        *right_counts.entry(term).or_default() += 1;
    }
    let mut intersection = 0usize;
    let mut union = 0usize;
    for (term, left_count) in &left_counts {
        let right_count = right_counts.get(term).copied().unwrap_or(0);
        intersection += (*left_count).min(right_count);
        union += (*left_count).max(right_count);
    }
    for (term, right_count) in &right_counts {
        if !left_counts.contains_key(term) {
            union += *right_count;
        }
    }
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

fn lexical_terms(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            current.extend(ch.to_lowercase());
        } else if !current.is_empty() {
            terms.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        terms.push(current);
    }
    terms
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for value in values {
        total += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

#[derive(Copy, Clone, Debug)]
enum MatmulMode {
    Decode,
    Prefill,
}

impl MatmulMode {
    fn name(self) -> &'static str {
        match self {
            Self::Decode => "decode_mv",
            Self::Prefill => "prefill_mm",
        }
    }
}

#[derive(Clone, Debug)]
struct MatmulShape {
    name: &'static str,
    family: &'static str,
    in_dim: usize,
    out_dim: usize,
}

fn quant_matmul_shapes(cfg: &MiniCpmConfig, include_lm_head: bool) -> Vec<MatmulShape> {
    let text = &cfg.text_config;
    let key_dim = text.linear_key_head_dim * text.linear_num_key_heads;
    let value_dim = text.linear_value_head_dim * text.linear_num_value_heads;
    let conv_dim = key_dim * 2 + value_dim;
    let mut shapes = vec![
        MatmulShape {
            name: "deltanet_in_proj_qkv",
            family: "text.deltanet",
            in_dim: text.hidden_size,
            out_dim: conv_dim,
        },
        MatmulShape {
            name: "deltanet_out_proj",
            family: "text.deltanet",
            in_dim: value_dim,
            out_dim: text.hidden_size,
        },
        MatmulShape {
            name: "mlp_up_or_gate_proj",
            family: "text.mlp",
            in_dim: text.hidden_size,
            out_dim: text.intermediate_size,
        },
        MatmulShape {
            name: "mlp_down_proj",
            family: "text.mlp",
            in_dim: text.intermediate_size,
            out_dim: text.hidden_size,
        },
        MatmulShape {
            name: "full_attention_q_proj",
            family: "text.full_attention",
            in_dim: text.hidden_size,
            out_dim: text.num_attention_heads * text.head_dim * 2,
        },
        MatmulShape {
            name: "full_attention_o_proj",
            family: "text.full_attention",
            in_dim: text.num_attention_heads * text.head_dim,
            out_dim: text.hidden_size,
        },
    ];
    if include_lm_head {
        shapes.push(MatmulShape {
            name: "lm_head",
            family: "text.lm_head",
            in_dim: text.hidden_size,
            out_dim: text.vocab_size,
        });
    }
    shapes
}

fn bench_dense_matmul(
    shape: &MatmulShape,
    mode: MatmulMode,
    activation_dtype: DType,
    weight_cpu: &Tensor,
    input_cpu: &Tensor,
    device: &Device,
    warmup: usize,
    iterations: usize,
) -> serde_json::Value {
    let result = (|| -> Result<(Duration, Duration)> {
        let prepare_started = Instant::now();
        let weight = weight_cpu.to_device(device)?.to_dtype(activation_dtype)?;
        let input = input_cpu.to_device(device)?.to_dtype(activation_dtype)?;
        device.synchronize()?;
        let prepare_elapsed = prepare_started.elapsed();
        let elapsed = time_iterations(device, warmup, iterations, || {
            let w = weight.t()?;
            let tokens = input.dim(1)?;
            Ok(input
                .reshape((tokens, shape.in_dim))?
                .matmul(&w)?
                .reshape((1, tokens, shape.out_dim))?)
        })?;
        Ok((prepare_elapsed, elapsed))
    })();
    matmul_bench_row(
        shape,
        mode,
        "dense",
        Some(format!("{activation_dtype:?}")),
        activation_dtype,
        None,
        result,
        iterations,
    )
}

fn bench_quant_matmul(
    shape: &MatmulShape,
    mode: MatmulMode,
    quant_dtype: GgmlDType,
    activation_dtype: DType,
    weight_cpu: &Tensor,
    input_cpu: &Tensor,
    device: &Device,
    warmup: usize,
    iterations: usize,
) -> serde_json::Value {
    let result = (|| -> Result<(Duration, Duration)> {
        let prepare_started = Instant::now();
        let qweight = QTensor::quantize_onto(weight_cpu, quant_dtype, device)?;
        let qmatmul = QMatMul::from_qtensor(qweight)?;
        let input = input_cpu.to_device(device)?.to_dtype(activation_dtype)?;
        device.synchronize()?;
        let prepare_elapsed = prepare_started.elapsed();
        let elapsed = if activation_dtype == DType::F32 {
            time_iterations(device, warmup, iterations, || Ok(qmatmul.forward(&input)?))?
        } else {
            time_iterations(device, warmup, iterations, || {
                let input = input.to_dtype(DType::F32)?;
                Ok(qmatmul.forward(&input)?)
            })?
        };
        Ok((prepare_elapsed, elapsed))
    })();
    matmul_bench_row(
        shape,
        mode,
        "quantized",
        Some(format!("{quant_dtype:?}")),
        activation_dtype,
        (activation_dtype != DType::F32).then_some("to_f32"),
        result,
        iterations,
    )
}

fn matmul_bench_row(
    shape: &MatmulShape,
    mode: MatmulMode,
    backend: &str,
    weight_dtype: Option<String>,
    activation_dtype: DType,
    activation_cast: Option<&str>,
    result: Result<(Duration, Duration)>,
    iterations: usize,
) -> serde_json::Value {
    match result {
        Ok((prepare_elapsed, elapsed)) => serde_json::json!({
            "shape": shape.name,
            "family": shape.family,
            "mode": mode.name(),
            "backend": backend,
            "weight_dtype": weight_dtype,
            "activation_dtype": format!("{activation_dtype:?}"),
            "activation_cast": activation_cast,
            "in_dim": shape.in_dim,
            "out_dim": shape.out_dim,
            "iterations": iterations,
            "prepare_seconds": secs(prepare_elapsed),
            "elapsed_seconds": secs(elapsed),
            "seconds_per_iteration": secs(elapsed) / iterations as f64,
            "tokens_per_second": tokens_per_second(match mode {
                MatmulMode::Decode => iterations,
                MatmulMode::Prefill => iterations,
            }, elapsed),
            "ok": true,
        }),
        Err(err) => serde_json::json!({
            "shape": shape.name,
            "family": shape.family,
            "mode": mode.name(),
            "backend": backend,
            "weight_dtype": weight_dtype,
            "activation_dtype": format!("{activation_dtype:?}"),
            "activation_cast": activation_cast,
            "in_dim": shape.in_dim,
            "out_dim": shape.out_dim,
            "iterations": iterations,
            "ok": false,
            "error": err.to_string(),
        }),
    }
}

fn time_iterations(
    device: &Device,
    warmup: usize,
    iterations: usize,
    mut f: impl FnMut() -> Result<Tensor>,
) -> Result<Duration> {
    for _ in 0..warmup {
        let _ = f()?;
    }
    device.synchronize()?;
    let started = Instant::now();
    for _ in 0..iterations {
        let _ = f()?;
    }
    device.synchronize()?;
    Ok(started.elapsed())
}

fn deterministic_values(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let a = (idx % 251) as f32 - 125.0;
            let b = ((idx / 251) % 127) as f32 - 63.0;
            (a * 0.007 + b * 0.003).sin() * scale
        })
        .collect()
}

fn eagle_chain_draft(args: EagleChainDraftArgs) -> Result<()> {
    if args.draft_width == 0 {
        anyhow::bail!("--draft-width must be greater than zero");
    }
    let started = Instant::now();
    let file = fs::File::open(&args.trace)
        .with_context(|| format!("open trace {}", args.trace.display()))?;
    let trace: EagleTraceReport = serde_json::from_reader(file)
        .with_context(|| format!("parse trace {}", args.trace.display()))?;
    if trace.steps.is_empty() {
        anyhow::bail!("trace contains no steps");
    }
    let width = args.draft_width.min(trace.steps.len());
    let draft_tokens = trace
        .steps
        .iter()
        .take(width)
        .map(|step| {
            step.top_logits
                .first()
                .map(|logit| logit.token_id)
                .unwrap_or(step.target_token_id)
        })
        .collect::<Vec<_>>();
    let target_tokens = trace
        .generated_token_ids
        .iter()
        .take(width)
        .copied()
        .collect::<Vec<_>>();
    let accepted_tokens = draft_tokens
        .iter()
        .zip(target_tokens.iter())
        .take_while(|(draft, target)| draft == target)
        .count();
    let first_rejected_index = (accepted_tokens < draft_tokens.len()).then_some(accepted_tokens);
    let bonus_token_id = trace
        .generated_token_ids
        .get(accepted_tokens)
        .copied()
        .or_else(|| trace.generated_token_ids.last().copied());
    let mut reconstructed = draft_tokens[..accepted_tokens].to_vec();
    if let Some(bonus) = bonus_token_id {
        reconstructed.push(bonus);
    }
    let feature_summary = trace
        .steps
        .iter()
        .take(width)
        .map(|step| {
            let hidden_state_count = step.hidden_states.len();
            let hidden_size = step
                .hidden_states
                .first()
                .map(|state| state.hidden_size)
                .unwrap_or(0);
            let fused_feature_l2 = step
                .hidden_states
                .iter()
                .flat_map(|state| state.values.iter())
                .map(|value| (*value as f64) * (*value as f64))
                .sum::<f64>()
                .sqrt();
            serde_json::json!({
                "step": step.step,
                "context_position": step.context_position,
                "target_token_id": step.target_token_id,
                "draft_token_id": step.top_logits.first().map(|logit| logit.token_id),
                "hidden_state_count": hidden_state_count,
                "hidden_size": hidden_size,
                "fused_feature_l2": fused_feature_l2,
            })
        })
        .collect::<Vec<_>>();
    let draft_elapsed = started.elapsed();

    let report = serde_json::json!({
        "kind": "lmbrrr_eagle_chain_draft_probe",
        "schema_version": 1,
        "trace": args.trace,
        "draft_source": "trace_top1_oracle_probe",
        "draft_width": args.draft_width,
        "scheduled_width": width,
        "prompt_tokens": trace.prompt_tokens,
        "generated_tokens": trace.generated_tokens,
        "capture_layers": trace.capture_layers,
        "draft_seconds": secs(draft_elapsed),
        "draft_token_ids": draft_tokens,
        "target_token_ids": target_tokens,
        "accepted_tokens": accepted_tokens,
        "bonus_token_id": bonus_token_id,
        "bonus_tokens": usize::from(bonus_token_id.is_some()),
        "accepted_length": accepted_tokens + usize::from(bonus_token_id.is_some()),
        "acceptance_rate": if width > 0 { Some(accepted_tokens as f64 / width as f64) } else { None },
        "first_rejected_index": first_rejected_index,
        "verifier_waste_tokens": first_rejected_index.map(|idx| width.saturating_sub(idx + 1)).unwrap_or(0),
        "reconstructed_token_ids": reconstructed,
        "exact_greedy_prefix_match": first_rejected_index.is_none(),
        "feature_summary": feature_summary,
        "note": "Offline EAGLE-style chain probe over trace top-1 tokens. This validates accepted-length accounting over hidden-state features but is not a trained drafter or speedup claim.",
    });
    write_json_report(args.output.as_ref(), &report)
}

fn eagle_live_probe(args: EagleLiveProbeArgs) -> Result<()> {
    if args.max_new_tokens == 0 {
        anyhow::bail!("--max-new-tokens must be greater than zero");
    }
    if args.draft_width == 0 {
        anyhow::bail!("--draft-width must be greater than zero");
    }

    let draft_head = EagleDraftHead::from_manifest(&args.draft_head_manifest)?;
    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let prompt_text = chat_prompt(&args.prompt, 0, args.enable_thinking);
    let prompt_tokens = tokenize_prompt(&tokenizer, prompt_text)?;
    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
    )?;
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());
    let trace_recorder = Qwen35TraceRecorder::new(draft_head.capture_layers.clone());
    model.set_text_trace_recorder(Some(trace_recorder.clone()));
    model.clear_cache();

    let mut generated_token_ids = Vec::with_capacity(args.max_new_tokens);
    let mut draft_token_ids = Vec::with_capacity(args.max_new_tokens);
    let mut confidences = Vec::with_capacity(args.max_new_tokens);
    let mut steps = Vec::with_capacity(args.max_new_tokens);
    let mut total_forward_elapsed = Duration::ZERO;
    let mut total_argmax_elapsed = Duration::ZERO;
    let mut total_draft_elapsed = Duration::ZERO;
    let mut eos_reached = false;

    trace_recorder.clear();
    let prompt_input = Tensor::from_slice(&prompt_tokens, (1, prompt_tokens.len()), &device)?;
    let mut forward_start = Instant::now();
    let mut logits = model.forward(
        &prompt_input,
        None::<&ProcessedImages>,
        &args.model.downsample_mode,
        0,
    )?;
    device.synchronize()?;
    let mut forward_elapsed = forward_start.elapsed();
    let mut hidden_states = trace_recorder.take();
    let mut phase = "prefill";
    let mut context_position = prompt_tokens.len() - 1;
    let mut offset = 0usize;
    let mut seq_len = prompt_tokens.len();

    for step_index in 0..args.max_new_tokens {
        total_forward_elapsed += forward_elapsed;
        let (target_token, argmax_elapsed) = argmax_token(&logits, &device)?;
        total_argmax_elapsed += argmax_elapsed;

        let feature = eagle_feature_from_hidden_states(
            &hidden_states,
            &draft_head.capture_layers,
            draft_head.input_dim,
        )?;
        let draft_start = Instant::now();
        let prediction = draft_head.predict(&feature)?;
        let draft_elapsed = draft_start.elapsed();
        total_draft_elapsed += draft_elapsed;

        let step_eos = eos_ids.contains(&target_token);
        generated_token_ids.push(target_token);
        draft_token_ids.push(prediction.token_id);
        confidences.push(prediction.confidence);
        steps.push(serde_json::json!({
            "step": step_index,
            "phase": phase,
            "context_position": context_position,
            "offset": offset,
            "seq_len": seq_len,
            "target_token_id": target_token,
            "target_token": decode_token_lossy(&tokenizer, target_token),
            "draft_token_id": prediction.token_id,
            "draft_token": decode_token_lossy(&tokenizer, prediction.token_id),
            "token_match": prediction.token_id == target_token,
            "draft_confidence": prediction.confidence,
            "draft_head_seconds": secs(draft_elapsed),
            "target_forward_seconds": secs(forward_elapsed),
            "argmax_seconds": secs(argmax_elapsed),
            "hidden_state_count": hidden_states.len(),
            "top_draft_logits": prediction.top_logits.iter().map(|item| {
                serde_json::json!({
                    "token_id": item.token_id,
                    "token": decode_token_lossy(&tokenizer, item.token_id),
                    "logit": item.logit,
                    "confidence": item.confidence,
                })
            }).collect::<Vec<_>>(),
        }));

        if step_eos {
            eos_reached = true;
            break;
        }
        if generated_token_ids.len() == args.max_new_tokens {
            break;
        }

        phase = "decode";
        context_position = prompt_tokens.len() + generated_token_ids.len() - 1;
        offset = context_position;
        seq_len = 1;
        trace_recorder.clear();
        let input = Tensor::from_slice(&[target_token], (1, 1), &device)?;
        forward_start = Instant::now();
        logits = model.forward(
            &input,
            None::<&ProcessedImages>,
            &args.model.downsample_mode,
            offset,
        )?;
        device.synchronize()?;
        forward_elapsed = forward_start.elapsed();
        hidden_states = trace_recorder.take();
    }
    model.set_text_trace_recorder(None);

    let candidate_width = args
        .draft_width
        .min(draft_token_ids.len())
        .min(generated_token_ids.len());
    let mut draft_chain = draft_token_ids[..candidate_width].to_vec();
    let confidence_schedule = if args.schedule_confidence_threshold.is_some() {
        apply_confidence_schedule(
            &mut draft_chain,
            &confidences[..candidate_width],
            args.schedule_confidence_threshold,
        )?
    } else {
        None
    };
    let scheduled_width = draft_chain.len();
    let target_chain = generated_token_ids[..scheduled_width].to_vec();
    let bonus_token_id = generated_token_ids
        .get(scheduled_width)
        .copied()
        .or_else(|| generated_token_ids.last().copied())
        .context("live probe generated no target tokens")?;
    let analysis = analyze_verification(&draft_chain, &target_chain, bonus_token_id)?;
    let exact_greedy_prefix_match = generated_token_ids
        .get(..analysis.reconstructed_token_ids.len())
        .map(|prefix| prefix == analysis.reconstructed_token_ids.as_slice())
        .unwrap_or(false);

    let report = serde_json::json!({
        "kind": "lmbrrr_eagle_live_probe",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "downsample_mode": args.model.downsample_mode.as_str(),
        "enable_thinking": args.enable_thinking,
        "artifact_seconds": secs(bundle.elapsed),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "draft_head_manifest": args.draft_head_manifest,
        "draft_head": draft_head.manifest_json(),
        "prompt": args.prompt.as_str(),
        "prompt_tokens": prompt_tokens.len(),
        "generated_tokens": generated_token_ids.len(),
        "generated_token_ids": &generated_token_ids,
        "generated_text": decode_tokens(&tokenizer, &generated_token_ids)?,
        "eos_reached": eos_reached,
        "requested_draft_width": args.draft_width,
        "candidate_draft_width": candidate_width,
        "scheduled_draft_width": scheduled_width,
        "confidence_schedule": confidence_schedule_json(&confidence_schedule),
        "draft_token_ids": &draft_token_ids,
        "draft_text": decode_tokens(&tokenizer, &draft_chain)?,
        "draft_confidences": &confidences,
        "target_token_ids": target_chain,
        "accepted_tokens": analysis.accepted_tokens,
        "accepted_length": analysis.accepted_length(),
        "acceptance_rate": analysis.acceptance_rate(),
        "first_rejected_index": analysis.first_rejected_index,
        "verifier_waste_tokens": analysis.verifier_waste_tokens(),
        "verifier_waste_share": analysis.verifier_waste_share(),
        "target_decode_steps_covered_estimate": analysis.accepted_length(),
        "target_forward_calls_saved_estimate": analysis.accepted_length().saturating_sub(1),
        "bonus_token_id": analysis.bonus_token_id,
        "bonus_token": decode_token_lossy(&tokenizer, analysis.bonus_token_id),
        "reconstructed_token_ids": &analysis.reconstructed_token_ids,
        "reconstructed_text": decode_tokens(&tokenizer, &analysis.reconstructed_token_ids)?,
        "exact_greedy_prefix_match": exact_greedy_prefix_match,
        "target_forward_seconds": secs(total_forward_elapsed),
        "target_forward_tokens_per_second": tokens_per_second(generated_token_ids.len(), total_forward_elapsed),
        "argmax_seconds": secs(total_argmax_elapsed),
        "draft_head_seconds": secs(total_draft_elapsed),
        "draft_head_tokens_per_second": tokens_per_second(draft_token_ids.len(), total_draft_elapsed),
        "draft_overhead_share_vs_target_forward": if total_forward_elapsed.as_secs_f64() > 0.0 {
            Some(total_draft_elapsed.as_secs_f64() / total_forward_elapsed.as_secs_f64())
        } else {
            None
        },
        "steps": steps,
        "note": "Live target-model probe: the draft head runs on captured target hidden states during greedy generation. This measures draft-head overhead and proposal quality, not an accelerated EAGLE decode yet.",
    });

    write_json_report(args.output.as_ref(), &report)
}

fn eagle_recurrent_draft(args: EagleRecurrentDraftArgs) -> Result<()> {
    if args.draft_width == 0 {
        anyhow::bail!("--draft-width must be greater than zero");
    }

    let drafter = EagleRecurrentDrafter::from_manifest(&args.drafter_manifest)?;
    let draft_width = args.draft_width.min(drafter.max_draft_width);
    if draft_width == 0 {
        anyhow::bail!("drafter max draft width is zero");
    }

    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let prompt_text = chat_prompt(&args.prompt, 0, args.enable_thinking);
    let prompt_tokens = tokenize_prompt(&tokenizer, prompt_text)?;
    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
    )?;

    let trace_recorder = Qwen35TraceRecorder::new(drafter.capture_layers.clone());
    model.set_text_trace_recorder(Some(trace_recorder.clone()));
    model.clear_cache();
    trace_recorder.clear();

    let prompt_input = Tensor::from_slice(&prompt_tokens, (1, prompt_tokens.len()), &device)?;
    let prefill_start = Instant::now();
    let prompt_logits = model.forward(
        &prompt_input,
        None::<&ProcessedImages>,
        &args.model.downsample_mode,
        0,
    )?;
    device.synchronize()?;
    let prefill_elapsed = prefill_start.elapsed();
    let hidden_states = trace_recorder.take();
    let anchor_feature = eagle_feature_from_hidden_states(
        &hidden_states,
        &drafter.capture_layers,
        drafter.feature_dim,
    )?;

    let mut draft_chain = Vec::with_capacity(draft_width);
    let mut draft_confidences = Vec::with_capacity(draft_width);
    let mut draft_steps = Vec::with_capacity(draft_width);
    let mut previous_token = *prompt_tokens
        .last()
        .context("prompt tokenization produced no tokens")?;
    let draft_start = Instant::now();
    for draft_position in 0..draft_width {
        let prediction = drafter.predict(&anchor_feature, previous_token, draft_position)?;
        draft_steps.push(serde_json::json!({
            "draft_position": draft_position,
            "previous_token_id": previous_token,
            "previous_token": decode_token_lossy(&tokenizer, previous_token),
            "previous_token_seen_by_drafter": drafter.prev_token_ids.contains(&previous_token),
            "draft_token_id": prediction.token_id,
            "draft_token": decode_token_lossy(&tokenizer, prediction.token_id),
            "draft_confidence": prediction.confidence,
            "top_draft_logits": prediction.top_logits.iter().map(|item| {
                serde_json::json!({
                    "token_id": item.token_id,
                    "token": decode_token_lossy(&tokenizer, item.token_id),
                    "logit": item.logit,
                    "confidence": item.confidence,
                })
            }).collect::<Vec<_>>(),
        }));
        previous_token = prediction.token_id;
        draft_confidences.push(prediction.confidence);
        draft_chain.push(prediction.token_id);
    }
    let draft_elapsed = draft_start.elapsed();

    let confidence_schedule = if args.schedule_confidence_threshold.is_some() {
        let candidate_width = draft_chain.len();
        let scheduled_confidences = draft_confidences[..candidate_width].to_vec();
        apply_confidence_schedule(
            &mut draft_chain,
            &scheduled_confidences,
            args.schedule_confidence_threshold,
        )?
    } else {
        None
    };

    let (first_target_token, first_argmax_elapsed) = argmax_token(&prompt_logits, &device)?;
    let mut target_token_ids = Vec::with_capacity(draft_chain.len());
    let mut verify_elapsed = Duration::ZERO;
    let mut verify_argmax_elapsed = first_argmax_elapsed;
    let bonus_after_all = if draft_chain.is_empty() {
        first_target_token
    } else {
        target_token_ids.push(first_target_token);
        let draft_input = Tensor::from_slice(&draft_chain, (1, draft_chain.len()), &device)?;
        let verify_start = Instant::now();
        let draft_logits = model.forward_all_logits(
            &draft_input,
            None::<&ProcessedImages>,
            &args.model.downsample_mode,
            prompt_tokens.len(),
        )?;
        device.synchronize()?;
        verify_elapsed = verify_start.elapsed();
        let (chunk_target_tokens, chunk_argmax_elapsed) = argmax_tokens(&draft_logits, &device)?;
        verify_argmax_elapsed += chunk_argmax_elapsed;
        if chunk_target_tokens.len() != draft_chain.len() {
            anyhow::bail!(
                "verifier chunk returned {} target tokens for {} draft tokens",
                chunk_target_tokens.len(),
                draft_chain.len()
            );
        }
        target_token_ids.extend(
            chunk_target_tokens
                .iter()
                .take(draft_chain.len().saturating_sub(1))
                .copied(),
        );
        chunk_target_tokens
            .last()
            .copied()
            .context("missing verifier bonus token")?
    };
    model.set_text_trace_recorder(None);

    let analysis = analyze_verification(&draft_chain, &target_token_ids, bonus_after_all)?;
    let greedy_cap = analysis.accepted_length().max(1);
    let baseline_generation = GenerationArgs {
        max_new_tokens: greedy_cap,
        temperature: 0.0,
        top_p: None,
        top_k: None,
        seed: 299792458,
        repeat_penalty: 1.0,
        repeat_last_n: 64,
        enable_thinking: args.enable_thinking,
    };
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());
    let baseline_started = Instant::now();
    let baseline_stats = generate_tokens(
        &mut model,
        &device,
        &baseline_generation,
        &prompt_tokens,
        None::<&ProcessedImages>,
        &args.model.downsample_mode,
        &eos_ids,
        |_, _, _, _| Ok(()),
    )?;
    let baseline_elapsed = baseline_started.elapsed();
    let exact_greedy_prefix_match = baseline_stats
        .generated_token_ids
        .get(..analysis.reconstructed_token_ids.len())
        .map(|prefix| prefix == analysis.reconstructed_token_ids.as_slice())
        .unwrap_or(false);

    let speculative_elapsed =
        prefill_elapsed + draft_elapsed + verify_elapsed + verify_argmax_elapsed;
    let baseline_model_elapsed =
        baseline_stats.prefill_elapsed + baseline_stats.decode_model_elapsed;
    let speedup_estimate = if speculative_elapsed.is_zero() {
        None
    } else {
        Some(baseline_model_elapsed.as_secs_f64() / speculative_elapsed.as_secs_f64())
    };

    let report = serde_json::json!({
        "kind": "lmbrrr_eagle_recurrent_draft",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "downsample_mode": args.model.downsample_mode.as_str(),
        "enable_thinking": args.enable_thinking,
        "artifact_seconds": secs(bundle.elapsed),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "drafter_manifest": args.drafter_manifest,
        "drafter": drafter.manifest_json(),
        "prompt": args.prompt.as_str(),
        "prompt_tokens": prompt_tokens.len(),
        "requested_draft_width": args.draft_width,
        "draft_width": draft_width,
        "scheduled_draft_width": draft_chain.len(),
        "confidence_schedule": confidence_schedule_json(&confidence_schedule),
        "draft_token_ids": &draft_chain,
        "draft_text": decode_tokens(&tokenizer, &draft_chain)?,
        "draft_confidences": &draft_confidences,
        "target_token_ids": &target_token_ids,
        "accepted_tokens": analysis.accepted_tokens,
        "accepted_length": analysis.accepted_length(),
        "acceptance_rate": analysis.acceptance_rate(),
        "first_rejected_index": analysis.first_rejected_index,
        "verifier_waste_tokens": analysis.verifier_waste_tokens(),
        "verifier_waste_share": analysis.verifier_waste_share(),
        "target_decode_steps_covered_estimate": analysis.accepted_length(),
        "target_forward_calls_saved_estimate": analysis.accepted_length().saturating_sub(1),
        "bonus_token_id": analysis.bonus_token_id,
        "bonus_token": decode_token_lossy(&tokenizer, analysis.bonus_token_id),
        "reconstructed_token_ids": &analysis.reconstructed_token_ids,
        "reconstructed_text": decode_tokens(&tokenizer, &analysis.reconstructed_token_ids)?,
        "baseline_greedy_token_ids": &baseline_stats.generated_token_ids,
        "baseline_greedy_text": decode_tokens(&tokenizer, &baseline_stats.generated_token_ids)?,
        "exact_greedy_prefix_match": exact_greedy_prefix_match,
        "prefill_seconds": secs(prefill_elapsed),
        "draft_seconds": secs(draft_elapsed),
        "verify_seconds": secs(verify_elapsed),
        "argmax_seconds": secs(verify_argmax_elapsed),
        "speculative_model_seconds": secs(speculative_elapsed),
        "baseline_model_seconds_for_same_window": secs(baseline_model_elapsed),
        "baseline_wall_seconds_for_same_window": secs(baseline_elapsed),
        "end_to_end_model_speedup_estimate": speedup_estimate,
        "draft_tokens_per_second": tokens_per_second(draft_width, draft_elapsed),
        "hidden_state_count": hidden_states.len(),
        "steps": draft_steps,
        "note": "Recurrent smoke drafter: one target anchor feature is computed, then the drafter proposes a block before target chunk verification. The observed-vocabulary head is a smoke artifact, not a production EAGLE model.",
    });
    write_json_report(args.output.as_ref(), &report)
}

#[derive(Debug)]
struct EagleRecurrentDrafter {
    capture_layers: Vec<usize>,
    feature_dim: usize,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    target_token_ids: Vec<u32>,
    prev_token_ids: Vec<u32>,
    max_draft_width: usize,
    feature_mean: Vec<f32>,
    feature_std: Vec<f32>,
    w0: Vec<f32>,
    b0: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    source_manifest: EagleRecurrentDrafterManifest,
}

impl EagleRecurrentDrafter {
    fn from_manifest(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("read recurrent drafter manifest {}", path.display()))?;
        let manifest: EagleRecurrentDrafterManifest = serde_json::from_str(&text)
            .with_context(|| format!("parse recurrent drafter manifest {}", path.display()))?;
        if manifest.kind != "lmbrrr_eagle_recurrent_drafter" {
            anyhow::bail!("unsupported recurrent drafter kind {}", manifest.kind);
        }
        if manifest.draft_head_type != "observed-vocabulary-recurrent-mlp" {
            anyhow::bail!(
                "unsupported recurrent drafter type {}",
                manifest.draft_head_type
            );
        }
        if manifest.activation != "gelu_tanh" {
            anyhow::bail!(
                "unsupported recurrent drafter activation {}",
                manifest.activation
            );
        }
        if manifest.output_dim != manifest.target_token_ids.len() {
            anyhow::bail!(
                "recurrent drafter output_dim {} does not match {} target token ids",
                manifest.output_dim,
                manifest.target_token_ids.len()
            );
        }
        let expected_input_dim = manifest.feature_dim + manifest.prev_token_ids.len() + 1;
        if manifest.input_dim != expected_input_dim {
            anyhow::bail!(
                "recurrent drafter input_dim {} does not match feature {} + prev vocab {} + position",
                manifest.input_dim,
                manifest.feature_dim,
                manifest.prev_token_ids.len()
            );
        }
        let weights_path = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&manifest.weights);
        let bytes = fs::read(&weights_path).with_context(|| {
            format!("read recurrent drafter weights {}", weights_path.display())
        })?;
        let safetensors = safetensors::SafeTensors::deserialize(&bytes).with_context(|| {
            format!("parse recurrent drafter weights {}", weights_path.display())
        })?;
        let feature_mean = load_safetensor_f32(&safetensors, "feature_mean")?;
        let feature_std = load_safetensor_f32(&safetensors, "feature_std")?;
        let w0 = load_safetensor_f32(&safetensors, "net.0.weight")?;
        let b0 = load_safetensor_f32(&safetensors, "net.0.bias")?;
        let w2 = load_safetensor_f32(&safetensors, "net.2.weight")?;
        let b2 = load_safetensor_f32(&safetensors, "net.2.bias")?;

        ensure_len("feature_mean", &feature_mean, manifest.feature_dim)?;
        ensure_len("feature_std", &feature_std, manifest.feature_dim)?;
        ensure_len(
            "net.0.weight",
            &w0,
            manifest.input_dim * manifest.hidden_dim,
        )?;
        ensure_len("net.0.bias", &b0, manifest.hidden_dim)?;
        ensure_len(
            "net.2.weight",
            &w2,
            manifest.hidden_dim * manifest.output_dim,
        )?;
        ensure_len("net.2.bias", &b2, manifest.output_dim)?;

        Ok(Self {
            capture_layers: manifest.capture_layers.clone(),
            feature_dim: manifest.feature_dim,
            input_dim: manifest.input_dim,
            hidden_dim: manifest.hidden_dim,
            output_dim: manifest.output_dim,
            target_token_ids: manifest.target_token_ids.clone(),
            prev_token_ids: manifest.prev_token_ids.clone(),
            max_draft_width: manifest.max_draft_width,
            feature_mean,
            feature_std,
            w0,
            b0,
            w2,
            b2,
            source_manifest: manifest,
        })
    }

    fn predict(
        &self,
        anchor_feature: &[f32],
        previous_token: u32,
        draft_position: usize,
    ) -> Result<EagleDraftPrediction> {
        ensure_len("anchor feature", anchor_feature, self.feature_dim)?;
        let prev_index = self
            .prev_token_ids
            .iter()
            .position(|token_id| *token_id == previous_token);
        let mut input = Vec::with_capacity(self.input_dim);
        for (idx, value) in anchor_feature.iter().enumerate() {
            input.push((*value - self.feature_mean[idx]) / self.feature_std[idx]);
        }
        input.extend((0..self.prev_token_ids.len()).map(|idx| f32::from(prev_index == Some(idx))));
        let position = if self.max_draft_width <= 1 {
            0.0
        } else {
            draft_position as f32 / (self.max_draft_width - 1) as f32
        };
        input.push(position);
        ensure_len("recurrent drafter input", &input, self.input_dim)?;

        let mut hidden = vec![0f32; self.hidden_dim];
        for (row, hidden_value) in hidden.iter_mut().enumerate() {
            let mut acc = self.b0[row];
            let row_offset = row * self.input_dim;
            for (col, value) in input.iter().enumerate() {
                acc += self.w0[row_offset + col] * value;
            }
            *hidden_value = gelu_tanh(acc);
        }

        let mut logits = vec![0f32; self.output_dim];
        for (row, logit) in logits.iter_mut().enumerate() {
            let mut acc = self.b2[row];
            let row_offset = row * self.hidden_dim;
            for (col, value) in hidden.iter().enumerate() {
                acc += self.w2[row_offset + col] * value;
            }
            *logit = acc;
        }
        let top_logits = top_draft_logits(&logits, &self.target_token_ids, 5)?;
        let best = top_logits
            .first()
            .context("recurrent drafter produced no logits")?
            .clone();
        Ok(EagleDraftPrediction {
            token_id: best.token_id,
            confidence: best.confidence,
            top_logits,
        })
    }

    fn manifest_json(&self) -> serde_json::Value {
        serde_json::json!({
            "draft_head_type": self.source_manifest.draft_head_type,
            "activation": self.source_manifest.activation,
            "capture_layers": self.capture_layers,
            "feature_dim": self.feature_dim,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "target_vocab_size": self.target_token_ids.len(),
            "prev_vocab_size": self.prev_token_ids.len(),
            "max_draft_width": self.max_draft_width,
            "feature_normalization": self.source_manifest.feature_normalization,
            "recurrent_state": self.source_manifest.recurrent_state,
            "training": self.source_manifest.training,
            "metrics": self.source_manifest.metrics,
            "limits": self.source_manifest.limits,
        })
    }
}

#[derive(Debug, Deserialize)]
struct EagleRecurrentDrafterManifest {
    kind: String,
    draft_head_type: String,
    activation: String,
    weights: String,
    capture_layers: Vec<usize>,
    feature_dim: usize,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    target_token_ids: Vec<u32>,
    prev_token_ids: Vec<u32>,
    max_draft_width: usize,
    feature_normalization: String,
    recurrent_state: String,
    #[serde(default)]
    training: serde_json::Value,
    #[serde(default)]
    metrics: serde_json::Value,
    #[serde(default)]
    limits: Vec<String>,
}

#[derive(Debug)]
struct EagleDraftHead {
    capture_layers: Vec<usize>,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    token_ids: Vec<u32>,
    feature_mean: Vec<f32>,
    feature_std: Vec<f32>,
    w0: Vec<f32>,
    b0: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    source_manifest: EagleDraftHeadManifest,
}

impl EagleDraftHead {
    fn from_manifest(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("read draft head manifest {}", path.display()))?;
        let manifest: EagleDraftHeadManifest = serde_json::from_str(&text)
            .with_context(|| format!("parse draft head manifest {}", path.display()))?;
        if manifest.kind != "lmbrrr_eagle_draft_head" {
            anyhow::bail!("unsupported draft head kind {}", manifest.kind);
        }
        if manifest.draft_head_type != "observed-vocabulary-mlp" {
            anyhow::bail!("unsupported draft head type {}", manifest.draft_head_type);
        }
        if manifest.activation != "gelu_tanh" {
            anyhow::bail!("unsupported draft head activation {}", manifest.activation);
        }
        if manifest.output_dim != manifest.token_ids.len() {
            anyhow::bail!(
                "draft head output_dim {} does not match {} token ids",
                manifest.output_dim,
                manifest.token_ids.len()
            );
        }
        let weights_path = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&manifest.weights);
        let bytes = fs::read(&weights_path)
            .with_context(|| format!("read draft head weights {}", weights_path.display()))?;
        let safetensors = safetensors::SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parse draft head weights {}", weights_path.display()))?;
        let feature_mean = load_safetensor_f32(&safetensors, "feature_mean")?;
        let feature_std = load_safetensor_f32(&safetensors, "feature_std")?;
        let w0 = load_safetensor_f32(&safetensors, "net.0.weight")?;
        let b0 = load_safetensor_f32(&safetensors, "net.0.bias")?;
        let w2 = load_safetensor_f32(&safetensors, "net.2.weight")?;
        let b2 = load_safetensor_f32(&safetensors, "net.2.bias")?;

        ensure_len("feature_mean", &feature_mean, manifest.input_dim)?;
        ensure_len("feature_std", &feature_std, manifest.input_dim)?;
        ensure_len(
            "net.0.weight",
            &w0,
            manifest.input_dim * manifest.hidden_dim,
        )?;
        ensure_len("net.0.bias", &b0, manifest.hidden_dim)?;
        ensure_len(
            "net.2.weight",
            &w2,
            manifest.hidden_dim * manifest.output_dim,
        )?;
        ensure_len("net.2.bias", &b2, manifest.output_dim)?;

        Ok(Self {
            capture_layers: manifest.capture_layers.clone(),
            input_dim: manifest.input_dim,
            hidden_dim: manifest.hidden_dim,
            output_dim: manifest.output_dim,
            token_ids: manifest.token_ids.clone(),
            feature_mean,
            feature_std,
            w0,
            b0,
            w2,
            b2,
            source_manifest: manifest,
        })
    }

    fn predict(&self, feature: &[f32]) -> Result<EagleDraftPrediction> {
        ensure_len("live feature", feature, self.input_dim)?;
        let mut hidden = vec![0f32; self.hidden_dim];
        for (row, hidden_value) in hidden.iter_mut().enumerate() {
            let mut acc = self.b0[row];
            let row_offset = row * self.input_dim;
            for (col, value) in feature.iter().enumerate() {
                let normalized = (*value - self.feature_mean[col]) / self.feature_std[col];
                acc += self.w0[row_offset + col] * normalized;
            }
            *hidden_value = gelu_tanh(acc);
        }

        let mut logits = vec![0f32; self.output_dim];
        for (row, logit) in logits.iter_mut().enumerate() {
            let mut acc = self.b2[row];
            let row_offset = row * self.hidden_dim;
            for (col, value) in hidden.iter().enumerate() {
                acc += self.w2[row_offset + col] * value;
            }
            *logit = acc;
        }
        let top_logits = top_draft_logits(&logits, &self.token_ids, 5)?;
        let best = top_logits
            .first()
            .context("draft head produced no logits")?
            .clone();
        Ok(EagleDraftPrediction {
            token_id: best.token_id,
            confidence: best.confidence,
            top_logits,
        })
    }

    fn manifest_json(&self) -> serde_json::Value {
        serde_json::json!({
            "draft_head_type": self.source_manifest.draft_head_type,
            "activation": self.source_manifest.activation,
            "capture_layers": self.capture_layers,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "feature_normalization": self.source_manifest.feature_normalization,
            "training": self.source_manifest.training,
            "metrics": self.source_manifest.metrics,
            "limits": self.source_manifest.limits,
        })
    }
}

#[derive(Clone, Debug)]
struct EagleDraftPrediction {
    token_id: u32,
    confidence: f64,
    top_logits: Vec<EagleDraftLogit>,
}

#[derive(Clone, Debug)]
struct EagleDraftLogit {
    token_id: u32,
    logit: f32,
    confidence: f64,
}

#[derive(Debug, Deserialize)]
struct EagleDraftHeadManifest {
    kind: String,
    draft_head_type: String,
    activation: String,
    weights: String,
    capture_layers: Vec<usize>,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    token_ids: Vec<u32>,
    feature_normalization: String,
    #[serde(default)]
    training: serde_json::Value,
    #[serde(default)]
    metrics: serde_json::Value,
    #[serde(default)]
    limits: Vec<String>,
}

fn eagle_feature_from_hidden_states(
    hidden_states: &[Qwen35HiddenStateTrace],
    capture_layers: &[usize],
    input_dim: usize,
) -> Result<Vec<f32>> {
    let mut states = hidden_states.to_vec();
    states.sort_by_key(|state| state.layer_index);
    let layers = states
        .iter()
        .map(|state| state.layer_index)
        .collect::<Vec<_>>();
    if layers != capture_layers {
        anyhow::bail!(
            "captured hidden layers {:?}, expected {:?}",
            layers,
            capture_layers
        );
    }
    let mut feature = Vec::with_capacity(input_dim);
    for state in &states {
        feature.extend_from_slice(&state.values);
    }
    ensure_len("fused hidden-state feature", &feature, input_dim)?;
    Ok(feature)
}

fn load_safetensor_f32(safetensors: &safetensors::SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let view = safetensors
        .tensor(name)
        .with_context(|| format!("missing draft head tensor {name}"))?;
    let tensor = view.load(&Device::Cpu)?;
    Ok(tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

fn ensure_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        anyhow::bail!("{name} has {} values, expected {expected}", values.len());
    }
    Ok(())
}

fn gelu_tanh(value: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * value * (1.0 + (SQRT_2_OVER_PI * (value + 0.044_715 * value.powi(3))).tanh())
}

fn top_draft_logits(
    logits: &[f32],
    token_ids: &[u32],
    top_k: usize,
) -> Result<Vec<EagleDraftLogit>> {
    if logits.len() != token_ids.len() {
        anyhow::bail!(
            "draft logits length {} does not match token id count {}",
            logits.len(),
            token_ids.len()
        );
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom = logits
        .iter()
        .map(|logit| ((*logit - max_logit) as f64).exp())
        .sum::<f64>();
    let mut indices = (0..logits.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| logits[*right].total_cmp(&logits[*left]));
    Ok(indices
        .into_iter()
        .take(top_k.min(logits.len()))
        .map(|idx| EagleDraftLogit {
            token_id: token_ids[idx],
            logit: logits[idx],
            confidence: (((logits[idx] - max_logit) as f64).exp()) / denom,
        })
        .collect())
}

#[derive(Debug, Deserialize)]
struct EagleTraceReport {
    prompt_tokens: usize,
    generated_tokens: usize,
    generated_token_ids: Vec<u32>,
    capture_layers: Vec<usize>,
    steps: Vec<EagleTraceStep>,
}

#[derive(Debug, Deserialize)]
struct EagleTraceStep {
    step: usize,
    context_position: usize,
    target_token_id: u32,
    top_logits: Vec<EagleTraceTopLogit>,
    hidden_states: Vec<EagleTraceHiddenState>,
}

#[derive(Debug, Deserialize)]
struct EagleTraceTopLogit {
    token_id: u32,
}

#[derive(Debug, Deserialize)]
struct EagleTraceHiddenState {
    hidden_size: usize,
    values: Vec<f32>,
}

fn run_quant_baseline_case(
    model: &mut MiniCpmForConditionalGeneration,
    device: &Device,
    tokenizer: &Tokenizer,
    row: &CalibrationRow,
    downsample_mode: &str,
    top_k: usize,
) -> Result<serde_json::Value> {
    if row.token_ids.is_empty() {
        anyhow::bail!("calibration row {} has no token ids", row.id);
    }
    model.clear_cache();
    let input = Tensor::from_slice(&row.token_ids, (1, row.token_ids.len()), device)?;
    let started = Instant::now();
    let logits = model.forward(&input, None::<&ProcessedImages>, downsample_mode, 0)?;
    device.synchronize()?;
    let forward_elapsed = started.elapsed();
    let top_logits = top_k_logits(&logits.squeeze(0)?, top_k, tokenizer)?;
    let top1 = top_logits.first();
    Ok(serde_json::json!({
        "id": row.id.as_str(),
        "category": row.category.as_str(),
        "modality": row.modality.as_str(),
        "enable_thinking": row.enable_thinking,
        "media_status": row.media_status.as_deref(),
        "prompt_tokens": row.token_ids.len(),
        "declared_prompt_tokens": row.prompt_token_count,
        "prompt_token_count_match": row.token_ids.len() == row.prompt_token_count,
        "sensitivity_focus": row.sensitivity_focus,
        "prefill_seconds": secs(forward_elapsed),
        "prefill_tokens_per_second": tokens_per_second(row.token_ids.len(), forward_elapsed),
        "top1_token_id": top1.map(|item| item.token_id),
        "top1_token": top1.map(|item| item.token.as_str()),
        "top1_logit": top1.map(|item| item.logit),
        "top_logits": top_logits_json(&top_logits),
    }))
}

fn roofline(args: RooflineArgs) -> Result<()> {
    if args.iterations == 0 || args.dispatch_chain == 0 {
        anyhow::bail!("--iterations and --dispatch-chain must be greater than zero");
    }
    let device = select_device(args.cpu)?;
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };

    let mut copy_rows = Vec::new();
    for mb in [64usize, 256, 1024] {
        let elements = mb * 1024 * 1024 / dtype.size_in_bytes();
        let x = Tensor::zeros(elements, dtype, &device)?;
        let elapsed = time_iterations(&device, args.warmup, args.iterations, || Ok(x.copy()?))?;
        let bytes_moved = 2 * (elements * dtype.size_in_bytes()) as f64 * args.iterations as f64;
        copy_rows.push(serde_json::json!({
            "tensor_mb": mb,
            "seconds": secs(elapsed),
            "achieved_gbps": bytes_moved / secs(elapsed) / 1e9,
        }));
    }
    let peak_copy_gbps = copy_rows
        .iter()
        .filter_map(|row| row["achieved_gbps"].as_f64())
        .fold(0.0f64, f64::max);

    // Matvec throughput at the model's real decode shapes (weights dominate
    // the bytes, so achieved GB/s here is the effective decode-path bandwidth).
    let matvec_shapes: [(&str, usize, usize); 9] = [
        ("mlp_up_or_gate", 3584, 1024),
        ("mlp_down", 1024, 3584),
        ("deltanet_in_proj_qkv", 6144, 1024),
        ("deltanet_out_proj", 1024, 2048),
        ("full_attn_q_gate", 4096, 1024),
        ("full_attn_kv", 512, 1024),
        ("full_attn_o", 1024, 2048),
        ("lm_head", 248094, 1024),
        ("peak_square", 8192, 8192),
    ];
    let mut matvec_rows = Vec::new();
    for (name, out_dim, in_dim) in matvec_shapes {
        let weight = Tensor::zeros((out_dim, in_dim), dtype, &device)?;
        let linear = candle_nn::Linear::new(weight, None);
        let x = Tensor::zeros((1, in_dim), dtype, &device)?;
        let elapsed =
            time_iterations(&device, args.warmup, args.iterations, || linear.forward(&x))?;
        let weight_bytes = (out_dim * in_dim * dtype.size_in_bytes()) as f64;
        let per_iter = secs(elapsed) / args.iterations as f64;
        matvec_rows.push(serde_json::json!({
            "shape": name,
            "out_dim": out_dim,
            "in_dim": in_dim,
            "weight_bytes": weight_bytes as u64,
            "seconds_per_iteration": per_iter,
            "achieved_gbps": weight_bytes / per_iter / 1e9,
        }));
    }

    // Dependent-chain dispatch overhead: tiny affine ops that cannot overlap,
    // mirroring the serial structure of a decode forward.
    let tiny = Tensor::zeros(1, DType::F32, &device)?;
    for _ in 0..args.warmup {
        let mut y = tiny.clone();
        for _ in 0..args.dispatch_chain {
            y = y.affine(1.000001, 0.0)?;
        }
        let _ = y.to_scalar::<f32>()?;
    }
    device.synchronize()?;
    let started = Instant::now();
    for _ in 0..args.iterations {
        let mut y = tiny.clone();
        for _ in 0..args.dispatch_chain {
            y = y.affine(1.000001, 0.0)?;
        }
        let _ = y.to_scalar::<f32>()?;
    }
    device.synchronize()?;
    let tiny_chain_elapsed = started.elapsed();
    let per_dispatch_seconds =
        secs(tiny_chain_elapsed) / (args.iterations * args.dispatch_chain) as f64;

    // Same measurement with a dependent chain of real h=1024 matvecs.
    let weight = Tensor::zeros((1024usize, 1024usize), dtype, &device)?;
    let linear = candle_nn::Linear::new(weight, None);
    let chain = 64usize;
    for _ in 0..args.warmup {
        let mut y = Tensor::zeros((1, 1024usize), dtype, &device)?;
        for _ in 0..chain {
            y = linear.forward(&y)?;
        }
        device.synchronize()?;
    }
    device.synchronize()?;
    let started = Instant::now();
    for _ in 0..args.iterations {
        let mut y = Tensor::zeros((1, 1024usize), dtype, &device)?;
        for _ in 0..chain {
            y = linear.forward(&y)?;
        }
        device.synchronize()?;
    }
    let small_matvec_chain_elapsed = started.elapsed();
    let per_small_matvec_seconds =
        secs(small_matvec_chain_elapsed) / (args.iterations * chain) as f64;

    let dispatch_bound_tok_s =
        1.0 / (args.assumed_dispatches as f64 * per_dispatch_seconds).max(f64::EPSILON);
    let bandwidth_bound_tok_s = peak_copy_gbps * 1e9 / args.assumed_weight_bytes as f64;
    let combined_tok_s = 1.0
        / (args.assumed_dispatches as f64 * per_dispatch_seconds
            + args.assumed_weight_bytes as f64 / (peak_copy_gbps * 1e9));

    let report = serde_json::json!({
        "kind": "lmbrrr_metal_roofline",
        "schema_version": 1,
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "copy_bandwidth": copy_rows,
        "peak_copy_gbps": peak_copy_gbps,
        "matvec": matvec_rows,
        "dispatch_chain_length": args.dispatch_chain,
        "per_dispatch_seconds": per_dispatch_seconds,
        "per_dispatch_microseconds": per_dispatch_seconds * 1e6,
        "per_small_matvec_seconds": per_small_matvec_seconds,
        "per_small_matvec_microseconds": per_small_matvec_seconds * 1e6,
        "projections": {
            "assumed_dispatches_per_forward": args.assumed_dispatches,
            "assumed_weight_bytes_per_forward": args.assumed_weight_bytes,
            "dispatch_bound_tokens_per_second": dispatch_bound_tok_s,
            "bandwidth_bound_tokens_per_second": bandwidth_bound_tok_s,
            "combined_projection_tokens_per_second": combined_tok_s,
        },
        "note": "Dependent-chain timings mirror serial decode structure; copy bandwidth counts read+write bytes. Projections use the assumed dispatch count until encoder-level counting lands.",
    });
    write_json_report(args.output.as_ref(), &report)
}

fn verify_table(args: VerifyTableArgs) -> Result<()> {
    if args.iterations == 0 {
        anyhow::bail!("--iterations must be greater than zero");
    }
    let mut gammas = if args.gammas.is_empty() {
        vec![1, 2, 4, 8, 16, 32]
    } else {
        args.gammas.clone()
    };
    gammas.sort_unstable();
    gammas.dedup();
    if gammas.first() != Some(&1) {
        gammas.insert(0, 1);
    }
    let max_gamma = *gammas.last().expect("gammas is non-empty");
    let profiles = if args.profiles.is_empty() {
        BenchProfile::all().to_vec()
    } else {
        args.profiles.clone()
    };

    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
    )?;
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());

    let mut rows = Vec::new();
    for profile in &profiles {
        let prompt_text = chat_prompt(profile.prompt(), 0, false);
        let prompt_tokens = tokenize_prompt(&tokenizer, prompt_text)?;

        // Realistic verify content: greedy continuation tokens, padded by
        // repeating the last token if EOS arrives before max_gamma.
        let baseline = generate_tokens(
            &mut model,
            &device,
            &greedy_generation_args(max_gamma, false),
            &prompt_tokens,
            None::<&ProcessedImages>,
            &args.model.downsample_mode,
            &eos_ids,
            |_, _, _, _| Ok(()),
        )?;
        let mut chunk_tokens = baseline.generated_token_ids.clone();
        let pad = chunk_tokens.last().copied().unwrap_or(prompt_tokens[0]);
        while chunk_tokens.len() < max_gamma {
            chunk_tokens.push(pad);
        }

        for &gamma in &gammas {
            let mut samples = Vec::with_capacity(args.iterations);
            for iteration in 0..args.warmup + args.iterations {
                model.clear_cache();
                let prompt_input =
                    Tensor::from_slice(&prompt_tokens, (1, prompt_tokens.len()), &device)?;
                let _ = model.forward(
                    &prompt_input,
                    None::<&ProcessedImages>,
                    &args.model.downsample_mode,
                    0,
                )?;
                device.synchronize()?;

                let chunk = &chunk_tokens[..gamma];
                let chunk_input = Tensor::from_slice(chunk, (1, gamma), &device)?;
                let started = Instant::now();
                let logits = model.forward_all_logits(
                    &chunk_input,
                    None::<&ProcessedImages>,
                    &args.model.downsample_mode,
                    prompt_tokens.len(),
                )?;
                device.synchronize()?;
                let chunk_elapsed = started.elapsed();
                let (_, argmax_elapsed) = argmax_tokens(&logits, &device)?;
                if iteration >= args.warmup {
                    samples.push((secs(chunk_elapsed), secs(argmax_elapsed)));
                }
            }
            let mut chunk_seconds = samples.iter().map(|(chunk, _)| *chunk).collect::<Vec<_>>();
            let argmax_seconds = samples.iter().map(|(_, argmax)| *argmax).sum::<f64>()
                / samples.len().max(1) as f64;
            let median_seconds = median(&mut chunk_seconds);
            let spread = chunk_seconds
                .last()
                .copied()
                .unwrap_or(median_seconds)
                - chunk_seconds.first().copied().unwrap_or(median_seconds);
            rows.push(serde_json::json!({
                "profile": profile.name(),
                "context_tokens": prompt_tokens.len(),
                "gamma": gamma,
                "iterations": args.iterations,
                "median_verify_seconds": median_seconds,
                "spread_verify_seconds": spread,
                "mean_argmax_seconds": argmax_seconds,
                "verify_tokens_per_second": gamma as f64 / median_seconds.max(f64::EPSILON),
                "samples": chunk_seconds,
            }));
        }
    }

    // Per-token efficiency vs the gamma=1 step within each profile.
    let mut enriched = Vec::with_capacity(rows.len());
    for row in &rows {
        let profile = row["profile"].as_str().unwrap_or_default();
        let base = rows
            .iter()
            .find(|candidate| {
                candidate["profile"].as_str() == Some(profile)
                    && candidate["gamma"].as_u64() == Some(1)
            })
            .and_then(|candidate| candidate["median_verify_seconds"].as_f64());
        let mut row = row.clone();
        if let (Some(base), Some(seconds), Some(gamma)) = (
            base,
            row["median_verify_seconds"].as_f64(),
            row["gamma"].as_u64(),
        ) {
            row["chunk_cost_vs_single_step"] = serde_json::json!(seconds / base);
            row["per_token_efficiency_vs_decode"] =
                serde_json::json!(gamma as f64 * base / seconds);
        }
        enriched.push(row);
    }

    let report = serde_json::json!({
        "kind": "lmbrrr_verify_throughput_table",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "gammas": gammas,
        "concurrency": 1,
        "concurrency_note": "single-request only; batched verify lands with batched-multi-stream-decode-runner",
        "rows": enriched,
        "scheduler_contract": "T_verify(gamma) per context bucket = median_verify_seconds; T_round(gamma) = T_draft + T_verify(gamma).",
    });
    write_json_report(args.output.as_ref(), &report)
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn resolve_artifacts(model: &ModelArgs) -> Result<ArtifactBundle> {
    let started = Instant::now();
    let artifacts = resolve_model_artifacts(
        &model.model_id,
        &model.revision,
        ArtifactOverrides {
            config: model.config.clone(),
            tokenizer: model.tokenizer.clone(),
            generation_config: model.generation_config.clone(),
            preprocessor: model.preprocessor.clone(),
            weights: model.weights.clone(),
        },
    )?;
    let config = MiniCpmConfig::from_path(&artifacts.config)?;
    let generation_config = artifacts
        .generation_config
        .as_ref()
        .map(GenerationConfig::from_path)
        .transpose()?;
    let weight_report = validate_minicpm_header(&artifacts.weights, &config)?;
    Ok(ArtifactBundle {
        artifacts,
        config,
        generation_config,
        weight_report,
        elapsed: started.elapsed(),
    })
}

fn load_model(
    bundle: &ArtifactBundle,
    dtype: DType,
    device: &Device,
) -> Result<(
    MiniCpmForConditionalGeneration,
    Duration,
    Option<QuantizedLoadStats>,
)> {
    let load_start = Instant::now();
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&bundle.artifacts.weights, dtype, device)? };
    let model = MiniCpmForConditionalGeneration::new(&bundle.config, vb)?;
    Ok((model, load_start.elapsed(), None))
}

fn load_model_with_optional_quantization(
    bundle: &ArtifactBundle,
    dtype: DType,
    device: &Device,
    quantized_manifest: Option<&PathBuf>,
) -> Result<(
    MiniCpmForConditionalGeneration,
    Duration,
    Option<QuantizedLoadStats>,
)> {
    let (mut model, load_elapsed, _) = load_model(bundle, dtype, device)?;
    let Some(manifest) = quantized_manifest else {
        return Ok((model, load_elapsed, None));
    };
    let artifact = QuantizedTextArtifact::from_manifest(manifest, device, dtype)?;
    let quantized_tensors = artifact.quantized_tensor_count();
    let backend = artifact.backend().to_string();
    let quantized_data_bytes = artifact.quantized_data_bytes();
    let dense_equivalent_bytes = artifact.dense_equivalent_bytes();
    let replaced_text_linears = model.apply_quantized_text_artifact(&artifact)?;
    Ok((
        model,
        load_elapsed,
        Some(QuantizedLoadStats {
            manifest: artifact.manifest_path().to_path_buf(),
            quantized_tensors,
            replaced_text_linears,
            backend,
            quantized_data_bytes,
            dense_equivalent_bytes,
        }),
    ))
}

fn load_tokenizer(artifacts: &Artifacts) -> Result<Tokenizer> {
    Tokenizer::from_file(&artifacts.tokenizer)
        .map_err(|err| anyhow::anyhow!("load tokenizer {}: {err}", artifacts.tokenizer.display()))
}

fn tokenize_prompt(tokenizer: &Tokenizer, prompt_text: String) -> Result<Vec<u32>> {
    let tokens = tokenizer
        .encode(prompt_text, false)
        .map_err(|err| anyhow::anyhow!("tokenize prompt: {err}"))?
        .get_ids()
        .to_vec();
    if tokens.is_empty() {
        anyhow::bail!("prompt tokenized to zero tokens");
    }
    Ok(tokens)
}

fn decode_tokens(tokenizer: &Tokenizer, tokens: &[u32]) -> Result<String> {
    tokenizer
        .decode(tokens, true)
        .map_err(|err| anyhow::anyhow!("decode generated tokens: {err}"))
}

fn decode_token_lossy(tokenizer: &Tokenizer, token_id: u32) -> String {
    tokenizer
        .decode(&[token_id], false)
        .unwrap_or_else(|_| format!("<token:{token_id}>"))
}

fn greedy_generation_args(max_new_tokens: usize, enable_thinking: bool) -> GenerationArgs {
    GenerationArgs {
        max_new_tokens,
        temperature: 0.0,
        top_p: None,
        top_k: None,
        seed: 299792458,
        repeat_penalty: 1.0,
        repeat_last_n: 64,
        enable_thinking,
    }
}

#[derive(Clone, Debug)]
struct DraftCorruption {
    index: usize,
    original_token_id: u32,
    corrupted_token_id: u32,
}

fn corrupt_draft_token(
    draft_tokens: &mut [u32],
    index: usize,
    vocab_size: usize,
) -> Result<DraftCorruption> {
    let token = draft_tokens
        .get_mut(index)
        .with_context(|| format!("--corrupt-draft-at {index} is outside the draft sequence"))?;
    let original = *token;
    let corrupted = if (original as usize) + 1 < vocab_size {
        original + 1
    } else {
        original.saturating_sub(1)
    };
    if corrupted == original {
        anyhow::bail!("cannot corrupt draft token {original} with vocab size {vocab_size}");
    }
    *token = corrupted;
    Ok(DraftCorruption {
        index,
        original_token_id: original,
        corrupted_token_id: corrupted,
    })
}

fn apply_confidence_schedule(
    draft_tokens: &mut Vec<u32>,
    confidences: &[f64],
    threshold: Option<f64>,
) -> Result<Option<ConfidenceSchedule>> {
    let Some(threshold) = threshold else {
        if !confidences.is_empty() {
            anyhow::bail!("--draft-confidence requires --schedule-confidence-threshold");
        }
        return Ok(None);
    };
    if !(0.0..=1.0).contains(&threshold) {
        anyhow::bail!("--schedule-confidence-threshold must be between 0 and 1");
    }
    if confidences.len() < draft_tokens.len() {
        anyhow::bail!(
            "got {} draft confidences for {} draft tokens",
            confidences.len(),
            draft_tokens.len()
        );
    }
    if let Some(confidence) = confidences
        .iter()
        .find(|confidence| !(0.0..=1.0).contains(*confidence))
    {
        anyhow::bail!("draft confidence {confidence} is outside 0..=1");
    }

    let original_draft_tokens = draft_tokens.len();
    let (scheduled_draft_tokens, scheduled_cumulative_confidence, next_rejected) =
        schedule_prefix_len(&confidences[..original_draft_tokens], threshold);
    draft_tokens.truncate(scheduled_draft_tokens);
    Ok(Some(ConfidenceSchedule {
        threshold,
        original_draft_tokens,
        scheduled_draft_tokens,
        dropped_draft_tokens: original_draft_tokens.saturating_sub(scheduled_draft_tokens),
        scheduled_cumulative_confidence,
        next_rejected_cumulative_confidence: next_rejected,
        confidences: confidences[..original_draft_tokens].to_vec(),
    }))
}

fn schedule_prefix_len(confidences: &[f64], threshold: f64) -> (usize, f64, Option<f64>) {
    let mut cumulative = 1.0f64;
    let mut accepted = 0usize;
    for confidence in confidences {
        let next = cumulative * confidence;
        if next < threshold {
            return (accepted, cumulative, Some(next));
        }
        cumulative = next;
        accepted += 1;
    }
    (accepted, cumulative, None)
}

fn confidence_schedule_json(schedule: &Option<ConfidenceSchedule>) -> serde_json::Value {
    match schedule {
        Some(schedule) => serde_json::json!({
            "threshold": schedule.threshold,
            "original_draft_tokens": schedule.original_draft_tokens,
            "scheduled_draft_tokens": schedule.scheduled_draft_tokens,
            "dropped_draft_tokens": schedule.dropped_draft_tokens,
            "scheduled_cumulative_confidence": schedule.scheduled_cumulative_confidence,
            "next_rejected_cumulative_confidence": schedule.next_rejected_cumulative_confidence,
            "confidences": schedule.confidences,
        }),
        None => serde_json::Value::Null,
    }
}

fn verify_greedy_draft(
    model: &mut MiniCpmForConditionalGeneration,
    device: &Device,
    prompt_tokens: &[u32],
    draft_tokens: &[u32],
    downsample_mode: &str,
) -> Result<SpecVerifyStats> {
    model.clear_cache();
    let prompt_input = Tensor::from_slice(prompt_tokens, (1, prompt_tokens.len()), device)?;
    let prefill_start = Instant::now();
    let prompt_logits =
        model.forward(&prompt_input, None::<&ProcessedImages>, downsample_mode, 0)?;
    device.synchronize()?;
    let prefill_elapsed = prefill_start.elapsed();

    let (first_target_token, first_argmax_elapsed) = argmax_token(&prompt_logits, device)?;
    let mut argmax_elapsed = first_argmax_elapsed;
    let mut target_token_ids = Vec::with_capacity(draft_tokens.len());
    if draft_tokens.is_empty() {
        let analysis = analyze_verification(draft_tokens, &target_token_ids, first_target_token)?;
        return Ok(SpecVerifyStats {
            analysis,
            target_token_ids,
            prefill_elapsed,
            verify_elapsed: Duration::ZERO,
            argmax_elapsed,
        });
    }
    target_token_ids.push(first_target_token);

    let draft_input = Tensor::from_slice(draft_tokens, (1, draft_tokens.len()), device)?;
    let verify_start = Instant::now();
    let draft_logits = model.forward_all_logits(
        &draft_input,
        None::<&ProcessedImages>,
        downsample_mode,
        prompt_tokens.len(),
    )?;
    device.synchronize()?;
    let verify_elapsed = verify_start.elapsed();
    let (chunk_target_tokens, chunk_argmax_elapsed) = argmax_tokens(&draft_logits, device)?;
    argmax_elapsed += chunk_argmax_elapsed;

    if chunk_target_tokens.len() != draft_tokens.len() {
        anyhow::bail!(
            "verifier chunk returned {} target tokens for {} draft tokens",
            chunk_target_tokens.len(),
            draft_tokens.len()
        );
    }
    target_token_ids.extend(
        chunk_target_tokens
            .iter()
            .take(draft_tokens.len().saturating_sub(1))
            .copied(),
    );
    let bonus_after_all = chunk_target_tokens
        .last()
        .copied()
        .context("missing bonus token after draft chunk")?;
    let analysis = analyze_verification(draft_tokens, &target_token_ids, bonus_after_all)?;

    Ok(SpecVerifyStats {
        analysis,
        target_token_ids,
        prefill_elapsed,
        verify_elapsed,
        argmax_elapsed,
    })
}

fn analyze_verification(
    draft_tokens: &[u32],
    target_token_ids: &[u32],
    bonus_after_all: u32,
) -> Result<SpecVerifyAnalysis> {
    if draft_tokens.len() != target_token_ids.len() {
        anyhow::bail!(
            "draft length {} does not match target token length {}",
            draft_tokens.len(),
            target_token_ids.len()
        );
    }

    let accepted_tokens = draft_tokens
        .iter()
        .zip(target_token_ids.iter())
        .take_while(|(draft, target)| draft == target)
        .count();
    let first_rejected_index = (accepted_tokens < draft_tokens.len()).then_some(accepted_tokens);
    let bonus_token_id = first_rejected_index
        .map(|idx| target_token_ids[idx])
        .unwrap_or(bonus_after_all);
    let mut reconstructed_token_ids = draft_tokens[..accepted_tokens].to_vec();
    reconstructed_token_ids.push(bonus_token_id);

    let positions = draft_tokens
        .iter()
        .zip(target_token_ids.iter())
        .enumerate()
        .map(|(index, (draft, target))| SpecVerifyPosition {
            index,
            draft_token_id: *draft,
            target_token_id: *target,
            token_match: draft == target,
            accepted: index < accepted_tokens,
            first_rejected: first_rejected_index == Some(index),
        })
        .collect();

    Ok(SpecVerifyAnalysis {
        positions,
        accepted_tokens,
        first_rejected_index,
        bonus_token_id,
        reconstructed_token_ids,
    })
}

fn is_greedy_generation(generation: &GenerationArgs) -> bool {
    generation.temperature <= 0.0
}

fn sample_next_token(
    logits_processor: &mut Option<LogitsProcessor>,
    generation: &GenerationArgs,
    logits: &Tensor,
) -> Result<u32> {
    if is_greedy_generation(generation) {
        Ok(logits.argmax(D::Minus1)?.to_scalar::<u32>()?)
    } else {
        Ok(logits_processor
            .as_mut()
            .context("non-greedy generation requires a logits processor")?
            .sample(logits)?)
    }
}

fn generate_tokens(
    model: &mut MiniCpmForConditionalGeneration,
    device: &Device,
    generation: &GenerationArgs,
    input_tokens: &[u32],
    images: Option<&ProcessedImages>,
    downsample_mode: &str,
    eos_ids: &[u32],
    mut on_token: impl FnMut(u32, usize, Duration, Duration) -> Result<()>,
) -> Result<GenerationStats> {
    model.clear_cache();
    let mut tokens = input_tokens.to_vec();
    let mut logits_processor = (!is_greedy_generation(generation)).then(|| {
        LogitsProcessor::from_sampling(
            generation.seed,
            sampling(generation.temperature, generation.top_k, generation.top_p),
        )
    });

    let input = Tensor::from_slice(&tokens, (1, tokens.len()), device)?;
    let prefill_start = Instant::now();
    let mut logits = model.forward(&input, images, downsample_mode, 0)?;
    device.synchronize()?;
    let prefill_elapsed = prefill_start.elapsed();
    let mut position = tokens.len();
    let decode_start = Instant::now();
    let mut first_token_after_prefill = None::<Duration>;
    let mut generated = 0usize;
    let mut generated_token_ids = Vec::with_capacity(generation.max_new_tokens);
    let mut eos_reached = false;
    let mut decode_model_elapsed = Duration::ZERO;
    let mut sampling_elapsed = Duration::ZERO;
    let mut next_input_elapsed = Duration::ZERO;
    let mut callback_elapsed = Duration::ZERO;

    for _ in 0..generation.max_new_tokens {
        let sampling_start = Instant::now();
        let logits_1d = logits.squeeze(0)?;
        let logits_1d = if (generation.repeat_penalty - 1.0).abs() < f32::EPSILON {
            logits_1d
        } else {
            let start_at = tokens.len().saturating_sub(generation.repeat_last_n);
            apply_repeat_penalty(&logits_1d, generation.repeat_penalty, &tokens[start_at..])?
        };
        let next_token = sample_next_token(&mut logits_processor, generation, &logits_1d)?;
        sampling_elapsed += sampling_start.elapsed();

        if eos_ids.contains(&next_token) {
            eos_reached = true;
            break;
        }

        if first_token_after_prefill.is_none() {
            first_token_after_prefill = Some(decode_start.elapsed());
        }
        tokens.push(next_token);
        generated_token_ids.push(next_token);
        generated += 1;
        let callback_start = Instant::now();
        on_token(
            next_token,
            generated,
            decode_start.elapsed(),
            prefill_elapsed,
        )?;
        callback_elapsed += callback_start.elapsed();

        if generated == generation.max_new_tokens {
            break;
        }

        let input_start = Instant::now();
        let input = Tensor::from_slice(&[next_token], (1, 1), device)?;
        next_input_elapsed += input_start.elapsed();
        let decode_model_start = Instant::now();
        logits = model.forward(&input, None::<&ProcessedImages>, downsample_mode, position)?;
        device.synchronize()?;
        decode_model_elapsed += decode_model_start.elapsed();
        position += 1;
    }

    Ok(GenerationStats {
        prompt_tokens: input_tokens.len(),
        generated_tokens: generated,
        generated_token_ids,
        eos_reached,
        prefill_elapsed,
        decode_elapsed: decode_start.elapsed(),
        decode_model_elapsed,
        sampling_elapsed,
        next_input_elapsed,
        callback_elapsed,
        first_token_after_prefill,
    })
}

fn split_reasoning_text(raw_text: &str, enable_thinking: bool) -> ReasoningParts {
    let initial_channel = if enable_thinking {
        TextChannel::Reasoning
    } else {
        TextChannel::Answer
    };
    let mut parser = ReasoningTagParser::new(initial_channel);
    let mut reasoning_text = String::new();
    let mut answer_text = String::new();
    for event in parser.feed(raw_text).into_iter().chain(parser.finish()) {
        match event {
            TextEvent::Text(TextChannel::Reasoning, text) => reasoning_text.push_str(&text),
            TextEvent::Text(TextChannel::Answer, text) => answer_text.push_str(&text),
        }
    }
    ReasoningParts {
        raw_text: raw_text.to_string(),
        reasoning_text,
        answer_text,
    }
}

fn print_reasoning_parts(parts: &ReasoningParts) -> Result<()> {
    if !parts.reasoning_text.trim().is_empty() {
        println!("Reasoning:");
        println!("{}", parts.reasoning_text.trim_end());
        println!();
    }
    if !parts.answer_text.trim().is_empty() {
        println!("Answer:");
        println!("{}", parts.answer_text.trim_end());
    }
    std::io::stdout().flush().ok();
    Ok(())
}

fn prepare_run_prompt(
    args: &RunArgs,
    preprocessor: Option<&PreprocessorConfig>,
    processed_images: Option<&ProcessedImages>,
) -> Result<String> {
    let text = chat_prompt(
        &args.prompt,
        args.images.len(),
        args.generation.enable_thinking,
    );
    match (preprocessor, processed_images) {
        (Some(preprocessor), Some(images)) => expand_image_placeholders(
            text,
            images,
            preprocessor.use_image_id,
            &args.model.downsample_mode,
        ),
        _ => Ok(text),
    }
}

fn bench_prompts(args: &BenchArgs) -> Vec<BenchPrompt> {
    let profiles = if args.profiles.is_empty() && args.prompts.is_empty() {
        BenchProfile::all().to_vec()
    } else {
        args.profiles.clone()
    };
    let mut prompts = profiles
        .into_iter()
        .map(|profile| BenchPrompt {
            name: profile.name().to_string(),
            text: profile.prompt().to_string(),
        })
        .collect::<Vec<_>>();

    prompts.extend(
        args.prompts
            .iter()
            .enumerate()
            .map(|(idx, prompt)| BenchPrompt {
                name: format!("custom-{}", idx + 1),
                text: prompt.clone(),
            }),
    );
    prompts
}

fn benchmark_writer(path: Option<&PathBuf>, append: bool) -> Result<Box<dyn Write>> {
    match path {
        Some(path) => {
            let mut options = OpenOptions::new();
            options.create(true).write(true);
            if append {
                options.append(true);
            } else {
                options.truncate(true);
            }
            let file = options
                .open(path)
                .with_context(|| format!("open benchmark output {}", path.display()))?;
            Ok(Box::new(BufWriter::new(file)))
        }
        None => Ok(Box::new(BufWriter::new(std::io::stdout()))),
    }
}

fn write_json_report(path: Option<&PathBuf>, value: &serde_json::Value) -> Result<()> {
    match path {
        Some(path) => {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(path)
                .with_context(|| format!("open report output {}", path.display()))?;
            serde_json::to_writer_pretty(&mut file, value)?;
            file.write_all(b"\n")?;
        }
        None => {
            let stdout = std::io::stdout();
            let mut writer = BufWriter::new(stdout.lock());
            serde_json::to_writer_pretty(&mut writer, value)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
    }
    Ok(())
}

fn quantized_load_json(load: &Option<QuantizedLoadStats>) -> serde_json::Value {
    match load {
        Some(load) => serde_json::json!({
            "manifest": load.manifest,
            "quantized_tensors": load.quantized_tensors,
            "replaced_text_linears": load.replaced_text_linears,
            "backend": load.backend,
            "quantized_data_bytes": load.quantized_data_bytes,
            "dense_equivalent_bytes": load.dense_equivalent_bytes,
            "approx_dense_bytes_avoided": load.dense_equivalent_bytes.saturating_sub(load.quantized_data_bytes as usize),
        }),
        None => serde_json::Value::Null,
    }
}

#[derive(Debug)]
struct TopLogitComparison {
    top1_match: bool,
    top_k_overlap: usize,
    top_k_overlap_threshold: usize,
    max_abs_shared_logit_delta: Option<f32>,
    shared_logit_deltas: Vec<serde_json::Value>,
    passed: bool,
}

fn compare_top_logits(
    candle_top: &[TopLogit],
    oracle_token_ids: &[u32],
    oracle_logits: &[f32],
) -> TopLogitComparison {
    let oracle_by_id = oracle_token_ids
        .iter()
        .zip(oracle_logits.iter())
        .map(|(token_id, logit)| (*token_id, *logit))
        .collect::<HashMap<_, _>>();
    let oracle_set = oracle_by_id
        .keys()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let top_k_overlap = candle_top
        .iter()
        .filter(|item| oracle_set.contains(&item.token_id))
        .count();
    let top_k_overlap_threshold = candle_top.len().min(oracle_token_ids.len()).min(8);
    let top1_match =
        candle_top.first().map(|item| item.token_id) == oracle_token_ids.first().copied();
    let mut max_abs_shared_logit_delta = None::<f32>;
    let mut shared_logit_deltas = Vec::new();
    for item in candle_top {
        if let Some(oracle_logit) = oracle_by_id.get(&item.token_id) {
            let delta = item.logit - oracle_logit;
            let abs_delta = delta.abs();
            max_abs_shared_logit_delta = Some(
                max_abs_shared_logit_delta
                    .map(|current| current.max(abs_delta))
                    .unwrap_or(abs_delta),
            );
            shared_logit_deltas.push(serde_json::json!({
                "token_id": item.token_id,
                "token": item.token,
                "candle_logit": item.logit,
                "transformers_logit": oracle_logit,
                "delta": delta,
                "abs_delta": abs_delta,
            }));
        }
    }
    let passed = top1_match && top_k_overlap >= top_k_overlap_threshold;
    TopLogitComparison {
        top1_match,
        top_k_overlap,
        top_k_overlap_threshold,
        max_abs_shared_logit_delta,
        shared_logit_deltas,
        passed,
    }
}

fn top_k_logits(logits: &Tensor, top_k: usize, tokenizer: &Tokenizer) -> Result<Vec<TopLogit>> {
    let values = logits
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    let mut indexed = values
        .into_iter()
        .enumerate()
        .map(|(token_id, logit)| (token_id as u32, logit))
        .collect::<Vec<_>>();
    indexed.sort_by(|(_, left), (_, right)| right.total_cmp(left));
    indexed.truncate(top_k);
    indexed
        .into_iter()
        .map(|(token_id, logit)| {
            let token = tokenizer
                .decode(&[token_id], false)
                .map_err(|err| anyhow::anyhow!("decode token {token_id}: {err}"))?;
            Ok(TopLogit {
                token_id,
                token,
                logit,
            })
        })
        .collect()
}

fn top_logits_json(top_logits: &[TopLogit]) -> Vec<serde_json::Value> {
    top_logits
        .iter()
        .map(|item| {
            serde_json::json!({
                "token_id": item.token_id,
                "token": item.token,
                "logit": item.logit,
            })
        })
        .collect()
}

fn trace_capture_layers(requested: &[usize], num_layers: usize) -> Result<Vec<usize>> {
    if num_layers == 0 {
        anyhow::bail!("model has zero text layers");
    }
    let mut layers = if requested.is_empty() {
        vec![0, (num_layers - 1) / 2, num_layers - 1]
    } else {
        requested.to_vec()
    };
    layers.sort_unstable();
    layers.dedup();
    if let Some(layer) = layers.iter().find(|layer| **layer >= num_layers) {
        anyhow::bail!("capture layer {layer} is outside 0..{}", num_layers - 1);
    }
    Ok(layers)
}

fn argmax_token(logits: &Tensor, device: &Device) -> Result<(u32, Duration)> {
    device.synchronize()?;
    let started = Instant::now();
    let token = logits.squeeze(0)?.argmax(D::Minus1)?.to_scalar::<u32>()?;
    device.synchronize()?;
    Ok((token, started.elapsed()))
}

fn argmax_tokens(logits: &Tensor, device: &Device) -> Result<(Vec<u32>, Duration)> {
    device.synchronize()?;
    let started = Instant::now();
    let tokens = logits
        .squeeze(0)?
        .argmax(D::Minus1)?
        .to_device(&Device::Cpu)?
        .to_vec1::<u32>()?;
    device.synchronize()?;
    Ok((tokens, started.elapsed()))
}

fn aggregate_profile_events(events: &[Qwen35ProfileEvent]) -> Vec<serde_json::Value> {
    let mut groups = HashMap::<String, (usize, f64)>::new();
    for event in events {
        let entry = groups.entry(event.component.clone()).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += event.seconds;
    }
    aggregate_groups(groups)
}

fn aggregate_profile_events_by_layer_kind(events: &[Qwen35ProfileEvent]) -> Vec<serde_json::Value> {
    let mut groups = HashMap::<String, (usize, f64)>::new();
    for event in events {
        let key = event
            .layer_kind
            .clone()
            .unwrap_or_else(|| "unlayered".to_string());
        let entry = groups.entry(key).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += event.seconds;
    }
    aggregate_groups(groups)
}

fn aggregate_groups(groups: HashMap<String, (usize, f64)>) -> Vec<serde_json::Value> {
    let total = groups.values().map(|(_, seconds)| *seconds).sum::<f64>();
    let mut rows = groups
        .into_iter()
        .map(|(name, (count, seconds))| {
            serde_json::json!({
                "name": name,
                "count": count,
                "seconds": seconds,
                "avg_ms": if count > 0 { seconds * 1000.0 / count as f64 } else { 0.0 },
                "share": if total > 0.0 { seconds / total } else { 0.0 },
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        let left = left["seconds"].as_f64().unwrap_or(0.0);
        let right = right["seconds"].as_f64().unwrap_or(0.0);
        right.total_cmp(&left)
    });
    rows
}

fn sampling(temperature: f64, top_k: Option<usize>, top_p: Option<f64>) -> Sampling {
    if temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        match (top_k, top_p) {
            (None, None) => Sampling::All { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    }
}

fn secs(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn tokens_per_second(tokens: usize, duration: Duration) -> f64 {
    if tokens == 0 || duration.is_zero() {
        0.0
    } else {
        tokens as f64 / duration.as_secs_f64()
    }
}

fn safe_prefix_len(text: &str, keep_bytes: usize) -> usize {
    let mut idx = text.len().saturating_sub(keep_bytes);
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn text_scroll(text: &str, height: u16) -> u16 {
    let visible_lines = height.saturating_sub(2).max(1) as usize;
    let line_count = text.lines().count().max(1);
    line_count.saturating_sub(visible_lines) as u16
}

fn select_device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(device) => Ok(device),
            Err(err) => {
                eprintln!("Metal unavailable ({err}); falling back to CPU");
                Ok(Device::Cpu)
            }
        }
    }
    #[cfg(not(feature = "metal"))]
    {
        Ok(Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bench_args(profiles: Vec<BenchProfile>, prompts: Vec<String>) -> BenchArgs {
        BenchArgs {
            model: ModelArgs {
                model_id: "model".to_string(),
                revision: "main".to_string(),
                downsample_mode: "16x".to_string(),
                cpu: true,
                dtype: DTypeArg::Auto,
                config: None,
                tokenizer: None,
                generation_config: None,
                preprocessor: None,
                weights: Vec::new(),
                quantized_manifest: None,
            },
            generation: GenerationArgs {
                max_new_tokens: 128,
                temperature: 0.0,
                top_p: None,
                top_k: None,
                seed: 299792458,
                repeat_penalty: 1.0,
                repeat_last_n: 64,
                enable_thinking: false,
            },
            profiles,
            prompts,
            warmup: 1,
            iterations: 3,
            output: None,
            append: false,
        }
    }

    #[test]
    fn bench_prompts_defaults_to_all_profiles() {
        let prompts = bench_prompts(&bench_args(Vec::new(), Vec::new()));
        let names = prompts
            .into_iter()
            .map(|prompt| prompt.name)
            .collect::<Vec<_>>();
        assert_eq!(names, ["short", "medium", "long"]);
    }

    #[test]
    fn bench_prompts_custom_prompt_does_not_add_defaults() {
        let prompts = bench_prompts(&bench_args(Vec::new(), vec!["hello".to_string()]));
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].name, "custom-1");
        assert_eq!(prompts[0].text, "hello");
    }

    #[test]
    fn greedy_generation_is_temperature_driven() {
        let mut args = bench_args(Vec::new(), Vec::new()).generation;
        args.temperature = 0.0;
        args.top_k = Some(4);
        args.top_p = Some(0.8);
        assert!(is_greedy_generation(&args));

        args.temperature = 0.7;
        assert!(!is_greedy_generation(&args));
    }

    #[test]
    fn generation_stats_separates_model_and_non_model_decode_time() {
        let stats = GenerationStats {
            prompt_tokens: 10,
            generated_tokens: 4,
            generated_token_ids: vec![1, 2, 3, 4],
            eos_reached: false,
            prefill_elapsed: Duration::from_millis(20),
            decode_elapsed: Duration::from_millis(100),
            decode_model_elapsed: Duration::from_millis(70),
            sampling_elapsed: Duration::from_millis(12),
            next_input_elapsed: Duration::from_millis(3),
            callback_elapsed: Duration::from_millis(5),
            first_token_after_prefill: Some(Duration::from_millis(10)),
        };

        assert_eq!(stats.decode_model_tokens(), 3);
        assert_eq!(stats.decode_model_tokens_per_second(), Some(3000.0 / 70.0));
        assert_eq!(stats.decode_non_model_elapsed(), Duration::from_millis(30));
        assert_eq!(
            stats.decode_bookkeeping_elapsed(),
            Duration::from_millis(10)
        );
    }

    #[test]
    fn quality_comparison_accepts_exact_token_match() {
        let thresholds = QualityThresholds {
            min_prefix_ratio: 0.9,
            min_token_jaccard: 0.9,
            min_lexical_jaccard: 0.9,
            max_length_ratio_delta: 0.0,
        };
        let comparison = compare_quality_outputs(
            &[1, 2, 3],
            "Paris is the capital.",
            &[1, 2, 3],
            "Paris is the capital.",
            &thresholds,
        );

        assert!(comparison.exact_token_match);
        assert_eq!(comparison.common_prefix_tokens, 3);
        assert_eq!(comparison.divergence_index, None);
        assert!(comparison.passed_gate);
    }

    #[test]
    fn quality_comparison_reports_divergence_and_overlap() {
        let thresholds = QualityThresholds {
            min_prefix_ratio: 0.25,
            min_token_jaccard: 0.25,
            min_lexical_jaccard: 0.25,
            max_length_ratio_delta: 1.0,
        };
        let comparison = compare_quality_outputs(
            &[10, 11, 12, 13],
            "the answer is paris",
            &[10, 99, 12],
            "the answer is london",
            &thresholds,
        );

        assert!(!comparison.exact_token_match);
        assert_eq!(comparison.common_prefix_tokens, 1);
        assert_eq!(comparison.divergence_index, Some(1));
        assert!(comparison.token_jaccard > 0.0);
        assert!(comparison.lexical_jaccard > 0.0);
        assert!(comparison.passed_gate);
    }

    #[test]
    fn verifier_analysis_accepts_full_draft_and_bonus() {
        let analysis = analyze_verification(&[10, 11, 12], &[10, 11, 12], 13).unwrap();

        assert_eq!(analysis.accepted_tokens, 3);
        assert_eq!(analysis.first_rejected_index, None);
        assert_eq!(analysis.bonus_token_id, 13);
        assert_eq!(analysis.reconstructed_token_ids, [10, 11, 12, 13]);
        assert_eq!(analysis.verifier_waste_tokens(), 0);
        assert_eq!(analysis.acceptance_rate(), Some(1.0));
    }

    #[test]
    fn verifier_analysis_rejects_at_first_mismatch() {
        let analysis = analyze_verification(&[10, 99, 12, 13], &[10, 11, 55, 56], 57).unwrap();

        assert_eq!(analysis.accepted_tokens, 1);
        assert_eq!(analysis.first_rejected_index, Some(1));
        assert_eq!(analysis.bonus_token_id, 11);
        assert_eq!(analysis.reconstructed_token_ids, [10, 11]);
        assert_eq!(analysis.verifier_waste_tokens(), 2);
        assert_eq!(analysis.verifier_waste_share(), Some(0.5));
        assert!(analysis.positions[1].first_rejected);
        assert!(!analysis.positions[2].accepted);
    }

    #[test]
    fn confidence_scheduler_truncates_before_threshold_drop() {
        let (len, cumulative, next) = schedule_prefix_len(&[0.9, 0.9, 0.9], 0.75);

        assert_eq!(len, 2);
        assert!((cumulative - 0.81).abs() < 1e-9);
        assert!(next.is_some_and(|value| (value - 0.729).abs() < 1e-9));
    }

    #[test]
    fn confidence_scheduler_can_drop_all_tokens() {
        let (len, cumulative, next) = schedule_prefix_len(&[0.7, 0.9], 0.8);

        assert_eq!(len, 0);
        assert_eq!(cumulative, 1.0);
        assert_eq!(next, Some(0.7));
    }

    #[test]
    fn draft_corruption_changes_selected_token() {
        let mut draft = vec![7, 8, 9];
        let corruption = corrupt_draft_token(&mut draft, 1, 16).unwrap();

        assert_eq!(corruption.index, 1);
        assert_eq!(corruption.original_token_id, 8);
        assert_eq!(corruption.corrupted_token_id, 9);
        assert_eq!(draft, [7, 9, 9]);
    }

    #[test]
    fn trace_capture_layers_defaults_to_low_mid_high() {
        let layers = trace_capture_layers(&[], 24).unwrap();
        assert_eq!(layers, [0, 11, 23]);
    }

    #[test]
    fn trace_capture_layers_sorts_dedups_and_validates() {
        let layers = trace_capture_layers(&[7, 3, 7], 8).unwrap();
        assert_eq!(layers, [3, 7]);
        assert!(trace_capture_layers(&[8], 8).is_err());
    }

    #[test]
    fn split_reasoning_text_removes_think_tags() {
        let text = split_reasoning_text("<think>scratch</think>final", false);
        assert_eq!(text.raw_text, "<think>scratch</think>final");
        assert_eq!(text.reasoning_text, "scratch");
        assert_eq!(text.answer_text, "final");
    }

    #[test]
    fn split_reasoning_text_can_start_inside_think_block() {
        let text = split_reasoning_text("scratch</think>final", true);
        assert_eq!(text.reasoning_text, "scratch");
        assert_eq!(text.answer_text, "final");
    }

    #[test]
    fn parser_handles_split_reasoning_tag() {
        let mut parser = ReasoningTagParser::new(TextChannel::Answer);
        assert!(parser.feed("<thi").is_empty());
        let events = parser.feed("nk>scratch</think>final");
        let mut reasoning = String::new();
        let mut answer = String::new();
        for event in events.into_iter().chain(parser.finish()) {
            match event {
                TextEvent::Text(TextChannel::Reasoning, text) => reasoning.push_str(&text),
                TextEvent::Text(TextChannel::Answer, text) => answer.push_str(&text),
            }
        }
        assert_eq!(reasoning, "scratch");
        assert_eq!(answer, "final");
    }
}
