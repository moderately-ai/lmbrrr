#![recursion_limit = "256"]

use std::{
    collections::HashMap,
    fs::{self, OpenOptions},
    io::{BufWriter, IsTerminal, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use candle::{
    quantized::{GgmlDType, QTensor},
    DType, Device, Module, Tensor, D,
};
use candle_nn::VarBuilder;
use clap::{Args, Parser, Subcommand, ValueEnum};
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
    qwen35::{Qwen35ProfileEvent, Qwen35Profiler, Qwen35TraceRecorder},
    generate::{
        argmax_token, argmax_tokens, generate_tokens, greedy_generation_args,
        is_greedy_generation, secs, tokens_per_second, GenerationArgs, GenerationStats,
    },
    spec::recycle_topk::logits_argmax_and_topk,
    spec::scheduler::{
        mean_committed_per_round, schedule_prefix_width, RoundCostModel, StsCalibration,
    },
    token_stream::TokenOutputStream,
    tui::{
        print_reasoning_parts, split_reasoning_text, ReasoningRenderer, TextChannel, TuiOutput,
    },
    weights::{validate_minicpm_header, WeightReport},
};
use serde::Deserialize;
use tokenizers::Tokenizer;

mod commands;

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
    Roofline(RooflineArgs),
    VerifyTable(VerifyTableArgs),
    DsparkRun(DsparkRunArgs),
    DsparkDrafterParity(DsparkDrafterParityArgs),
    MultiBench(MultiBenchArgs),
    TreeCheck(TreeCheckArgs),
    VisionCheck(VisionCheckArgs),
    FakequantExport(FakequantExportArgs),
}

/// Deployment-config trace-generator prep: rewrite the HF checkpoint with
/// every q4k-full-text-policy tensor passed through candle's own Q4_K
/// quantize->dequantize, so CUDA-side data generation carries the deployed
/// target's weight noise. The lm_head stays dense (tied embedding;
/// deployment quantizes only the runtime head copy — residual mismatch,
/// documented in the corpus-scaling plan).
#[derive(Parser, Debug)]
struct FakequantExportArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(long, default_value = "target/minicpm-v46-fakequant-q4kft")]
    output_dir: PathBuf,
}

/// Vision feature parity gate: merged image features (vision tower + merger)
/// for the oracle's deterministic synthetic images vs the Transformers
/// fixture (evals/fixtures/minicpm_v46_transformers_image_features.json).
#[derive(Parser, Debug)]
struct VisionCheckArgs {
    #[command(flatten)]
    model: ModelArgs,

    /// Max |Δ| per sampled feature (bf16-CPU oracle vs bf16-Metal tower).
    #[arg(long, default_value_t = 0.125)]
    max_abs_delta: f32,

    /// Max mean |Δ| across each case's samples.
    #[arg(long, default_value_t = 0.02)]
    max_mean_delta: f32,
}

/// Tree-vs-chain equivalence gate for two-branch tree verification: the
/// flattened tree forward's main-branch rows must match a plain chain verify
/// (same kernel, same state), the alternate rows must match a chain
/// re-advance of the alternate tokens within the rollback noise envelope,
/// and both rollback paths must land on states indistinguishable from a
/// chain-built state.
#[derive(Parser, Debug)]
struct TreeCheckArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(long, default_value = "Explain how tides work.")]
    prompt: String,

    #[arg(long, default_value_t = 3)]
    branch_width: usize,

    #[arg(long, default_value_t = 4)]
    rounds: usize,

    /// Max |Δlogit| for main-branch rows (identical dispatch expected).
    #[arg(long, default_value_t = 0.02)]
    main_eps: f32,

    /// Max |Δlogit| for alternate rows and post-rollback probes (closed-form
    /// branch seeding vs chain re-advance; same envelope as partial-accept
    /// rollback).
    #[arg(long, default_value_t = 0.75)]
    alt_eps: f32,
}

/// Static-batched multi-stream greedy decode: N copies of the prompt run as
/// one batch through the whole text path; reports aggregate and per-stream
/// rates plus a single-stream-equivalence check on stream 0.
#[derive(Parser, Debug)]
struct MultiBenchArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(long)]
    prompt: String,

    /// Stream counts to sweep, comma separated; the model loads once.
    #[arg(long, value_delimiter = ',', default_values_t = [8usize])]
    streams: Vec<usize>,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct DsparkDrafterParityArgs {
    #[arg(long, default_value = "target/dspark-drafter-smoke/step_24")]
    checkpoint: PathBuf,

    #[arg(long, default_value = "target/dspark-fixtures/drafter-parity.safetensors")]
    fixture: PathBuf,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    output: Option<PathBuf>,
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

    /// Post-hoc lm_head quantization (quantized copy of the tied embedding;
    /// BF16 table stays for the gather). 508 MB/token head read -> 143 MB at
    /// q4k. Quality advisory per campaign policy.
    #[arg(long, value_enum)]
    quantize_lm_head: Option<DrafterQuantArg>,
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

    /// Fallback-ladder overrides for from-source policies, repeatable:
    /// "<name-suffix>=<rung>" with rung one of q4k/q6k/q8-0. Chosen by the
    /// quality harness where q4k collapses.
    #[arg(long = "fallback")]
    fallback: Vec<String>,
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

    /// Optional q4k-full-text manifest (from-source policy); included in the
    /// ladder when the file exists.
    #[arg(
        long,
        default_value = "target/minicpm-v46-q4k-full-text/manifest.json"
    )]
    full_text_manifest: PathBuf,

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

#[derive(Parser, Debug)]
struct DsparkRunArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(long)]
    prompt: String,

    /// Deployed default: gamma 6 beat 4 and 8 on the strong Spec-Bench
    /// classes with the scheduled round-3 stack (the scheduler truncates
    /// per-round, so gamma is a cap, not a fixed width).
    #[arg(long, default_value_t = 6)]
    gamma: usize,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    /// Corruption periods for the stub drafter, comma separated. Each value
    /// runs the full multi-round loop with every Nth draft token corrupted
    /// (0 = no corruption); all runs must produce identical output, which is
    /// the blocking state-rollback oracle.
    #[arg(long = "corrupt-every", value_delimiter = ',', default_values_t = [0usize, 3, 5])]
    corrupt_every: Vec<usize>,

    /// Run with a trained DSpark drafter checkpoint instead of the stub.
    #[arg(long)]
    drafter: Option<PathBuf>,

    /// Truncate each proposal to the leading positions whose calibrated
    /// confidence stays at or above this probability (DeepSpec inference
    /// contract; 0-draft rounds allowed). Calibration comes from sts.json
    /// in the drafter dir when present. Off when unset: full gamma.
    #[arg(long)]
    confidence_threshold: Option<f32>,

    /// Accept a draft token whose target logit is within this margin of the
    /// target's top logit (Medusa-style typical acceptance) instead of
    /// requiring exact argmax match. Changes outputs; quality is reported,
    /// not gated (campaign policy). Exact match when unset.
    #[arg(long)]
    accept_margin: Option<f32>,

    /// Post-hoc quantization tier for the drafter's 248k-vocab heads
    /// (lm_head + markov_w2). Cuts draft cost ~2x; risk surface is tau only
    /// (drafts are target-verified).
    #[arg(long, value_enum)]
    drafter_quantize: Option<DrafterQuantArg>,

    /// FR-Spec draft-vocabulary artifact: JSON with an "ids" array of
    /// frequency-ranked token ids. Drafting argmaxes only over these rows
    /// (lm_head + markov_w2 sliced at load); committed output is unaffected
    /// (verification is exact) — only draft cost and tau move.
    #[arg(long)]
    draft_vocab: Option<PathBuf>,

    /// Hardware-aware prefix scheduling (paper Appendix A): per-round
    /// admission maximizing expected tokens/sec from calibrated cumulative
    /// survival and the measured round-cost table. Supersedes
    /// --confidence-threshold when set. Deployed default: on (ablate with
    /// --schedule=false); the stub-oracle path ignores it.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    schedule: bool,

    /// Round-cost artifact for the scheduler (target/spec-round-cost-model
    /// .json shape). Falls back to the built-in measured defaults.
    #[arg(long)]
    cost_model: Option<PathBuf>,

    /// Override the fixed per-round host cost (ms) in the scheduler's
    /// throughput objective: T_round = T_fixed + T_draft + T_verify(w+1).
    /// Overrides the artifact's fixed_round_ms for A/B.
    #[arg(long)]
    cost_model_fixed_ms: Option<f64>,

    /// Two-branch tree verification: draft the runner-up branch at position
    /// 1 (propose_branching) and verify both paths in one flattened forward,
    /// committing the longer-accepted one. Caps effective width at 5 so the
    /// flattened chunk fits the l <= 12 kernel.
    #[arg(long)]
    tree: bool,

    /// Branch only when the calibrated position-0 survival lands in this
    /// inclusive band (lo,hi): outside it the runner-up carries too little
    /// mass (high) or the whole draft is doomed (low). Chain rounds
    /// otherwise.
    #[arg(long, value_delimiter = ',', default_values_t = [0.0f32, 1.0])]
    tree_band: Vec<f32>,

    /// Prompt-lookup drafting: when the trailing n-gram of the committed
    /// sequence matches an earlier occurrence in prompt+history, propose the
    /// tokens that followed it (verified exactly like any draft). Fires only
    /// on a match, so the greedy floor is preserved; the trained drafter
    /// handles non-matching rounds. Deployed default: on (--pld=false to
    /// ablate).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    pld: bool,

    /// Max tokens proposed per prompt-lookup match (chunk cap is 11).
    /// Default 2: measured, accepted runs are short even on copy-heavy text,
    /// so wider proposals only add rejected-tail verify cost.
    #[arg(long, default_value_t = 2)]
    pld_span: usize,

    /// Fire prompt-lookup on every match instead of only where the scheduler
    /// has given up on drafting. Ungated PLD preempts strong drafter rounds
    /// (measured net loss on math); the default gates PLD to skip rounds.
    #[arg(long)]
    pld_ungated: bool,

    /// Verify-logit token recycling: bank top-k candidates from every verify
    /// pass and, when no prompt-lookup match fires, chain the banked top-1
    /// through rows whose logit margin clears --recycle-margin. Same gate as
    /// PLD (fires only where the scheduler skips drafting). Deployed
    /// default: on (--recycle=false to ablate).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    recycle: bool,

    /// Max recycled chain depth (chained acceptance compounds; keep short).
    #[arg(long, default_value_t = 2)]
    recycle_depth: usize,

    /// Minimum banked top-1/top-2 logit margin to extend a recycled chain.
    /// The viability condition: a depth-1 draft needs ~77% acceptance to
    /// beat a greedy step on the measured cost model, which only
    /// near-deterministic (large-margin) continuations clear.
    #[arg(long, default_value_t = 6.0)]
    recycle_margin: f32,

    /// Candidates banked per verify row.
    #[arg(long, default_value_t = 8)]
    recycle_topk: usize,

    /// Smallest n-gram that may trigger a lookup match (2..=4).
    #[arg(long, default_value_t = 3)]
    pld_min_ngram: usize,

    #[arg(long)]
    enable_thinking: bool,

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
    Q4kFullText,
}

impl MixedPrecisionPolicyArg {
    fn resolve(self) -> MixedPrecisionPolicy {
        match self {
            Self::Q8TextLinears => MixedPrecisionPolicy::Q8TextLinears,
            Self::Q4kMlpOnly => MixedPrecisionPolicy::Q4KMlpOnly,
            Self::Q4kTextSafe => MixedPrecisionPolicy::Q4KTextSafe,
            Self::Q4kMlpQ8Text => MixedPrecisionPolicy::Q4KMlpQ8Text,
            Self::Q4kFullText => MixedPrecisionPolicy::Q4KFullText,
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(args),
        Command::Bench(args) => bench(args),
        Command::Logits(args) => logits(args),
        Command::Profile(args) => profile_decode(args),
        Command::SpecVerify(args) => commands::verify::spec_verify(args),
        Command::Trace(args) => trace_hidden_states(args),
        Command::QuantSensitivity(args) => quant_sensitivity(args),
        Command::QuantConvert(args) => quant_convert(args),
        Command::QuantMatmulBench(args) => quant_matmul_bench(args),
        Command::QuantQuality(args) => quant_quality(args),
        Command::Roofline(args) => roofline(args),
        Command::VerifyTable(args) => verify_table(args),
        Command::DsparkRun(args) => commands::dspark::dspark_run(args),
        Command::DsparkDrafterParity(args) => commands::verify::dspark_drafter_parity(args),
        Command::MultiBench(args) => multi_bench(args),
        Command::TreeCheck(args) => commands::verify::tree_check(args),
        Command::VisionCheck(args) => vision_check(args),
        Command::FakequantExport(args) => fakequant_export(args),
    }
}

fn fakequant_export(args: FakequantExportArgs) -> Result<()> {
    use candle::quantized::{GgmlDType, QTensor};

    let bundle = resolve_artifacts(&args.model)?;
    fs::create_dir_all(&args.output_dir)?;

    let eligible = |name: &str, shape: &[usize]| -> bool {
        name.ends_with(".weight")
            && shape.len() == 2
            && name.contains(".layers.")
            && (name.contains(".mlp.")
                || name.contains(".self_attn.")
                || name.contains(".linear_attn."))
            && !name.ends_with(".in_proj_a.weight")
            && !name.ends_with(".in_proj_b.weight")
            && shape[shape.len() - 1] % 256 == 0
    };

    let mut quantized = 0usize;
    let mut passthrough = 0usize;
    let mut max_shift = 0f32;
    for shard in &bundle.artifacts.weights {
        let tensors = candle::safetensors::load(shard, &Device::Cpu)?;
        let mut out = std::collections::HashMap::new();
        for (name, tensor) in tensors {
            if eligible(&name, tensor.dims()) {
                let f32_tensor = tensor.to_dtype(DType::F32)?;
                let q = QTensor::quantize(&f32_tensor, GgmlDType::Q4K)?;
                let restored = q.dequantize(&Device::Cpu)?;
                let shift = (&restored - &f32_tensor)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?;
                max_shift = max_shift.max(shift);
                out.insert(name, restored.to_dtype(tensor.dtype())?);
                quantized += 1;
            } else {
                out.insert(name, tensor);
                passthrough += 1;
            }
        }
        let file_name = shard
            .file_name()
            .context("weight shard has no file name")?;
        candle::safetensors::save(&out, args.output_dir.join(file_name))?;
        println!(
            "wrote {}: {} quantized so far",
            file_name.to_string_lossy(),
            quantized
        );
    }
    // Sidecar configs the generator needs (tokenizer, configs, index).
    if let Some(dir) = bundle.artifacts.weights.first().and_then(|p| p.parent()) {
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            if path.is_file() && !name.ends_with(".safetensors") {
                fs::copy(&path, args.output_dir.join(&name))?;
            }
        }
    }
    // lmbrrr's hub cache never fetches the chat template or tokenizer
    // config (the Rust runner doesn't need them), but transformers-side
    // generation does; backfill from the vendored model dir.
    let vendored = Path::new("docs/research/models/minicpm-v-4.6/hf-model");
    for name in ["chat_template.jinja", "tokenizer_config.json"] {
        let target = args.output_dir.join(name);
        let source = vendored.join(name);
        if !target.exists() && source.exists() {
            fs::copy(&source, &target)?;
        }
    }
    println!(
        "fakequant export: {} quantized, {} passthrough, max |Δw| {:.5} -> {}",
        quantized,
        passthrough,
        max_shift,
        args.output_dir.display()
    );
    Ok(())
}

fn vision_check(args: VisionCheckArgs) -> Result<()> {
    use lmbrrr::image_processor::preprocess_rgb_images;

    #[derive(serde::Deserialize)]
    struct FeatureFixture {
        downsample_mode: String,
        feature_cases: Vec<FeatureCase>,
    }
    #[derive(serde::Deserialize)]
    struct FeatureCase {
        id: String,
        height: usize,
        width: usize,
        feature_shape: Vec<usize>,
        sample_indices: Vec<usize>,
        sample_values: Vec<f32>,
    }

    let fixture: FeatureFixture = serde_json::from_str(include_str!(
        "../evals/fixtures/minicpm_v46_transformers_image_features.json"
    ))?;
    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let preprocessor_path = bundle
        .artifacts
        .preprocessor
        .as_ref()
        .context("vision check requires preprocessor_config.json")?;
    let preprocessor = PreprocessorConfig::from_path(preprocessor_path)?;
    let (model, _, _) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
        args.model.quantize_lm_head,
    )?;

    let mut worst_abs = 0f32;
    let mut worst_mean = 0f32;
    for case in &fixture.feature_cases {
        // Same generator as the oracle's synthetic_image.
        let mut img = image::RgbImage::new(case.width as u32, case.height as u32);
        for y in 0..case.height {
            for x in 0..case.width {
                img.put_pixel(
                    x as u32,
                    y as u32,
                    image::Rgb([
                        ((x * 17 + y * 3) % 256) as u8,
                        ((x * 5 + y * 11 + 37) % 256) as u8,
                        ((x * 13 + y * 7 + 91) % 256) as u8,
                    ]),
                );
            }
        }
        let processed = preprocess_rgb_images(
            &[(PathBuf::from(format!("{}.png", case.id)), img)],
            &preprocessor,
            &device,
        )?;
        let features = model.image_features(&processed, &fixture.downsample_mode, dtype)?;
        let features = Tensor::cat(&features.iter().collect::<Vec<_>>(), 0)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
        anyhow::ensure!(
            features.dims() == case.feature_shape.as_slice(),
            "{}: feature shape {:?} != oracle {:?}",
            case.id,
            features.dims(),
            case.feature_shape
        );
        let flat = features.flatten_all()?.to_vec1::<f32>()?;
        let mut max_d = 0f32;
        let mut sum_d = 0f32;
        for (index, expected) in case.sample_indices.iter().zip(case.sample_values.iter()) {
            let got = flat[*index];
            let d = (got - expected).abs();
            max_d = max_d.max(d);
            sum_d += d;
        }
        let mean_d = sum_d / case.sample_indices.len() as f32;
        println!(
            "{}: shape {:?} ok, max |Δ| {max_d:.4}, mean |Δ| {mean_d:.4}",
            case.id,
            features.dims()
        );
        worst_abs = worst_abs.max(max_d);
        worst_mean = worst_mean.max(mean_d);
    }
    println!(
        "vision-check: worst max |Δ| {worst_abs:.4} (eps {}), worst mean |Δ| {worst_mean:.4} (eps {})",
        args.max_abs_delta, args.max_mean_delta
    );
    if worst_abs > args.max_abs_delta || worst_mean > args.max_mean_delta {
        anyhow::bail!("vision-check FAILED");
    }
    println!("vision-check PASSED");
    Ok(())
}

/// See [`TreeCheckArgs`]. Per round, with the pre-round state snapshotted:
/// the main branch is the greedy continuation, the alternate is the
/// runner-up token at the anchor followed by its greedy continuation. Chain
/// references are built by restoring the snapshot and running plain chunk
/// forwards over the same tokens.

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
        args.model.quantize_lm_head,
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
        args.model.quantize_lm_head,
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
        args.model.quantize_lm_head,
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
        args.model.quantize_lm_head,
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
        args.model.quantize_lm_head,
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
        args.model.quantize_lm_head,
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
        fallback_overrides: args
            .fallback
            .iter()
            .map(|spec| {
                spec.split_once('=')
                    .map(|(suffix, rung)| (suffix.to_string(), rung.to_string()))
                    .ok_or_else(|| anyhow::anyhow!("--fallback wants <suffix>=<rung>, got {spec}"))
            })
            .collect::<Result<Vec<_>>>()?,
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
    let mut policy_specs = vec![
        ("dense", None::<&PathBuf>),
        ("q8-text-linears", Some(&args.q8_manifest)),
        ("q4k-mlp-only", Some(&args.q4_mlp_manifest)),
        ("q4k-text-safe", Some(&args.q4_text_safe_manifest)),
        ("q4k-mlp-q8-text", Some(&args.mixed_manifest)),
    ];
    if args.full_text_manifest.exists() {
        policy_specs.push(("q4k-full-text", Some(&args.full_text_manifest)));
    }

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
        load_model_with_optional_quantization(bundle, dtype, device, quantized_manifest, None)?;
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
        input_cpu.dim(1).unwrap_or(1),
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
    // Route through MixedLinear so the bench measures the DEPLOYED path:
    // bf16_direct kernels for Q8_0/Q4K/Q6K on Metal (F32 accumulate + one
    // output hop), the F32 input cast only where the runner actually pays it.
    let bf16_direct = activation_dtype == DType::BF16
        && device.is_metal()
        && matches!(
            quant_dtype,
            GgmlDType::Q8_0 | GgmlDType::Q4K | GgmlDType::Q6K
        );
    let result = (|| -> Result<(Duration, Duration)> {
        let prepare_started = Instant::now();
        let qweight = QTensor::quantize_onto(weight_cpu, quant_dtype, device)?;
        let linear = lmbrrr::quantized_linear::MixedLinear::from_qtensor(qweight)?;
        let input = input_cpu.to_device(device)?.to_dtype(activation_dtype)?;
        device.synchronize()?;
        let prepare_elapsed = prepare_started.elapsed();
        let elapsed =
            time_iterations(device, warmup, iterations, || Ok(linear.forward(&input)?))?;
        Ok((prepare_elapsed, elapsed))
    })();
    matmul_bench_row(
        shape,
        mode,
        "quantized",
        Some(format!("{quant_dtype:?}")),
        activation_dtype,
        if activation_dtype == DType::F32 {
            None
        } else if bf16_direct {
            Some("bf16_direct")
        } else {
            Some("to_f32")
        },
        result,
        iterations,
        input_cpu.dim(1).unwrap_or(1),
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
    tokens_per_iteration: usize,
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
            "tokens_per_second": tokens_per_second(tokens_per_iteration * iterations, elapsed),
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
        // Materialize a non-trivial buffer first; a bare zeros tensor can be
        // elided by the backend and reports absurd copy bandwidth.
        let x = Tensor::zeros(elements, dtype, &device)?.affine(1.0, 0.5)?;
        device.synchronize()?;
        let elapsed = time_iterations(&device, args.warmup, args.iterations, || {
            Ok(x.affine(1.000001, 0.0)?)
        })?;
        let bytes_moved = 2.0 * (elements * dtype.size_in_bytes()) as f64 * args.iterations as f64;
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
        let elapsed = time_iterations(&device, args.warmup, args.iterations, || {
            Ok(linear.forward(&x)?)
        })?;
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
        let _ = y.to_vec1::<f32>()?;
    }
    device.synchronize()?;
    let started = Instant::now();
    for _ in 0..args.iterations {
        let mut y = tiny.clone();
        for _ in 0..args.dispatch_chain {
            y = y.affine(1.000001, 0.0)?;
        }
        let _ = y.to_vec1::<f32>()?;
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
        args.model.quantize_lm_head,
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
                // LMBRRR_VT_PROFILE=1 attaches the component profiler to the
                // final iteration so the chunk cost decomposes (used to chase
                // the l=1 -> l=2 doubling). Host-side attribution: encode +
                // queue backpressure per component, not GPU time.
                let profile_this = std::env::var("LMBRRR_VT_PROFILE").is_ok_and(|v| v == "1")
                    && iteration + 1 == args.warmup + args.iterations;
                let vt_profiler = profile_this.then(Qwen35Profiler::new);
                if let Some(p) = &vt_profiler {
                    model.set_text_profiler(Some(p.clone()));
                }
                let started = Instant::now();
                let logits = model.forward_all_logits(
                    &chunk_input,
                    None::<&ProcessedImages>,
                    &args.model.downsample_mode,
                    prompt_tokens.len(),
                )?;
                device.synchronize()?;
                let chunk_elapsed = started.elapsed();
                if let Some(p) = &vt_profiler {
                    model.set_text_profiler(None);
                    let events = p.events();
                    eprintln!(
                        "vt-profile gamma={gamma} ctx={}: {}",
                        prompt_tokens.len(),
                        serde_json::to_string(&aggregate_profile_events(&events))?
                    );
                }
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
        "scheduler_contract": "T_verify(gamma) per context bucket = median_verify_seconds; T_round(gamma) = T_fixed + T_draft + T_verify(gamma). T_fixed is the per-round host residual measured in-loop (dspark report round_residual_ms.drafter_rounds.median_ms), carried by the cost-model artifact's fixed_round_ms.",
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


/// Static-batched N-stream greedy decode. Same prompt per stream (static
/// batching), batch dimension through the whole text path. The fused
/// DeltaNet decode kernel currently gates to b == 1, so batched steps take
/// the tensor path — dispatch counts amortize across streams, which is the
/// aggregate lane's core economics; kernel batching is the follow-up.
fn multi_bench(args: MultiBenchArgs) -> Result<()> {
    if args.streams.is_empty() || args.streams.contains(&0) {
        anyhow::bail!("--streams must contain positive stream counts");
    }
    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let prompt_text = chat_prompt(&args.prompt, 0, false);
    let prompt_tokens = tokenize_prompt(&tokenizer, prompt_text)?;
    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
        args.model.quantize_lm_head,
    )?;
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());
    let len = prompt_tokens.len();

    // Single-stream reference for the equivalence check (advisory: batched
    // numerics can tie-flip).
    model.clear_cache();
    let mut rows = Vec::new();
    let reference = generate_tokens(
        &mut model,
        &device,
        &greedy_generation_args(args.max_new_tokens, false),
        &prompt_tokens,
        None::<&ProcessedImages>,
        &args.model.downsample_mode,
        &eos_ids,
        |_, _, _, _| Ok(()),
    )?;

    for &n in &args.streams {
    model.clear_cache();
    let mut batched: Vec<u32> = Vec::with_capacity(n * len);
    for _ in 0..n {
        batched.extend_from_slice(&prompt_tokens);
    }
    let input = Tensor::from_slice(&batched, (n, len), &device)?;
    let prefill_start = Instant::now();
    let mut logits = model.forward(&input, None::<&ProcessedImages>, &args.model.downsample_mode, 0)?;
    device.synchronize()?;
    let prefill_elapsed = prefill_start.elapsed();

    let mut streams: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut finished = vec![false; n];
    let mut position = len;
    let decode_start = Instant::now();
    let mut steps = 0usize;
    for _ in 0..args.max_new_tokens {
        // logits [n, vocab]
        let next = logits
            .to_dtype(DType::F32)?
            .argmax(D::Minus1)?
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        let mut all_done = true;
        for (i, tok) in next.iter().enumerate() {
            if !finished[i] {
                if eos_ids.contains(tok) {
                    finished[i] = true;
                } else {
                    streams[i].push(*tok);
                    all_done = false;
                }
            }
        }
        steps += 1;
        if all_done {
            break;
        }
        // Finished streams keep decoding their last token (static batch);
        // their outputs are ignored above.
        let feed: Vec<u32> = next
            .iter()
            .enumerate()
            .map(|(i, t)| if finished[i] { eos_ids[0] } else { *t })
            .collect();
        let step_input = Tensor::from_slice(&feed, (n, 1), &device)?;
        logits = model.forward(&step_input, None::<&ProcessedImages>, &args.model.downsample_mode, position)?;
        position += 1;
    }
    let decode_elapsed = decode_start.elapsed();
    let total_tokens: usize = streams.iter().map(|s| s.len()).sum();
    let aggregate_tps = total_tokens as f64 / decode_elapsed.as_secs_f64();

    let equiv = reference
        .generated_token_ids
        .iter()
        .zip(streams[0].iter())
        .take_while(|(a, b)| a == b)
        .count();

    rows.push(serde_json::json!({
        "streams": n,
        "decode_steps": steps,
        "total_generated_tokens": total_tokens,
        "prefill_seconds": secs(prefill_elapsed),
        "decode_seconds": secs(decode_elapsed),
        "aggregate_tokens_per_second": aggregate_tps,
        "per_stream_tokens_per_second": aggregate_tps / n as f64,
        "single_stream_equivalence_prefix": equiv,
        "stream0_text_head": decode_tokens(&tokenizer, &streams[0][..streams[0].len().min(32)])?,
    }));
    eprintln!(
        "streams={n}: aggregate {aggregate_tps:.0} tok/s ({:.1}/stream)",
        aggregate_tps / n as f64
    );
    }

    let report = serde_json::json!({
        "kind": "lmbrrr_multi_bench",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "prompt_tokens": len,
        "max_new_tokens": args.max_new_tokens,
        "single_stream_reference_tokens": reference.generated_token_ids.len(),
        "rows": rows,
    });
    write_json_report(args.output.as_ref(), &report)
}

fn load_model_with_optional_quantization(
    bundle: &ArtifactBundle,
    dtype: DType,
    device: &Device,
    quantized_manifest: Option<&PathBuf>,
    quantize_lm_head: Option<DrafterQuantArg>,
) -> Result<(
    MiniCpmForConditionalGeneration,
    Duration,
    Option<QuantizedLoadStats>,
)> {
    let (mut model, load_elapsed, _) = load_model(bundle, dtype, device)?;
    if let Some(tier) = quantize_lm_head {
        model.quantize_lm_head(tier.ggml())?;
    }
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

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DrafterQuantArg {
    Q8_0,
    Q4k,
    Q6k,
}

impl DrafterQuantArg {
    fn ggml(self) -> candle::quantized::GgmlDType {
        match self {
            Self::Q8_0 => candle::quantized::GgmlDType::Q8_0,
            Self::Q4k => candle::quantized::GgmlDType::Q4K,
            Self::Q6k => candle::quantized::GgmlDType::Q6K,
        }
    }
}

/// `LMBRRR_LOOP_TIMING=1` re-enables the per-phase synchronize() calls in the
/// speculative round so draft/verify/rollback buckets measure GPU time. Off
/// (default) the round pays exactly two readback waits and the buckets
/// attribute encode+queue time only.
fn loop_timing() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("LMBRRR_LOOP_TIMING").is_ok_and(|v| v == "1"))
}

/// `LMBRRR_READVANCE_ROLLBACK=1` restores the legacy restore + re-advance
/// rollback (reference path for the state-selection mechanism).
fn readvance_rollback() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("LMBRRR_READVANCE_ROLLBACK").is_ok_and(|v| v == "1"))
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
                quantize_lm_head: None,
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

}
