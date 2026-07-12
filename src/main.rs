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
        Command::QuantSensitivity(args) => commands::quant::quant_sensitivity(args),
        Command::QuantConvert(args) => commands::quant::quant_convert(args),
        Command::QuantMatmulBench(args) => commands::quant::quant_matmul_bench(args),
        Command::QuantQuality(args) => commands::quant::quant_quality(args),
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
