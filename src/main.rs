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

    /// Wrap exactly one decode step (this index) in a Metal .gputrace
    /// capture. Disables hidden-state capture (no recorder readbacks in the
    /// trace) and requires METAL_CAPTURE_ENABLED=1. The trace lands next to
    /// --output (or the cwd) as decode-step-<N>.gputrace.
    #[arg(long)]
    gpu_capture_step: Option<usize>,

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

    /// One-round-lag scheduling with on-device chunk assembly: draft ids
    /// stay on device, the verify chunk is a device-side cat, and the
    /// proposal readback rides the verify drain — 2 pipeline drains per
    /// drafted round become 1 (~1-2ms OS wait each). Width for round r is
    /// scheduled from round r-1's confidence vector (offline EV probe: 68.8%
    /// width agreement, ~2% mean regret); the first drafted round and --tree
    /// runs stay synchronous. Requires --schedule.
    #[arg(long)]
    lag_schedule: bool,

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

    /// Consecutive zero-width (scheduler-rejected) rounds before the
    /// skip-hysteresis stops paying for drafts.
    #[arg(long, default_value_t = 3)]
    skip_draft_after: usize,

    /// While parked, probe a drafted round every Nth round.
    #[arg(long, default_value_t = 8)]
    probe_every: usize,

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
struct TopLogit {
    token_id: u32,
    token: String,
    logit: f32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => commands::run_bench::run(args),
        Command::Bench(args) => commands::run_bench::bench(args),
        Command::Logits(args) => commands::diag::logits(args),
        Command::Profile(args) => commands::diag::profile_decode(args),
        Command::SpecVerify(args) => commands::verify::spec_verify(args),
        Command::Trace(args) => commands::diag::trace_hidden_states(args),
        Command::QuantSensitivity(args) => commands::quant::quant_sensitivity(args),
        Command::QuantConvert(args) => commands::quant::quant_convert(args),
        Command::QuantMatmulBench(args) => commands::quant::quant_matmul_bench(args),
        Command::QuantQuality(args) => commands::quant::quant_quality(args),
        Command::Roofline(args) => commands::diag::roofline(args),
        Command::VerifyTable(args) => commands::diag::verify_table(args),
        Command::DsparkRun(args) => commands::dspark::dspark_run(args),
        Command::DsparkDrafterParity(args) => commands::verify::dspark_drafter_parity(args),
        Command::MultiBench(args) => commands::run_bench::multi_bench(args),
        Command::TreeCheck(args) => commands::verify::tree_check(args),
        Command::VisionCheck(args) => commands::diag::vision_check(args),
        Command::FakequantExport(args) => commands::diag::fakequant_export(args),
    }
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

fn select_device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    match Device::new_metal(0) {
        Ok(device) => Ok(device),
        Err(err) => {
            eprintln!("Metal unavailable ({err}); falling back to CPU");
            Ok(Device::Cpu)
        }
    }
}
