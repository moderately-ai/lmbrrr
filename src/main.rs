#![recursion_limit = "256"]

use std::{
    collections::HashMap,
    fs::{self, OpenOptions},
    io::{BufWriter, IsTerminal, Stdout, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use candle::{DType, Device, Tensor, D};
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
    qwen35::{Qwen35ProfileEvent, Qwen35Profiler},
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

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DTypeArg {
    Auto,
    F32,
    F16,
    Bf16,
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

    let (mut model, load_elapsed) = load_model(&bundle, dtype, &device)?;
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

    let (mut model, load_elapsed) = load_model(&bundle, dtype, &device)?;
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
    let (mut model, load_elapsed) = load_model(&bundle, dtype, &device)?;

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
    let (mut model, load_elapsed) = load_model(&bundle, dtype, &device)?;
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
) -> Result<(MiniCpmForConditionalGeneration, Duration)> {
    let load_start = Instant::now();
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&bundle.artifacts.weights, dtype, device)? };
    let model = MiniCpmForConditionalGeneration::new(&bundle.config, vb)?;
    Ok((model, load_start.elapsed()))
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

fn argmax_token(logits: &Tensor, device: &Device) -> Result<(u32, Duration)> {
    device.synchronize()?;
    let started = Instant::now();
    let token = logits.squeeze(0)?.argmax(D::Minus1)?.to_scalar::<u32>()?;
    device.synchronize()?;
    Ok((token, started.elapsed()))
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
