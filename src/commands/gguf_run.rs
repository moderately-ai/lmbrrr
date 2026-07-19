//! Lean text-decode + throughput harness for a qwen35-hybrid GGUF (the ternary
//! Ternary-Bonsai-27B target). Baseline path: host argmax per token with a
//! per-token sync for honest timing — no fused-argmax / async-readback fast
//! paths yet (those land with the generic decode-loop unification). The number
//! this reports is the floor to iterate up from, measured on the M3 referee.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use candle::{DType, Device, Tensor, D};
use clap::{Args, Parser, Subcommand};

use lmbrrr::gguf::GgufFile;
use lmbrrr::model_ctx::ModelCtx;
use lmbrrr::qwen35::{CausalTextModel, Qwen35CausalLM, Qwen35Profiler};
use tokenizers::Tokenizer;

/// Text-decode + throughput harness for the ternary Ternary-Bonsai-27B GGUF.
#[derive(Parser, Debug)]
pub(crate) struct GgufArgs {
    #[command(subcommand)]
    cmd: GgufCmd,
}

#[derive(Subcommand, Debug)]
enum GgufCmd {
    /// Greedy text decode with a per-token sync (the honest throughput floor).
    Decode(DecodeArgs),
    /// DSpark speculative decode with a Bonsai drafter GGUF.
    Spec(SpecArgs),
    /// Attach the per-op profiler and dump the decode-step breakdown by
    /// component (deltanet_recurrent_rule / mlp / attention / norms / ...).
    Profile(ProfileArgs),
    /// Teacher-force a token continuation under the target and report its
    /// mean log-prob / perplexity — the margin-acceptance quality gate
    /// (compare a spec run's committed ids against the greedy ids).
    Score(ScoreArgs),
    /// Requantize a GGUF's 2D float tensors to a new ggml dtype (arch-agnostic
    /// — works on the dspark drafter, which stock llama.cpp cannot quantize).
    /// 1D and non-divisible tensors pass through unchanged; metadata is copied.
    Requant(RequantArgs),
    /// Diagnostic (ticket mm2d-fullk-pow2-requant-verify STEP 1): read the Q2_0
    /// per-128 block scales `d` and report how far they sit from powers of two
    /// (ue8m0), plus the per-32-pow2 vs per-32-arbitrary code-refit reconstruction
    /// error. Decides whether the fold-free fullk requant is viable FROM the
    /// deployed Q2_0 weights, or must come from the original bf16 (Modal-side).
    Pow2Scales(Pow2ScalesArgs),
    /// Build the mm2d (tensor-op verify) plane artifact for a Q2_0 GGUF: repack
    /// every eligible weight into the planar layout and write the plane files
    /// next to the model. One-time; runs point at the artifact via
    /// LMBRRR_MM2D_CACHE_DIR (no load-time repacking).
    Repack(RepackArgs),
    /// Micro-bench the ternary Q2_0 verify/decode GEMV kernels on the ffn shape
    /// (random weights; no model load) — isolates kernel bandwidth.
    BenchGemv,
    /// mc-vs-mm2d on the model's REAL verify shapes at m=5 (interleaved):
    /// attributes the in-loop planar-only verify shortfall per weight class.
    BenchShapes,
    /// Loop ONE quantized matmul kernel in isolation so a gpucapture/gpudebug
    /// capture is dominated by it (counters are timeline-aggregate). No model.
    ProfileKernel(ProfileKernelArgs),
    /// Loop the DeltaNet prefill kernels in isolation (streaming vs chunk-loop)
    /// on synthetic Bonsai-shaped inputs — for capture/counter attribution of
    /// why streaming under/over-performs the host-looped chunk. No model.
    BenchDeltanet(BenchDeltanetArgs),
}

#[derive(Args, Debug)]
struct BenchDeltanetArgs {
    /// Which kernel to loop: "stream" | "chunk". Loop it in isolation so a
    /// gpucapture is dominated by it.
    #[arg(long, default_value = "stream")]
    which: String,
    /// Sequence length (prefill span).
    #[arg(long, default_value_t = 232)]
    l: usize,
    /// Iterations of the whole-sequence work (each = 1 stream dispatch or
    /// ceil(l/12) chunk dispatches). Large so the kernel dominates a capture.
    #[arg(long, default_value_t = 200)]
    iters: usize,
}

#[derive(Args, Debug)]
struct ProfileKernelArgs {
    /// Kernel to loop: mv | mm2d-k32 | mm2d-k128 | q4k-mm2d.
    #[arg(long, default_value = "mm2d-k128")]
    which: String,
    /// Dispatches to enqueue (make it large so the kernel dominates the capture).
    #[arg(long, default_value_t = 3000)]
    iters: usize,
    /// Activation rows (verify width). mv ignores this (always m=1).
    #[arg(long, default_value_t = 8)]
    m: usize,
}

/// Model-loading args shared by the decode / spec / profile subcommands.
#[derive(Args, Debug)]
struct ModelArgs {
    /// Path to the qwen35-hybrid GGUF (e.g. Ternary-Bonsai-27B-Q2_0.gguf).
    #[arg(long)]
    gguf: PathBuf,

    #[arg(long, default_value = "Explain quantum computing in simple terms.")]
    prompt: String,

    /// Feed the prompt verbatim instead of wrapping it in the ChatML template.
    #[arg(long)]
    raw: bool,
}

#[derive(Args, Debug)]
struct DecodeArgs {
    #[command(flatten)]
    model: ModelArgs,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    /// Run one untimed warmup forward (prefill + a decode step) before the
    /// timed prefill, so prefill_seconds/TTFT reflect warm compute instead of
    /// the process's first-forward shader-compile cost. Reports both the cold
    /// (compile-inclusive) and warm prefill so the compile share is explicit.
    #[arg(long)]
    warmup: bool,
}

#[derive(Args, Debug)]
struct SpecArgs {
    #[command(flatten)]
    model: ModelArgs,

    /// Drafter GGUF. Recommended: Ternary-Bonsai-27B-dspark-Q8_0.gguf — Q8_0
    /// is ~25% faster propose than Q4_1 with identical acceptance (lossless;
    /// build via `gguf requant --dtype q8_0`). The draft width is the drafter's
    /// block_size (read from GGUF metadata).
    #[arg(long)]
    drafter: PathBuf,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    /// Typical acceptance: a draft survives while its target logit is within
    /// this margin of the top logit (port of the MiniCPM loop's flag).
    /// Committed tokens remain the drafts, so output may legitimately differ
    /// from greedy. DEFAULT (unset) = margin 1.0, the quality-FREE speed point
    /// (PPL 1.124 vs greedy 1.114 — the PPL gate passed, ~+15% tok/s). Pass
    /// `0` (or `--exact`) for the lossless byte-match path; `3.0` (or `--fast`)
    /// for the fast operating point (~19 tok/s, PPL +8-16% — a real tradeoff).
    #[arg(long)]
    accept_margin: Option<f32>,

    /// Lossless byte-exact acceptance (equivalent to `--accept-margin 0`):
    /// commit only exact-argmax matches, output byte-identical to greedy.
    /// Overrides the default margin. Wins over `--fast`.
    #[arg(long)]
    exact: bool,

    /// Fast operating point: sugar for `--accept-margin 3.0` (~19 tok/s, PPL
    /// +8-16% vs greedy — coherent but not quality-preserving). Ignored if an
    /// explicit `--accept-margin` or `--exact` is given.
    #[arg(long)]
    fast: bool,

    /// Peakedness gate for --accept-margin: apply the margin only on rows
    /// whose top LOGPROB >= -this value; flat rows fall back to exact argmax
    /// match. Motivated by the per-class PPL gate: margin drift concentrates
    /// on flat next-token distributions (factual +21.4% PPL vs +4-5% on
    /// peaked classes), where a logit margin does not bound logprob.
    #[arg(long, requires = "accept_margin")]
    margin_peak: Option<f32>,

    /// Prompt-lookup drafting. Default OFF: measured a net LOSS against this
    /// drafter on both prose (18.3 -> 15.2 tok/s, 0 copy tokens accepted) and
    /// code (20.0 -> 16.1 — copy rounds preempt stronger drafter rounds),
    /// reproducing the MiniCPM campaign's ungated-PLD lesson.
    #[arg(long)]
    pld: bool,

    /// Two-branch tree verification: when the drafter's runner-up root token
    /// is live, verify [anchor, a_1..a_3, b_1..b_3] in one flattened 7-row
    /// chunk (fits the flat m<=8 tensor tile) and commit the longer branch.
    /// Exact-argmax acceptance only.
    #[arg(long)]
    tree: bool,

    /// Skip verify when mean drafter confidence is below this threshold: feed
    /// the pending anchor as a plain 1-token greedy step instead (~69 ms vs
    /// ~218 ms verify). Live A/B 2026-07-19: thr≥1.0 over-skips (−25% tok/s)
    /// because offline EV assumed a fixed trajectory; use only very low thr
    /// (≤0) or prefer `--skip-after-reject`. Default OFF.
    #[arg(long)]
    skip_low_conf: Option<f32>,

    /// After a fully-rejected draft round (accepted==0), take the next step as
    /// plain greedy (no propose/verify). Causal, no conf calibration needed.
    /// Offline EV modest; compounds with low-conf skip. Default OFF.
    #[arg(long)]
    skip_after_reject: bool,

    /// Disable the default planar-mm2d verify path and run the packed GEMV
    /// path instead. `gguf spec` defaults to planar mm2d (the ~19 tok/s
    /// operating point; builds + caches planes to ~/.cache/lmbrrr/mm2d on the
    /// first run). Use this on a pre-Metal-4 OS or to A/B the packed path.
    /// `LMBRRR_MM2D` in the environment overrides this either way.
    #[arg(long)]
    no_mm2d: bool,
}

#[derive(Args, Debug)]
struct ProfileArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Chunk width to profile: 1 = plain decode steps; 5 = verify-shaped
    /// chunks (block_size 4 + anchor) — attributes the per-round verify cost
    /// by component under whatever route env is set (LMBRRR_MM2D etc.).
    #[arg(long, default_value_t = 1)]
    verify_width: usize,
    /// Replicate the spec loop's tap-layer device capture during each chunk
    /// forward (e.g. "1,16,31,46,61" = the Bonsai drafter taps). The in-loop
    /// verify always captures; the profiler default doesn't — this flag makes
    /// the two measure the same work.
    #[arg(long)]
    capture_taps: Option<String>,
}

#[derive(Args, Debug)]
struct ScoreArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Comma-separated committed token ids to score as the continuation of
    /// the (templated) prompt.
    #[arg(long)]
    ids: String,
}

/// Teacher-forced continuation score: one forward over prompt+ids, then the
/// mean log-prob of each id given its prefix (and its exp, perplexity). The
/// greedy ids' score is the reference; a margin-acceptance run passes the
/// quality gate when its score is within a small delta of greedy's.
fn score(model: &mut Qwen35CausalLM, device: &Device, prompt_ids: &[u32], gen: &[u32]) -> Result<()> {
    anyhow::ensure!(!gen.is_empty(), "no ids to score");
    let full: Vec<u32> = prompt_ids.iter().chain(gen.iter()).copied().collect();
    model.clear_cache();
    let input = Tensor::from_slice(&full, (1, full.len()), device)?;
    let logits = model.forward_all_logits(&input, 0)?;
    // Positions prompt_len-1 .. full_len-2 predict exactly the gen tokens.
    let rows = logits
        .narrow(1, prompt_ids.len() - 1, gen.len())?
        .to_dtype(DType::F32)?;
    let logsm = candle_nn::ops::log_softmax(&rows, D::Minus1)?;
    let idx = Tensor::from_slice(gen, (1, gen.len(), 1), device)?;
    let lps = logsm
        .gather(&idx, D::Minus1)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mean_lp: f64 = lps.iter().map(|&x| x as f64).sum::<f64>() / lps.len() as f64;
    let min_lp = lps.iter().cloned().fold(f32::INFINITY, f32::min);
    // Worst positions, for attributing WHERE a margin run diverges from the
    // reference (round boundaries? the prefill anchor? rollback drift?).
    let mut ranked: Vec<(usize, u32, f32)> = lps
        .iter()
        .enumerate()
        .map(|(i, &lp)| (i, gen[i], lp))
        .collect();
    ranked.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    let worst: Vec<String> = ranked
        .iter()
        .take(5)
        .map(|(i, id, lp)| format!("pos {i} id {id} lp {lp:.2}"))
        .collect();
    println!(
        "{}",
        serde_json::json!({
            "tokens": gen.len(),
            "mean_logprob": mean_lp,
            "ppl": (-mean_lp).exp(),
            "min_logprob": min_lp,
            "worst": worst,
        })
    );
    Ok(())
}

#[derive(Args, Debug)]
struct RequantArgs {
    /// Source GGUF.
    #[arg(long)]
    gguf: PathBuf,
    /// Output GGUF path.
    #[arg(long)]
    out: PathBuf,
    /// Target ggml dtype for 2D float tensors: q8_0 | q4_1 | q4k | q6k.
    #[arg(long, default_value = "q8_0")]
    dtype: String,
}

#[derive(Args, Debug)]
struct Pow2ScalesArgs {
    /// Q2_0 GGUF to analyze.
    #[arg(long)]
    gguf: PathBuf,
    /// Analyze only the first N Q2_0 tensors (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

/// STEP 1 diagnostic for the fullk/pow2-requant lever. For each Q2_0 2D tensor:
/// (a) read the per-128 block scales `d` and measure their log2-distance to the
/// nearest integer exponent (0 = already a power of two, 0.5 = worst case), and
/// (b) requant the DEQUANTIZED weights to per-32 blocks under an arbitrary scale
/// vs a power-of-two scale (2-bit codes {-1,0,1,2} refit to each), reporting the
/// relative reconstruction error of each. If (a) shows the d's are far from pow2
/// and (b) shows per-32-pow2 >> per-32-arbitrary, the fullk requant CANNOT be
/// done from the deployed Q2_0 weights (per-32 granularity does not help when the
/// input is already per-128-quantized) and must requant the ORIGINAL bf16.
fn pow2_scales(args: &Pow2ScalesArgs) -> Result<()> {
    use candle::quantized::{gguf_file, GgmlDType};
    let mut file = std::fs::File::open(&args.gguf)?;
    let content = gguf_file::Content::read(&mut file)?;
    let device = Device::Cpu;
    let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    names.sort();

    // (a) pow2-distance of the per-128 d scales, over sampled Q2_0 tensors.
    let mut log2dist: Vec<f32> = Vec::new();
    // (b) per-32 requant reconstruction error, arbitrary vs power-of-two.
    let (mut se_arb, mut se_p2, mut wnorm) = (0f64, 0f64, 0f64);
    let mut analyzed = 0usize;
    for name in &names {
        let info = &content.tensor_infos[name];
        if !matches!(info.ggml_dtype, GgmlDType::Q2_0) || info.shape.dims().len() != 2 {
            continue;
        }
        if args.limit != 0 && analyzed >= args.limit {
            break;
        }
        let qt = content.tensor(&mut file, name, &device)?;
        // (a) raw block scales: Q2_0 block = half d (2 bytes) + 32 code bytes.
        let bytes = qt.data()?;
        for blk in bytes.chunks_exact(34) {
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            if d > 0.0 && d.is_finite() {
                let l = d.log2();
                log2dist.push((l - l.round()).abs());
            }
        }
        // (b) per-32 requant error on the dequantized weights.
        let w = qt.dequantize(&device)?.flatten_all()?.to_vec1::<f32>()?;
        for blk in w.chunks(32) {
            let amax = blk.iter().fold(0f32, |a, &x| a.max(x.abs()));
            for &x in blk {
                wnorm += (x as f64) * (x as f64);
            }
            if amax <= 0.0 {
                continue;
            }
            // arbitrary scale: grid search d in (0, amax], codes {-1,0,1,2}.
            let mut best_arb = f64::INFINITY;
            for step in 1..=24 {
                let d = amax * (step as f32) / 24.0;
                let mut se = 0f64;
                for &x in blk {
                    let c = (x / d).round().clamp(-1.0, 2.0);
                    let e = (x - c * d) as f64;
                    se += e * e;
                }
                best_arb = best_arb.min(se);
            }
            se_arb += best_arb;
            // power-of-two scale: exponents spanning the block magnitude.
            let kmax = amax.log2().ceil() as i32;
            let mut best_p2 = f64::INFINITY;
            for kk in (kmax - 4)..=(kmax + 1) {
                let d = 2f32.powi(kk);
                let mut se = 0f64;
                for &x in blk {
                    let c = (x / d).round().clamp(-1.0, 2.0);
                    let e = (x - c * d) as f64;
                    se += e * e;
                }
                best_p2 = best_p2.min(se);
            }
            se_p2 += best_p2;
        }
        analyzed += 1;
    }

    log2dist.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = log2dist.len().max(1);
    let median = log2dist[n / 2];
    let mean = log2dist.iter().sum::<f32>() / n as f32;
    let near = log2dist.iter().filter(|&&x| x < 0.083).count(); // within ~6% of a pow2
    let wnorm = wnorm.sqrt().max(1e-12);
    println!(
        "pow2-scales: {analyzed} Q2_0 tensors, {n} per-128 scales\n\
         (a) |log2(d) - round| distance to nearest power-of-two: median {median:.3}, mean {mean:.3} (0=pow2, 0.5=worst); within 6% of pow2: {:.1}%\n\
         (b) per-32 refit reconstruction rel_err vs deployed Q2_0 weights: arbitrary {:.4}, power-of-two {:.4} (ratio {:.2}x)",
        100.0 * near as f64 / n as f64,
        se_arb.sqrt() / wnorm,
        se_p2.sqrt() / wnorm,
        (se_p2 / se_arb.max(1e-30)).sqrt()
    );
    Ok(())
}

/// CPU-only requant: read every tensor, quantize eligible 2D float tensors to
/// the target dtype, pass everything else (f32 norms, 1D, already-quantized)
/// through, and write a fresh GGUF with the source metadata verbatim.
fn requant(args: &RequantArgs) -> Result<()> {
    use candle::quantized::{gguf_file, GgmlDType, QTensor};
    let dtype = match args.dtype.as_str() {
        "q8_0" => GgmlDType::Q8_0,
        "q4_1" => GgmlDType::Q4_1,
        "q4k" => GgmlDType::Q4K,
        "q6k" => GgmlDType::Q6K,
        other => anyhow::bail!("unsupported requant dtype {other}"),
    };
    let started = Instant::now();
    let mut file = std::fs::File::open(&args.gguf)?;
    let content = gguf_file::Content::read(&mut file)?;
    let device = Device::Cpu;
    let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    names.sort();
    let mut tensors: Vec<(String, QTensor)> = Vec::with_capacity(names.len());
    let (mut converted, mut passed) = (0usize, 0usize);
    for name in &names {
        let qt = content.tensor(&mut file, name, &device)?;
        let dims = qt.shape().dims().to_vec();
        let eligible = matches!(qt.dtype(), GgmlDType::BF16 | GgmlDType::F16)
            && dims.len() == 2
            && dims[1] % dtype.block_size() == 0;
        if eligible {
            let t = qt.dequantize(&device)?;
            tensors.push((name.clone(), QTensor::quantize(&t, dtype)?));
            converted += 1;
        } else {
            tensors.push((name.clone(), qt));
            passed += 1;
        }
    }
    let metadata: Vec<(&str, &gguf_file::Value)> = content
        .metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    let tensor_refs: Vec<(&str, &QTensor)> =
        tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args.out)?);
    gguf_file::write(&mut out, &metadata, &tensor_refs)?;
    println!(
        "requantized {converted} tensors to {dtype:?} ({passed} passed through) in {:.1}s -> {}",
        started.elapsed().as_secs_f64(),
        args.out.display()
    );
    Ok(())
}

#[derive(Args, Debug)]
struct RepackArgs {
    /// Path to the qwen35-hybrid Q2_0 GGUF to repack.
    #[arg(long)]
    gguf: PathBuf,
    /// Plane-artifact directory (default: `mm2d-planes` next to the GGUF).
    #[arg(long)]
    out: Option<PathBuf>,
}

/// Build the mm2d plane artifact: construct the model exactly as a run would
/// (same fused weights, so the sha256 cache keys match byte-exactly) with the
/// plane build enabled and pointed at the artifact dir, then report what was
/// written. Runs consume it via LMBRRR_MM2D_CACHE_DIR=<dir> with zero repack
/// work at load.
fn repack(device: &Device, args: &RepackArgs) -> Result<()> {
    let out = args.out.clone().unwrap_or_else(|| {
        args.gguf
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("mm2d-planes")
    });
    std::fs::create_dir_all(&out)?;
    let started = Instant::now();
    let gguf = GgufFile::open(&args.gguf)?;
    let cfg = gguf.config()?;
    let ctx = ModelCtx {
        mm2d: std::sync::Arc::new(lmbrrr::mm2d::Mm2dConfig {
            enabled: true,
            plane_cache_dir: Some(out.clone()),
            ..Default::default()
        }),
        routes: Default::default(),
    };
    let model = {
        let src = gguf.source(&ctx, DType::BF16, device.clone());
        Qwen35CausalLM::new(&cfg, &src, &ctx)?
    };
    drop(model);
    let (mut files, mut bytes) = (0usize, 0u64);
    for entry in std::fs::read_dir(&out)? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|e| e == "bin") {
            files += 1;
            bytes += entry.metadata()?.len();
        }
    }
    println!(
        "repacked {files} plane files, {:.2} GB, in {:.1}s -> {}",
        bytes as f64 / 1e9,
        started.elapsed().as_secs_f64(),
        out.display()
    );
    // Kernel-ineligible weights stay on the GEMV route by design; report the
    // classes once instead of a per-weight warning wall.
    {
        use candle::quantized::GgmlDType;
        let mut skipped = std::collections::BTreeMap::<(usize, usize), usize>::new();
        for (_name, dims, dtype) in gguf.tensor_infos() {
            if matches!(dtype, GgmlDType::Q2_0)
                && dims.len() == 2
                && !lmbrrr::mm2d::mm2d_q2_k_supported(dims[1])
            {
                *skipped.entry((dims[0], dims[1])).or_default() += 1;
            }
        }
        for ((n, k), count) in &skipped {
            println!(
                "skipped {count} x [{n}, {k}]: k={k} unsupported by the mm2d kernel; these stay on the GEMV route"
            );
        }
    }
    Ok(())
}

/// mc vs mm2d at m=5 on every Q2_0 weight shape the verify actually runs.
/// Random weights; interleaved timing (DVFS-fair); correctness vs dense f32
/// where the dense reference fits.
fn bench_shapes(device: &Device) -> Result<()> {
    use candle::quantized::k_quants::BlockQ2_0;
    use candle::quantized::metal::q2_0_mm2d_planes;
    use candle::quantized::{GgmlDType, QTensor};
    use candle_metal_kernels::{
        call_quantized_matmul_mm2d_q2_0, call_quantized_matmul_mv_mc, Mm2dQ2Variant,
    };
    let mdev = match device {
        Device::Metal(d) => d.clone(),
        _ => anyhow::bail!("bench needs metal"),
    };
    let m = 5usize; // verify width (block_size 4 + anchor)
    // (label, n, k) — the per-layer verify matmuls + the head.
    let shapes: [(&str, usize, usize); 6] = [
        ("ba", 96, 5120),
        ("o/out", 5120, 5120),
        ("down(k17408)", 5120, 17408),
        ("qkvz", 16384, 5120),
        ("gate_up", 34816, 5120),
        ("head", 248320, 5120),
    ];
    for (label, n, k) in shapes {
        // Giant shapes (the head) are timing-only: synthesize random Q2_0
        // blocks directly — the dense f32 weight (5+ GB) OOMs the bench.
        let dense_ok = n * k <= 96_000_000;
        let qt = if dense_ok {
            let w = Tensor::randn(0f32, 1f32, (n, k), device)?;
            QTensor::quantize(&w, GgmlDType::Q2_0)?
        } else {
            use candle::quantized::ggml_file::qtensor_from_ggml;
            let nb = n * (k / 128);
            let mut raw = Vec::with_capacity(nb * 34);
            let d = [0x1Fu8, 0x21]; // f16 ~0.01, little-endian
            let mut s = 0x1234_5678u32;
            for _ in 0..nb {
                raw.extend_from_slice(&d);
                for _ in 0..32 {
                    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    raw.push((s >> 24) as u8);
                }
            }
            qtensor_from_ggml(GgmlDType::Q2_0, &raw, vec![n, k], device)?
        };
        let bytes = qt.storage_size_in_bytes();
        // Dense reference only where the f32 weight is affordable.
        let wdeq = if dense_ok {
            Some(qt.dequantize(device)?)
        } else {
            None
        };
        let data = qt.data()?;
        let wbuf = mdev
            .new_buffer_builder()
            .with_data(&data)
            .with_label("bs_w")
            .build()?;
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ2_0, data.len() / 34)
        };
        let planes = q2_0_mm2d_planes(blocks, n, k)?;
        let codes_buf = mdev
            .new_buffer_builder()
            .with_data(&planes.codes)
            .with_label("bs_codes")
            .build()?;
        let d_bytes = unsafe {
            std::slice::from_raw_parts(planes.d.as_ptr() as *const u8, planes.d.len() * 2)
        };
        let d_buf = mdev
            .new_buffer_builder()
            .with_data(d_bytes)
            .with_label("bs_d")
            .build()?;
        drop(data);

        let x = Tensor::randn(0f32, 1f32, (m, k), device)?.to_dtype(DType::BF16)?;
        let (xs, xl) = x.storage_and_layout();
        let xbuf = match &*xs {
            candle::Storage::Metal(ms) => ms.buffer().clone(),
            _ => anyhow::bail!("x not metal"),
        };
        let xoff = xl.start_offset() * 2;
        let dst_mc = mdev
            .new_buffer_builder()
            .with_size(m * n * 2)
            .with_label("bs_dst_mc")
            .build()?;
        let dst_mm = mdev
            .new_buffer_builder()
            .with_size(m * n * 2)
            .with_label("bs_dst_mm")
            .build()?;
        let variant = if k <= Mm2dQ2Variant::DEFAULT.max_k {
            Mm2dQ2Variant::DEFAULT
        } else {
            Mm2dQ2Variant::T64_K128_K17408
        };
        let disp_mc = || -> Result<()> {
            let enc = mdev.command_encoder()?;
            call_quantized_matmul_mv_mc(
                mdev.metal_device(),
                &enc,
                mdev.kernels(),
                GgmlDType::Q2_0.into(),
                true,
                true,
                (1, m, n, k),
                &xbuf,
                xoff,
                &wbuf,
                0,
                &dst_mc,
            )?;
            Ok(())
        };
        let disp_mm = || -> Result<()> {
            let enc = mdev.command_encoder()?;
            call_quantized_matmul_mm2d_q2_0(
                mdev.metal_device(),
                &enc,
                mdev.kernels(),
                (m, n, planes.n_pad, k),
                &xbuf,
                xoff,
                &codes_buf,
                &d_buf,
                0,
                &dst_mm,
                variant,
            )?;
            Ok(())
        };
        for _ in 0..8 {
            disp_mc()?;
            disp_mm()?;
        }
        device.synchronize()?;
        // Interleaved timing (DVFS-fair): alternate mc/mm2d batches.
        let rounds = 30usize;
        let batch = 8usize;
        let (mut t_mc, mut t_mm) = (0.0f64, 0.0f64);
        for _ in 0..rounds {
            device.synchronize()?;
            let t = Instant::now();
            for _ in 0..batch {
                disp_mc()?;
            }
            device.synchronize()?;
            t_mc += t.elapsed().as_secs_f64();
            let t = Instant::now();
            for _ in 0..batch {
                disp_mm()?;
            }
            device.synchronize()?;
            t_mm += t.elapsed().as_secs_f64();
        }
        let per_mc = t_mc / (rounds * batch) as f64;
        let per_mm = t_mm / (rounds * batch) as f64;
        // Correctness against the dense reference (skipped for the head).
        let rels = if let Some(wdeq) = &wdeq {
            let refr = x.to_dtype(DType::F32)?.matmul(&wdeq.t()?)?;
            let denom = refr.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-6);
            let rel_of = |buf: &std::sync::Arc<candle_metal_kernels::metal::Buffer>| -> Result<f32> {
                let out = candle::MetalStorage::new(buf.clone(), mdev.clone(), m * n, DType::BF16);
                let got = Tensor::from_storage(
                    candle::Storage::Metal(out),
                    (m, n),
                    candle::op::BackpropOp::none(),
                    false,
                )
                .to_dtype(DType::F32)?;
                Ok(got.sub(&refr)?.abs()?.mean_all()?.to_scalar::<f32>()? / denom)
            };
            disp_mc()?;
            disp_mm()?;
            device.synchronize()?;
            Some((rel_of(&dst_mc)?, rel_of(&dst_mm)?))
        } else {
            None
        };
        let rel_str = match rels {
            Some((a, b)) => format!("rel mc {a:.4} mm2d {b:.4}"),
            None => "rel skipped".to_string(),
        };
        println!(
            "{label:>13} [{n:>6}x{k:>5}] m={m}: mc {:>7.3} ms ({:>5.1} GB/s) | mm2d {:>7.3} ms ({:>5.1} GB/s) | {}",
            1000.0 * per_mc,
            bytes as f64 / per_mc / 1e9,
            1000.0 * per_mm,
            bytes as f64 / per_mm / 1e9,
            rel_str
        );
    }
    Ok(())
}

fn bench_gemv(device: &Device) -> Result<()> {
    use candle::quantized::{GgmlDType, QTensor};
    use lmbrrr::quantized_linear::MixedLinear;
    let ctx = ModelCtx::default();
    // ffn_up shape (the bulk of the per-layer weight traffic).
    for (n, k) in [(17408usize, 5120usize), (5120, 5120)] {
        let w = Tensor::randn(0f32, 1f32, (n, k), device)?;
        let x = Tensor::randn(0f32, 1f32, (1, 1, k), device)?.to_dtype(DType::BF16)?;
        for dt in [GgmlDType::Q2_0, GgmlDType::Q4K] {
            if k % dt.block_size() != 0 {
                continue;
            }
            let qt = QTensor::quantize(&w, dt)?;
            let bytes = qt.storage_size_in_bytes();
            let lin = MixedLinear::from_qtensor(qt, ctx.mm2d.clone())?;
            for _ in 0..8 {
                let _ = lin.forward(&x)?;
            }
            device.synchronize()?;
            let iters = 200;
            let t = Instant::now();
            for _ in 0..iters {
                let _ = lin.forward(&x)?;
            }
            device.synchronize()?;
            let s = t.elapsed().as_secs_f64();
            let gbps = (bytes as f64 * iters as f64) / s / 1e9;
            println!(
                "{dt:?} GEMV {n}x{k}: {:.3} ms/call, {:.1} GB/s ({} MB)",
                1000.0 * s / iters as f64,
                gbps,
                bytes / 1_000_000
            );
        }
    }

    // Verify-width path (DSpark spec verify): Q2_0 at m in {1,2,4,8}. The mc
    // kernel shares the weight read across the draft chunk, so m=8 should cost
    // ~1x the weight bandwidth (near m=1 ms/call) instead of ~8x.
    let (n, k) = (17408usize, 5120usize);
    let w = Tensor::randn(0f32, 1f32, (n, k), device)?;
    let qt = QTensor::quantize(&w, GgmlDType::Q2_0)?;
    let bytes = qt.storage_size_in_bytes();
    let wdeq = qt.dequantize(device)?; // (n, k) f32 reference weight
    let lin = MixedLinear::from_qtensor(qt, ctx.mm2d.clone())?;
    for m in [1usize, 2, 4, 8] {
        let x = Tensor::randn(0f32, 1f32, (1, m, k), device)?.to_dtype(DType::BF16)?;
        for _ in 0..8 {
            let _ = lin.forward(&x)?;
        }
        device.synchronize()?;
        let iters = 200;
        let t = Instant::now();
        for _ in 0..iters {
            let _ = lin.forward(&x)?;
        }
        device.synchronize()?;
        let s = t.elapsed().as_secs_f64();
        // correctness: mc output vs dense f32 reference (ternary noise bound).
        let got = lin.forward(&x)?.to_dtype(DType::F32)?.reshape((m, n))?;
        let refr = x.to_dtype(DType::F32)?.reshape((m, k))?.matmul(&wdeq.t()?)?;
        let denom = refr.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-6);
        let rel = got.sub(&refr)?.abs()?.mean_all()?.to_scalar::<f32>()? / denom;
        println!(
            "Q2_0 VERIFY {n}x{k} m={m}: {:.3} ms/call ({:.1} GB/s eff), rel_err {:.4}",
            1000.0 * s / iters as f64,
            (bytes as f64 * iters as f64) / s / 1e9,
            rel
        );
    }

    // Transposed-activation verify GEMV (mct): src1 = activation transposed to
    // [K][M] so the NC column reads are contiguous — the fix for the mc's L1
    // thrash (measured l1_eviction 1.42, occupancy capped at 41%).
    {
        use candle_metal_kernels::call_quantized_matmul_mv_q2_0_mct;
        let qtm = QTensor::quantize(&w, GgmlDType::Q2_0)?;
        let data = qtm.data()?;
        let mdev = match device {
            Device::Metal(d) => d.clone(),
            _ => anyhow::bail!("mct bench needs metal"),
        };
        let wbuf = mdev
            .new_buffer_builder()
            .with_data(&data)
            .with_label("mct_w")
            .build()?;
        for m in [1usize, 2, 4, 8] {
            let x = Tensor::randn(0f32, 1f32, (m, k), device)?.to_dtype(DType::BF16)?;
            let refr = x.to_dtype(DType::F32)?.matmul(&wdeq.t()?)?;
            let xt = x.t()?.contiguous()?; // [k][m]
            let (xs, xl) = xt.storage_and_layout();
            let xtbuf = match &*xs {
                candle::Storage::Metal(ms) => ms.buffer().clone(),
                _ => anyhow::bail!("xt not metal"),
            };
            let xoff = xl.start_offset() * 2;
            let dst = mdev
                .new_buffer_builder()
                .with_size(m * n * 4)
                .with_label("mct_dst")
                .build()?;
            let dispatch = || -> Result<()> {
                let enc = mdev.command_encoder()?;
                call_quantized_matmul_mv_q2_0_mct(
                    mdev.metal_device(),
                    &enc,
                    mdev.kernels(),
                    (m, n, k),
                    &wbuf,
                    &xtbuf,
                    xoff,
                    &dst,
                )?;
                Ok(())
            };
            for _ in 0..8 {
                dispatch()?;
            }
            device.synchronize()?;
            let iters = 200;
            let t = Instant::now();
            for _ in 0..iters {
                dispatch()?;
            }
            device.synchronize()?;
            let s = t.elapsed().as_secs_f64();
            dispatch()?;
            device.synchronize()?;
            let out = candle::MetalStorage::new(dst.clone(), mdev.clone(), m * n, DType::F32);
            let got = Tensor::from_storage(
                candle::Storage::Metal(out),
                (m, n),
                candle::op::BackpropOp::none(),
                false,
            );
            let denom = refr.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-6);
            let rel = got.sub(&refr)?.abs()?.mean_all()?.to_scalar::<f32>()? / denom;
            println!(
                "Q2_0 MCT   {n}x{k} m={m}: {:.3} ms/call ({:.1} GB/s), rel_err {:.4}",
                1000.0 * s / iters as f64,
                (bytes as f64 * iters as f64) / s / 1e9,
                rel
            );
        }
    }

    // Verify GEMV knob sweep (mcx): NR (amortize activation re-read), NC, NSG,
    // VEC. m=8 (verify width). Label = nr_nc_nsg_vec.
    {
        use candle_metal_kernels::call_quantized_matmul_mv_q2_0_mcx;
        let qtx = QTensor::quantize(&w, GgmlDType::Q2_0)?;
        let data = qtx.data()?;
        let mdev = match device {
            Device::Metal(d) => d.clone(),
            _ => anyhow::bail!("mcx bench needs metal"),
        };
        let wbuf = mdev
            .new_buffer_builder()
            .with_data(&data)
            .with_label("mcx_w")
            .build()?;
        let mm = 8usize;
        let x = Tensor::randn(0f32, 1f32, (mm, k), device)?.to_dtype(DType::BF16)?;
        let refr = x.to_dtype(DType::F32)?.matmul(&wdeq.t()?)?;
        let (xs, xl) = x.storage_and_layout();
        let xbuf = match &*xs {
            candle::Storage::Metal(ms) => ms.buffer().clone(),
            _ => anyhow::bail!("x not metal"),
        };
        let xoff = xl.start_offset() * 2;
        let variants: [(&'static str, usize, usize, usize); 9] = [
            // Control: the ACTUAL mc kernel via THIS harness (same geometry).
            // If == mcx_2_8_2 -> harness/position, not the mcx codegen.
            ("kernel_mul_mv_q2_0_bf16_mc", 2, 8, 2),
            ("kernel_mul_mv_q2_0_bf16_mcx_2_8_2", 2, 8, 2),
            ("kernel_mul_mv_q2_0_bf16_mcx_4_8_2", 4, 8, 2),
            ("kernel_mul_mv_q2_0_bf16_mcx_8_8_2", 8, 8, 2),
            ("kernel_mul_mv_q2_0_bf16_mcx_16_8_2", 16, 8, 2),
            ("kernel_mul_mv_q2_0_bf16_mcx_8_4_2", 8, 4, 2),
            ("kernel_mul_mv_q2_0_bf16_mcx_8_16_2", 8, 16, 2),
            ("kernel_mul_mv_q2_0_bf16_mcx_4_8_4", 4, 8, 4),
            ("kernel_mul_mv_q2_0_bf16_mcx_8_8_4", 8, 8, 4),
        ];
        let nv = variants.len();
        let dsts: Vec<_> = (0..nv)
            .map(|_| {
                mdev.new_buffer_builder()
                    .with_size(mm * n * 4)
                    .with_label("mcx_dst")
                    .build()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dispatch = |name: &'static str,
                        nr: usize,
                        nc: usize,
                        nsg: usize,
                        dst: &std::sync::Arc<candle_metal_kernels::metal::Buffer>|
         -> Result<()> {
            let enc = mdev.command_encoder()?;
            call_quantized_matmul_mv_q2_0_mcx(
                mdev.metal_device(),
                &enc,
                mdev.kernels(),
                name,
                (nr, nc, nsg),
                (mm, n, k),
                &wbuf,
                &xbuf,
                xoff,
                dst,
            )?;
            Ok(())
        };
        // Warmup every variant.
        for (vi, (name, nr, nc, nsg)) in variants.iter().enumerate() {
            for _ in 0..8 {
                dispatch(name, *nr, *nc, *nsg, &dsts[vi])?;
            }
        }
        device.synchronize()?;
        // INTERLEAVED timing: round-robin batches so DVFS clock drift across the
        // run hits every variant equally (a sequential per-variant sweep is
        // position-confounded — later variants run at a drooped clock).
        let rounds = 40usize;
        let batch = 8usize;
        let mut acc = vec![0.0f64; nv];
        for _ in 0..rounds {
            for (vi, (name, nr, nc, nsg)) in variants.iter().enumerate() {
                device.synchronize()?;
                let t = Instant::now();
                for _ in 0..batch {
                    dispatch(name, *nr, *nc, *nsg, &dsts[vi])?;
                }
                device.synchronize()?;
                acc[vi] += t.elapsed().as_secs_f64();
            }
        }
        for (vi, (name, nr, nc, nsg)) in variants.iter().enumerate() {
            let per = acc[vi] / (rounds * batch) as f64;
            // correctness (one clean dispatch + read)
            dispatch(name, *nr, *nc, *nsg, &dsts[vi])?;
            device.synchronize()?;
            let out =
                candle::MetalStorage::new(dsts[vi].clone(), mdev.clone(), mm * n, DType::F32);
            let got = Tensor::from_storage(
                candle::Storage::Metal(out),
                (mm, n),
                candle::op::BackpropOp::none(),
                false,
            );
            let denom = refr.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-6);
            let rel = got.sub(&refr)?.abs()?.mean_all()?.to_scalar::<f32>()? / denom;
            let label = name
                .strip_prefix("kernel_mul_mv_q2_0_bf16_")
                .unwrap_or(name);
            println!(
                "Q2_0 MCX[{label:>12}] {n}x{k} m={mm}: {:.3} ms/call ({:.1} GB/s), rel_err {:.4}",
                1000.0 * per,
                (bytes as f64) / per / 1e9,
                rel
            );
        }
    }

    // Bit-plane popcount ternary GEMV (B3 spike): weights as +/- sign planes
    // (2 bits + f16/128 = the same 2.125 bpw as Q2_0), activations int4
    // bit-sliced — per-weight work is AND+popcount, no unpack, no multiply.
    // Bar to clear: mm2d's ~43 GB/s at m=5-8; roof is the mv's ~106.
    {
        use candle_metal_kernels::call_ternary_bitplane_qmv;
        let kw = k / 32;
        let nb = k / 128;
        let mut s_rng = 0x2468_ace0u32;
        let mut next = move || {
            s_rng = s_rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s_rng
        };
        let mut wpos = vec![0u32; n * kw];
        let mut wneg = vec![0u32; n * kw];
        let mut dsc = vec![0u16; n * nb];
        let mut wdense = vec![0f32; n * k];
        for col in 0..n {
            let dvals: Vec<f32> = (0..nb)
                .map(|_| 0.005f32 + (next() % 1000) as f32 * 1e-5)
                .collect();
            for (blk, d) in dvals.iter().enumerate() {
                dsc[col * nb + blk] = half::f16::from_f32(*d).to_bits();
            }
            for wi in 0..kw {
                let (mut p, mut ng) = (0u32, 0u32);
                for bit in 0..32 {
                    let kk = wi * 32 + bit;
                    // Use the f16-rounded d so the dense reference matches the
                    // kernel's arithmetic exactly.
                    let d = half::f16::from_f32(dvals[kk / 128]).to_f32();
                    match next() % 4 {
                        2 => {
                            p |= 1 << bit;
                            wdense[col * k + kk] = d;
                        }
                        3 => {
                            ng |= 1 << bit;
                            wdense[col * k + kk] = -d;
                        }
                        _ => {}
                    }
                }
                wpos[col * kw + wi] = p;
                wneg[col * kw + wi] = ng;
            }
        }
        let mdev = match device {
            Device::Metal(d) => d.clone(),
            _ => anyhow::bail!("bitplane bench needs metal"),
        };
        let as_u8 = |v: &[u32]| unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
        };
        let wpos_buf = mdev.new_buffer_builder().with_data(as_u8(&wpos)).with_label("bp_wpos").build()?;
        let wneg_buf = mdev.new_buffer_builder().with_data(as_u8(&wneg)).with_label("bp_wneg").build()?;
        let dsc_bytes = unsafe {
            std::slice::from_raw_parts(dsc.as_ptr() as *const u8, dsc.len() * 2)
        };
        let dsc_buf = mdev.new_buffer_builder().with_data(dsc_bytes).with_label("bp_d").build()?;
        let wdense_t = Tensor::from_vec(wdense, (n, k), device)?;
        let plane_bytes = (2 * n * kw * 4 + n * nb * 2) as f64;

        for m in [1usize, 2, 4, 8] {
            let x = Tensor::randn(0f32, 1f32, (m, k), device)?;
            let xv = x.to_vec2::<f32>()?;
            let mut ascale = vec![0f32; m];
            let mut xq = vec![0f32; m * k];
            let mut aplane = vec![0u32; m * 4 * kw];
            for (row, xr) in xv.iter().enumerate() {
                let s = xr.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-8) / 7.0;
                ascale[row] = s;
                for (kk, v) in xr.iter().enumerate() {
                    let q = (v / s).round().clamp(-7.0, 7.0) as i32;
                    xq[row * k + kk] = s * q as f32;
                    let u = (q + 8) as u32;
                    for b in 0..4 {
                        if (u >> b) & 1 == 1 {
                            aplane[(row * 4 + b) * kw + kk / 32] |= 1 << (kk % 32);
                        }
                    }
                }
            }
            let ap_buf = mdev.new_buffer_builder().with_data(as_u8(&aplane)).with_label("bp_ap").build()?;
            let as_bytes = unsafe {
                std::slice::from_raw_parts(ascale.as_ptr() as *const u8, ascale.len() * 4)
            };
            let asc_buf = mdev.new_buffer_builder().with_data(as_bytes).with_label("bp_as").build()?;
            let dst = mdev.new_buffer_builder().with_size(m * n * 4).with_label("bp_dst").build()?;
            let refr_q = Tensor::from_vec(xq.clone(), (m, k), device)?.matmul(&wdense_t.t()?)?;
            let refr_true = x.matmul(&wdense_t.t()?)?;
            let dispatch = || -> Result<()> {
                let enc = mdev.command_encoder()?;
                call_ternary_bitplane_qmv(
                    mdev.metal_device(),
                    &enc,
                    mdev.kernels(),
                    (m, n, k),
                    &wpos_buf,
                    &wneg_buf,
                    &dsc_buf,
                    &ap_buf,
                    &asc_buf,
                    &dst,
                )?;
                Ok(())
            };
            for _ in 0..8 {
                dispatch()?;
            }
            device.synchronize()?;
            let iters = 200;
            let t = Instant::now();
            for _ in 0..iters {
                dispatch()?;
            }
            device.synchronize()?;
            let s = t.elapsed().as_secs_f64();
            dispatch()?;
            device.synchronize()?;
            let out = candle::MetalStorage::new(dst.clone(), mdev.clone(), m * n, DType::F32);
            let got = Tensor::from_storage(
                candle::Storage::Metal(out),
                (m, n),
                candle::op::BackpropOp::none(),
                false,
            );
            let denom = refr_true.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-6);
            // Kernel-vs-int4-reference: should be near-exact (same integers,
            // different float fold order). Act-err: what int4 activations cost.
            let rel_kernel =
                got.sub(&refr_q)?.abs()?.mean_all()?.to_scalar::<f32>()? / denom;
            let act_err =
                refr_q.sub(&refr_true)?.abs()?.mean_all()?.to_scalar::<f32>()? / denom;
            println!(
                "Q2_0 BITPLANE {n}x{k} m={m}: {:.3} ms/call ({:.1} GB/s), rel_vs_int4ref {:.4}, int4_act_err {:.4}",
                1000.0 * s / iters as f64,
                (plane_bytes * iters as f64) / s / 1e9,
                rel_kernel,
                act_err
            );
        }
    }

    // Planar coalesced verify GEMM (q2_0_mm2d): reads a [k][n_pad] repack so the
    // cross-row weight read coalesces (the [row][block] kernels were capped ~21
    // GB/s). This is the DSpark-verify weight-bound lever.
    {
        use candle::quantized::k_quants::BlockQ2_0;
        use candle::quantized::metal::q2_0_mm2d_planes;
        use candle_metal_kernels::{
            call_quantized_matmul_mm2d_q2_0, call_quantized_matmul_mm2d_q2_0_smallm, Mm2dQ2Variant,
        };
        let qtp = QTensor::quantize(&w, GgmlDType::Q2_0)?;
        let data = qtp.data()?;
        let blocks = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const BlockQ2_0, data.len() / 34)
        };
        let planes = q2_0_mm2d_planes(blocks, n, k)?;
        let mdev = match device {
            Device::Metal(d) => d.clone(),
            _ => anyhow::bail!("planar bench needs metal"),
        };
        let codes_buf = mdev
            .new_buffer_builder()
            .with_data(&planes.codes)
            .with_label("q2p_codes")
            .build()?;
        let d_bytes =
            unsafe { std::slice::from_raw_parts(planes.d.as_ptr() as *const u8, planes.d.len() * 2) };
        let d_buf = mdev.new_buffer_builder().with_data(d_bytes).with_label("q2p_d").build()?;
        for m in [1usize, 2, 4, 8] {
            let x = Tensor::randn(0f32, 1f32, (m, k), device)?;
            let refr = x.matmul(&wdeq.t()?)?;
            let run = |mdev: &candle::MetalDevice| -> Result<candle::MetalStorage> {
                let (xs, xl) = x.storage_and_layout();
                let xbuf = match &*xs {
                    candle::Storage::Metal(ms) => ms.buffer().clone(),
                    _ => anyhow::bail!("x not metal"),
                };
                let dst = mdev.new_buffer_builder().with_size(m * n * 4).with_label("q2p_dst").build()?;
                let enc = mdev.command_encoder()?;
                call_quantized_matmul_mm2d_q2_0_smallm(
                    mdev.metal_device(),
                    &enc,
                    mdev.kernels(),
                    (m, n, k, planes.n_pad),
                    &codes_buf,
                    &d_buf,
                    &xbuf,
                    xl.start_offset() * 4,
                    0,
                    &dst,
                )?;
                Ok(candle::MetalStorage::new(dst, mdev.clone(), m * n, DType::F32))
            };
            for _ in 0..8 {
                let _ = run(&mdev)?;
            }
            device.synchronize()?;
            let iters = 200;
            let t = Instant::now();
            for _ in 0..iters {
                let _ = run(&mdev)?;
            }
            device.synchronize()?;
            let s = t.elapsed().as_secs_f64();
            let out = run(&mdev)?;
            let got = Tensor::from_storage(
                candle::Storage::Metal(out),
                (m, n),
                candle::op::BackpropOp::none(),
                false,
            );
            let denom = refr.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-6);
            let rel = got.sub(&refr)?.abs()?.mean_all()?.to_scalar::<f32>()? / denom;
            println!(
                "Q2_0 PLANAR {n}x{k} m={m}: {:.3} ms/call ({:.1} GB/s), rel_err {:.4}",
                1000.0 * s / iters as f64,
                (bytes as f64 * iters as f64) / s / 1e9,
                rel
            );

            // Hardware tensor-op path (matmul2d, uint2b_format B) — sweep every
            // compile-time variant (K-tile / tile-N / relaxed-precision) to find
            // the weight-bound optimum. Same planes; bf16 in / bf16 out; the
            // 2-bit lanes unpack in silicon (no software staging).
            let xbf = x.to_dtype(DType::BF16)?;
            let (xs, xl) = xbf.storage_and_layout();
            let xbuf = match &*xs {
                candle::Storage::Metal(ms) => ms.buffer().clone(),
                _ => anyhow::bail!("x not metal"),
            };
            let xoff = xl.start_offset() * 2;
            for variant in Mm2dQ2Variant::ALL {
                // One reused dst for the timing loop (no per-call allocation, so
                // we measure the kernel, not buffer/residency churn).
                let dst = mdev
                    .new_buffer_builder()
                    .with_size(m * n * 2)
                    .with_label("q2mm2d_dst")
                    .build()?;
                let dispatch = |mdev: &candle::MetalDevice| -> Result<()> {
                    let enc = mdev.command_encoder()?;
                    call_quantized_matmul_mm2d_q2_0(
                        mdev.metal_device(),
                        &enc,
                        mdev.kernels(),
                        (m, n, planes.n_pad, k),
                        &xbuf,
                        xoff,
                        &codes_buf,
                        &d_buf,
                        0,
                        &dst,
                        variant,
                    )?;
                    Ok(())
                };
                for _ in 0..8 {
                    dispatch(&mdev)?;
                }
                device.synchronize()?;
                let t = Instant::now();
                for _ in 0..iters {
                    dispatch(&mdev)?;
                }
                device.synchronize()?;
                let s2 = t.elapsed().as_secs_f64();
                // Correctness: the last dispatch left a valid result in dst.
                dispatch(&mdev)?;
                device.synchronize()?;
                let out2 =
                    candle::MetalStorage::new(dst.clone(), mdev.clone(), m * n, DType::BF16);
                let got2 = Tensor::from_storage(
                    candle::Storage::Metal(out2),
                    (m, n),
                    candle::op::BackpropOp::none(),
                    false,
                )
                .to_dtype(DType::F32)?;
                let rel2 = got2.sub(&refr)?.abs()?.mean_all()?.to_scalar::<f32>()? / denom;
                let label = variant
                    .kernel
                    .strip_prefix("kernel_mul_mm2d_q2_0_")
                    .unwrap_or(variant.kernel);
                println!(
                    "Q2_0 MM2D[{label:>15}] {n}x{k} m={m}: {:.3} ms/call ({:.1} GB/s), rel_err {:.4}",
                    1000.0 * s2 / iters as f64,
                    (bytes as f64 * iters as f64) / s2 / 1e9,
                    rel2
                );
            }
        }
    }

    // Q4K MM2D reference on the SAME shape: the proven-fast tensor-op path
    // (kernel_mul_mm2d_q4k_bf16, ~140 GB/s). This is the control that answers
    // whether a slow Q2_0 mm2d is a 2-bit hardware property or our kernel — if
    // q4_K reaches its usual bandwidth here, the gap is ours to close.
    {
        use candle::quantized::k_quants::BlockQ4K;
        use candle::quantized::metal::q4k_mm2d_planes;
        use candle_metal_kernels::call_quantized_matmul_mm2d_q4k;
        let qt4 = QTensor::quantize(&w, GgmlDType::Q4K)?;
        let bytes4 = qt4.storage_size_in_bytes();
        let wdeq4 = qt4.dequantize(device)?;
        let data = qt4.data()?;
        let blocks = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const BlockQ4K,
                data.len() / std::mem::size_of::<BlockQ4K>(),
            )
        };
        let planes = q4k_mm2d_planes(blocks, n, k)?;
        let mdev = match device {
            Device::Metal(d) => d.clone(),
            _ => anyhow::bail!("q4k ref bench needs metal"),
        };
        let nib_buf = mdev
            .new_buffer_builder()
            .with_data(&planes.nibbles)
            .with_label("q4k_nib")
            .build()?;
        let dsc_bytes = unsafe {
            std::slice::from_raw_parts(planes.dsc.as_ptr() as *const u8, planes.dsc.len() * 2)
        };
        let dmm_bytes = unsafe {
            std::slice::from_raw_parts(planes.dmm.as_ptr() as *const u8, planes.dmm.len() * 2)
        };
        let dsc_buf = mdev
            .new_buffer_builder()
            .with_data(dsc_bytes)
            .with_label("q4k_dsc")
            .build()?;
        let dmm_buf = mdev
            .new_buffer_builder()
            .with_data(dmm_bytes)
            .with_label("q4k_dmm")
            .build()?;
        let iters = 200;
        for m in [1usize, 2, 4, 8] {
            let x = Tensor::randn(0f32, 1f32, (m, k), device)?.to_dtype(DType::BF16)?;
            let refr = x.to_dtype(DType::F32)?.matmul(&wdeq4.t()?)?;
            let (xs, xl) = x.storage_and_layout();
            let xbuf = match &*xs {
                candle::Storage::Metal(ms) => ms.buffer().clone(),
                _ => anyhow::bail!("x not metal"),
            };
            let xoff = xl.start_offset() * 2;
            let dst = mdev
                .new_buffer_builder()
                .with_size(m * n * 2)
                .with_label("q4k_dst")
                .build()?;
            let dispatch = |mdev: &candle::MetalDevice| -> Result<()> {
                let enc = mdev.command_encoder()?;
                call_quantized_matmul_mm2d_q4k(
                    mdev.metal_device(),
                    &enc,
                    mdev.kernels(),
                    (m, n, planes.n_pad, k),
                    &xbuf,
                    xoff,
                    &nib_buf,
                    &dsc_buf,
                    &dmm_buf,
                    0,
                    &dst,
                )?;
                Ok(())
            };
            for _ in 0..8 {
                dispatch(&mdev)?;
            }
            device.synchronize()?;
            let t = Instant::now();
            for _ in 0..iters {
                dispatch(&mdev)?;
            }
            device.synchronize()?;
            let s = t.elapsed().as_secs_f64();
            dispatch(&mdev)?;
            device.synchronize()?;
            let out = candle::MetalStorage::new(dst.clone(), mdev.clone(), m * n, DType::BF16);
            let got = Tensor::from_storage(
                candle::Storage::Metal(out),
                (m, n),
                candle::op::BackpropOp::none(),
                false,
            )
            .to_dtype(DType::F32)?;
            let denom = refr.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-6);
            let rel = got.sub(&refr)?.abs()?.mean_all()?.to_scalar::<f32>()? / denom;
            println!(
                "Q4K  MM2D (ref) {n}x{k} m={m}: {:.3} ms/call ({:.1} GB/s), rel_err {:.4}",
                1000.0 * s / iters as f64,
                (bytes4 as f64 * iters as f64) / s / 1e9,
                rel
            );
        }
    }
    Ok(())
}

/// Loop ONE quantized matmul kernel in isolation (17408x5120 ffn shape) so a
/// gpucapture capture is dominated by it — gpudebug counters are
/// timeline-aggregate, so isolation is how we attribute occupancy/ALU/bandwidth
/// to a single kernel. Enqueues `iters` dispatches then one sync. Correctness is
/// not checked here (that's bench-gemv's job); this exists purely for profiling.
fn profile_kernel(device: &Device, which: &str, iters: usize, m: usize) -> Result<()> {
    use candle::quantized::k_quants::{BlockQ2_0, BlockQ4K};
    use candle::quantized::metal::{q2_0_mm2d_planes, q4k_mm2d_planes};
    use candle::quantized::{GgmlDType, QTensor};
    use candle_metal_kernels::{
        call_quantized_matmul_mm2d_q2_0, call_quantized_matmul_mm2d_q4k, Mm2dQ2Variant,
    };
    use lmbrrr::quantized_linear::MixedLinear;

    let (n, k) = (17408usize, 5120usize);
    let w = Tensor::randn(0f32, 1f32, (n, k), device)?;
    let mdev = match device {
        Device::Metal(d) => d.clone(),
        _ => anyhow::bail!("profile-kernel needs metal"),
    };
    let ctx = ModelCtx::default();
    eprintln!("profile-kernel: which={which} iters={iters} m={m} shape={n}x{k}");

    // Occupancy-limiter diagnosis: print the two per-pipeline resources that cap
    // occupancy — staticThreadgroupMemoryLength (bytes) and
    // maxTotalThreadsPerThreadgroup (< 1024 ⇒ register pressure). No dispatch.
    if which == "pipeline-info" {
        use candle_metal_kernels::source::Source;
        let kernels = mdev.kernels();
        let dev = mdev.metal_device();
        println!("{:<42} {:>8} {:>14}", "kernel", "maxTPT", "tgMem(B)");
        for v in Mm2dQ2Variant::ALL {
            let p = kernels.load_pipeline(dev, Source::Mm2dQ2_0, v.kernel)?;
            println!(
                "{:<42} {:>8} {:>14}",
                v.kernel,
                p.max_total_threads_per_threadgroup(),
                p.static_threadgroup_memory_length()
            );
        }
        for name in ["kernel_mul_mm2d_q4k_bf16", "kernel_mul_mm2d_q4k_bf16_t32"] {
            let p = kernels.load_pipeline(dev, Source::Mm2dQ4k, name)?;
            println!(
                "{:<42} {:>8} {:>14}",
                name,
                p.max_total_threads_per_threadgroup(),
                p.static_threadgroup_memory_length()
            );
        }
        // GEMV verify kernels (register-pressure diagnosis: maxTPT < 1024 ⇒
        // register-limited occupancy).
        for name in [
            "kernel_mul_mv_q2_0_bf16",
            "kernel_mul_mv_q2_0_bf16_mc",
            "kernel_mul_mv_q2_0_bf16_mc2",
            "kernel_mul_mv_q2_0_bf16_mct",
        ] {
            let p = kernels.load_pipeline(dev, Source::Quantized, name)?;
            println!(
                "{:<42} {:>8} {:>14}",
                name,
                p.max_total_threads_per_threadgroup(),
                p.static_threadgroup_memory_length()
            );
        }
        // DeltaNet recurrence kernels (occupancy-cap diagnosis: staticTgMem
        // ~26KB -> ~1 threadgroup/core on the 32KB M3; maxTPT < 1024 -> also
        // register-limited). These run the hot verify + prefill.
        for (src, name) in [
            (
                candle_metal_kernels::source::Source::GatedDeltaChunk,
                "gated_delta_chunk_bf16_l5",
            ),
            (
                candle_metal_kernels::source::Source::GatedDeltaChunk,
                "gated_delta_chunk_bf16_l8",
            ),
            (
                candle_metal_kernels::source::Source::GatedDeltaChunk,
                "gated_delta_chunk_bf16_l12",
            ),
            (
                candle_metal_kernels::source::Source::GatedDeltaPrefill,
                "gated_delta_prefill_bf16",
            ),
            // v2 (re-gridded) chunk path — the l>=8 GQA verify + prefill route.
            // GD2_MAX_L is a flat #define(12); q_raw+k_raw = 2*12*128*4 = 12KB
            // in prep alone. If prep/core are >16KB the same occupancy lever
            // (template GD2_MAX_L to the width) extends to width-7 verify.
            (
                candle_metal_kernels::source::Source::GatedDeltaV2,
                "gated_delta_v2_prep_bf16",
            ),
            (
                candle_metal_kernels::source::Source::GatedDeltaV2,
                "gated_delta_v2_core",
            ),
            (
                candle_metal_kernels::source::Source::GatedDeltaV2,
                "gated_delta_v2_epilogue_bf16",
            ),
            // The m=1 decode kernel (25% of the decode floor): maxTPT<1024 =>
            // register-limited (an occupancy lever exists); =1024 => the 34%
            // occupancy is grid-limited (48 tg = one/value-head), structural.
            (
                candle_metal_kernels::source::Source::GatedDeltaV2,
                "gated_delta_v2_decode_bf16",
            ),
        ] {
            let p = kernels.load_pipeline(dev, src, name)?;
            println!(
                "{:<42} {:>8} {:>14}",
                name,
                p.max_total_threads_per_threadgroup(),
                p.static_threadgroup_memory_length()
            );
        }
        return Ok(());
    }

    // Build whichever inputs the chosen kernel needs, then a `dispatch` closure.
    let q2_variant = match which {
        "mm2d-k32" => Some(Mm2dQ2Variant::T64_K32),
        "mm2d-k128" => Some(Mm2dQ2Variant::T64_K128),
        _ => None,
    };
    if which == "mct" {
        // Transposed-activation verify GEMV, isolated for counter capture.
        use candle_metal_kernels::call_quantized_matmul_mv_q2_0_mct;
        let qtm = QTensor::quantize(&w, GgmlDType::Q2_0)?;
        let data = qtm.data()?;
        let wbuf = mdev
            .new_buffer_builder()
            .with_data(&data)
            .with_label("pk_mct_w")
            .build()?;
        let x = Tensor::randn(0f32, 1f32, (m, k), device)?.to_dtype(DType::BF16)?;
        let xt = x.t()?.contiguous()?;
        let (xs, xl) = xt.storage_and_layout();
        let xtbuf = match &*xs {
            candle::Storage::Metal(ms) => ms.buffer().clone(),
            _ => anyhow::bail!("xt not metal"),
        };
        let xoff = xl.start_offset() * 2;
        let dst = mdev
            .new_buffer_builder()
            .with_size(m * n * 4)
            .with_label("pk_mct_dst")
            .build()?;
        let dispatch = || -> Result<()> {
            let enc = mdev.command_encoder()?;
            call_quantized_matmul_mv_q2_0_mct(
                mdev.metal_device(),
                &enc,
                mdev.kernels(),
                (m, n, k),
                &wbuf,
                &xtbuf,
                xoff,
                &dst,
            )?;
            Ok(())
        };
        for _ in 0..8 {
            dispatch()?;
        }
        device.synchronize()?;
        for _ in 0..iters {
            dispatch()?;
        }
        device.synchronize()?;
        eprintln!("profile-kernel: done ({iters} dispatches)");
        return Ok(());
    }
    if which == "mv" || which == "mc" {
        // GEMV path via MixedLinear: "mv" = decode m=1 (bandwidth baseline);
        // "mc" = verify at width m (the weight-shared-columns kernel). Set
        // LMBRRR_Q2_MC2=1 to route "mc" to the ILP variant.
        let mrows = if which == "mv" { 1 } else { m };
        let qt = QTensor::quantize(&w, GgmlDType::Q2_0)?;
        let lin = MixedLinear::from_qtensor(qt, ctx.mm2d.clone())?;
        let x = Tensor::randn(0f32, 1f32, (1, mrows, k), device)?.to_dtype(DType::BF16)?;
        for _ in 0..8 {
            let _ = lin.forward(&x)?;
        }
        device.synchronize()?;
        let t = Instant::now();
        for _ in 0..iters {
            let _ = lin.forward(&x)?;
        }
        device.synchronize()?;
        // Q2_0 weight bytes/dispatch = n*k*2.125/8; the dominant read at m=1.
        let secs = t.elapsed().as_secs_f64();
        let wbytes = (n * k) as f64 * 2.125 / 8.0;
        let gbps = wbytes * iters as f64 / secs / 1e9;
        eprintln!(
            "profile-kernel mv timing: {iters} disp in {:.3}s = {:.3} ms/disp, {:.1} GB/s (weight-read, m={mrows})",
            secs,
            secs * 1e3 / iters as f64,
            gbps
        );
    } else if let Some(variant) = q2_variant {
        let qt = QTensor::quantize(&w, GgmlDType::Q2_0)?;
        let data = qt.data()?;
        let blocks =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const BlockQ2_0, data.len() / 34) };
        let planes = q2_0_mm2d_planes(blocks, n, k)?;
        let codes_buf = mdev
            .new_buffer_builder()
            .with_data(&planes.codes)
            .with_label("pk_codes")
            .build()?;
        let d_bytes = unsafe {
            std::slice::from_raw_parts(planes.d.as_ptr() as *const u8, planes.d.len() * 2)
        };
        let d_buf = mdev
            .new_buffer_builder()
            .with_data(d_bytes)
            .with_label("pk_d")
            .build()?;
        let x = Tensor::randn(0f32, 1f32, (m, k), device)?.to_dtype(DType::BF16)?;
        let (xs, xl) = x.storage_and_layout();
        let xbuf = match &*xs {
            candle::Storage::Metal(ms) => ms.buffer().clone(),
            _ => anyhow::bail!("x not metal"),
        };
        let xoff = xl.start_offset() * 2;
        let dst = mdev
            .new_buffer_builder()
            .with_size(m * n * 2)
            .with_label("pk_dst")
            .build()?;
        let dispatch = || -> Result<()> {
            let enc = mdev.command_encoder()?;
            call_quantized_matmul_mm2d_q2_0(
                mdev.metal_device(),
                &enc,
                mdev.kernels(),
                (m, n, planes.n_pad, k),
                &xbuf,
                xoff,
                &codes_buf,
                &d_buf,
                0,
                &dst,
                variant,
            )?;
            Ok(())
        };
        for _ in 0..8 {
            dispatch()?;
        }
        device.synchronize()?;
        for _ in 0..iters {
            dispatch()?;
        }
        device.synchronize()?;
    } else if which == "q4k-mm2d" {
        let qt = QTensor::quantize(&w, GgmlDType::Q4K)?;
        let data = qt.data()?;
        let blocks = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const BlockQ4K,
                data.len() / std::mem::size_of::<BlockQ4K>(),
            )
        };
        let planes = q4k_mm2d_planes(blocks, n, k)?;
        let nib_buf = mdev
            .new_buffer_builder()
            .with_data(&planes.nibbles)
            .with_label("pk_nib")
            .build()?;
        let dsc_bytes = unsafe {
            std::slice::from_raw_parts(planes.dsc.as_ptr() as *const u8, planes.dsc.len() * 2)
        };
        let dmm_bytes = unsafe {
            std::slice::from_raw_parts(planes.dmm.as_ptr() as *const u8, planes.dmm.len() * 2)
        };
        let dsc_buf = mdev
            .new_buffer_builder()
            .with_data(dsc_bytes)
            .with_label("pk_dsc")
            .build()?;
        let dmm_buf = mdev
            .new_buffer_builder()
            .with_data(dmm_bytes)
            .with_label("pk_dmm")
            .build()?;
        let x = Tensor::randn(0f32, 1f32, (m, k), device)?.to_dtype(DType::BF16)?;
        let (xs, xl) = x.storage_and_layout();
        let xbuf = match &*xs {
            candle::Storage::Metal(ms) => ms.buffer().clone(),
            _ => anyhow::bail!("x not metal"),
        };
        let xoff = xl.start_offset() * 2;
        let dst = mdev
            .new_buffer_builder()
            .with_size(m * n * 2)
            .with_label("pk_dst")
            .build()?;
        let dispatch = || -> Result<()> {
            let enc = mdev.command_encoder()?;
            call_quantized_matmul_mm2d_q4k(
                mdev.metal_device(),
                &enc,
                mdev.kernels(),
                (m, n, planes.n_pad, k),
                &xbuf,
                xoff,
                &nib_buf,
                &dsc_buf,
                &dmm_buf,
                0,
                &dst,
            )?;
            Ok(())
        };
        for _ in 0..8 {
            dispatch()?;
        }
        device.synchronize()?;
        for _ in 0..iters {
            dispatch()?;
        }
        device.synchronize()?;
    } else {
        anyhow::bail!("unknown --which {which:?}; use mv | mm2d-k32 | mm2d-k128 | q4k-mm2d");
    }
    eprintln!("profile-kernel: done ({iters} dispatches)");
    Ok(())
}

/// Isolated DeltaNet-prefill kernel loop (Bonsai shapes) for capture/counter
/// attribution: streaming (one dispatch/seq) vs chunk (host loop over l/12).
fn bench_deltanet(device: &Device, which: &str, l: usize, iters: usize) -> Result<()> {
    use lmbrrr::fused_deltanet::{
        gated_delta_chunk, gated_delta_prefill, gated_delta_v2_decode, GatedDeltaDecodeWeights,
        GatedDeltaDims,
    };
    // Bonsai DeltaNet dims.
    let (heads, num_k_heads, dk, dv, ksz) = (48usize, 16usize, 128usize, 128usize, 4usize);
    let key_dim = num_k_heads * dk;
    let value_dim = heads * dv;
    let conv_dim = 2 * key_dim + value_dim;
    let row_stride = conv_dim + value_dim + 2 * heads;
    let dims = GatedDeltaDims {
        heads,
        dk,
        dv,
        conv_dim,
        key_dim,
        value_dim,
        ksz,
        num_k_heads,
    };
    let proj = Tensor::randn(0f32, 1f32, (1, l, row_stride), device)?.to_dtype(DType::BF16)?;
    let conv_state = Tensor::zeros((1, conv_dim, ksz), DType::BF16, device)?;
    let recurrent_state = Tensor::zeros((1, heads, dk, dv), DType::F32, device)?;
    let conv_w = Tensor::randn(0f32, 0.1f32, (conv_dim, ksz), device)?.to_dtype(DType::BF16)?;
    let dt_bias = Tensor::zeros((heads,), DType::F32, device)?;
    let a_log_exp = (Tensor::randn(0f32, 0.1f32, (heads,), device)?.abs()? + 0.5)?;
    let norm_w = Tensor::ones((dv,), DType::F32, device)?;

    // v2 decode (l=1) inputs: split qkvz/ba projections + transposed state.
    let qkvz = Tensor::randn(0f32, 1f32, (1, 1, conv_dim + value_dim), device)?
        .to_dtype(DType::BF16)?;
    let ba = Tensor::randn(0f32, 1f32, (1, 1, 2 * heads), device)?.to_dtype(DType::BF16)?;
    let state_t = Tensor::zeros((1, heads, dv, dk), DType::F32, device)?;
    let decode_weights = GatedDeltaDecodeWeights {
        conv_weight: &conv_w,
        dt_bias_f32: &dt_bias,
        a_log_exp_f32: &a_log_exp,
        norm_weight_f32: &norm_w,
        dims: &dims,
        l2_eps: 1e-6,
        norm_eps: 1e-6,
    };
    let run_decode = || -> Result<()> {
        let _ = gated_delta_v2_decode(
            &qkvz.flatten_all()?.contiguous()?,
            &ba.flatten_all()?.contiguous()?,
            1,
            &conv_state,
            &state_t.contiguous()?,
            &decode_weights,
        )?;
        Ok(())
    };

    let n_chunks = l.div_ceil(12);
    eprintln!(
        "bench-deltanet which={which} l={l} iters={iters} heads={heads} (stream=1 dispatch/seq, chunk={n_chunks} dispatches/seq)"
    );
    // Warmup.
    for _ in 0..5 {
        match which {
            "stream" => {
                let _ = gated_delta_prefill(
                    &proj.flatten_to(1)?, l, &conv_state, &recurrent_state,
                    &conv_w, &dt_bias, &a_log_exp, &norm_w, &dims, 1e-6, 1e-6,
                )?;
            }
            "chunk" => {
                for ci in 0..n_chunks {
                    let start = ci * 12;
                    let c = 12.min(l - start);
                    let pc = proj.narrow(1, start, c)?.flatten_to(1)?;
                    let _ = gated_delta_chunk(
                        &pc, c, &conv_state, &recurrent_state,
                        &conv_w, &dt_bias, &a_log_exp, &norm_w, &dims, 1e-6, 1e-6,
                    )?;
                }
            }
            "decode" => run_decode()?,
            other => anyhow::bail!("unknown --which {other}; use stream | chunk | decode"),
        }
    }
    device.synchronize()?;
    let t = Instant::now();
    for _ in 0..iters {
        match which {
            "stream" => {
                let _ = gated_delta_prefill(
                    &proj.flatten_to(1)?, l, &conv_state, &recurrent_state,
                    &conv_w, &dt_bias, &a_log_exp, &norm_w, &dims, 1e-6, 1e-6,
                )?;
            }
            "chunk" => {
                for ci in 0..n_chunks {
                    let start = ci * 12;
                    let c = 12.min(l - start);
                    let pc = proj.narrow(1, start, c)?.flatten_to(1)?;
                    let _ = gated_delta_chunk(
                        &pc, c, &conv_state, &recurrent_state,
                        &conv_w, &dt_bias, &a_log_exp, &norm_w, &dims, 1e-6, 1e-6,
                    )?;
                }
            }
            "decode" => run_decode()?,
            _ => unreachable!(),
        }
    }
    device.synchronize()?;
    let s = t.elapsed().as_secs_f64();
    // Per-seq cost = one layer's DeltaNet at this l; x48 layers approximates the
    // per-forward DeltaNet wall (no MLP/attn/proj).
    println!(
        "{which}: {:.3} ms/seq ({:.1} seq/s over {iters} iters); x48 layers ~= {:.1} ms/forward",
        1000.0 * s / iters as f64,
        iters as f64 / s,
        1000.0 * s / iters as f64 * 48.0,
    );
    Ok(())
}

fn profile_decode(
    model: &mut Qwen35CausalLM,
    device: &Device,
    ids: &[u32],
    width: usize,
    capture_taps: Option<&str>,
) -> Result<()> {
    use candle::D;
    use std::collections::HashMap;

    let taps: Option<Vec<usize>> = capture_taps
        .map(|s| {
            s.split(',')
                .map(|t| t.trim().parse::<usize>())
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .map_err(|e| anyhow::anyhow!("bad --capture-taps: {e}"))?;

    let prof = Qwen35Profiler::new();
    model.set_profiler(Some(prof.clone()));

    // Prefill, then discard its events — we profile steady-state decode only.
    let input = Tensor::from_slice(ids, (1, ids.len()), device)?;
    let mut logits = model.forward(&input, 0)?;
    device.synchronize()?;
    prof.clear();

    let steps = 24usize;
    let mut offset = ids.len();
    if width == 1 {
        for _ in 0..steps {
            let next = logits
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .flatten_all()?
                .to_vec1::<u32>()?[0];
            let step = Tensor::from_slice(&[next], (1, 1), device)?;
            logits = model.forward(&step, offset)?;
            device.synchronize()?;
            offset += 1;
        }
    } else {
        // Verify-shaped chunks: forward `width` tokens per step through the
        // full-logits path (what the spec verify runs). Token identity is
        // irrelevant to timing; reuse the prompt's first `width` ids.
        let chunk: Vec<u32> = ids.iter().cycle().take(width).copied().collect();
        let wall = Instant::now();
        for _ in 0..steps {
            let chunk_input = Tensor::from_slice(&chunk, (1, width), device)?;
            if let Some(taps) = &taps {
                model.set_device_capture(Some(taps.clone()));
            }
            let logits = model.forward_all_logits(&chunk_input, offset)?;
            // Include the verify's argmax readback in the step, as spec does.
            let _ = logits
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            if taps.is_some() {
                let caps = model.take_device_captures();
                let _ = Tensor::cat(&caps, D::Minus1)?;
            }
            device.synchronize()?;
            offset += width;
        }
        eprintln!(
            "chunk wall (serialized, incl. profiler syncs): {:.2} ms/step, capture={}",
            wall.elapsed().as_secs_f64() * 1000.0 / steps as f64,
            taps.is_some()
        );
    }

    // Aggregate step events (seq_len == width) by component.
    let mut agg: HashMap<String, (f64, usize)> = HashMap::new();
    for e in prof.events() {
        if e.seq_len != width {
            continue;
        }
        let slot = agg.entry(e.component.clone()).or_default();
        slot.0 += e.seconds;
        slot.1 += 1;
    }
    let total: f64 = agg.values().map(|v| v.0).sum();
    let mut rows: Vec<_> = agg.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
    println!(
        "profiled {steps} steps at width {width}; summed op-time {:.1} ms ({:.2} ms/step)",
        total * 1000.0,
        total * 1000.0 / steps as f64
    );
    println!("{:<40} {:>10} {:>8} {:>6}", "component", "ms/step", "calls", "pct");
    for (name, (secs, calls)) in rows {
        println!(
            "{:<40} {:>10.3} {:>8} {:>6.1}",
            name,
            secs * 1000.0 / steps as f64,
            calls,
            100.0 * secs / total
        );
    }
    Ok(())
}

fn argmax_row(logits: &Tensor) -> Result<u32> {
    Ok(logits
        .argmax(D::Minus1)?
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1::<u32>()?[0])
}

/// DSpark speculative decode: prefill+capture the target's tap layers, seed the
/// drafter context, then per round draft a block, verify it in one target pass,
/// accept the longest exact-match prefix, roll the target KV back to the
/// commit point, and extend the drafter context with the committed captures.
#[allow(clippy::too_many_arguments)]
fn spec_decode(
    model: &mut Qwen35CausalLM,
    drafter_path: &std::path::Path,
    device: &Device,
    ids: &[u32],
    eos: u32,
    max_new_tokens: usize,
    ctx: &ModelCtx,
    tok: &tokenizers::Tokenizer,
    readvance_rollback: bool,
    accept_margin: Option<f32>,
    margin_peak: Option<f32>,
    use_pld: bool,
    use_tree: bool,
    skip_low_conf: Option<f32>,
    skip_after_reject: bool,
) -> Result<()> {
    use lmbrrr::dspark::DsparkDrafter;
    let load = Instant::now();
    let dgguf = GgufFile::open(drafter_path)?;
    // The drafter gets its OWN ModelCtx: sharing the target's mm2d scratch made
    // the drafter's quantized backbone read corrupted state (block_hidden -> 0).
    let drafter_ctx = ModelCtx::default();
    let mut drafter = DsparkDrafter::load_gguf(&dgguf, device, DType::BF16, &drafter_ctx, false)?;
    let _ = ctx;
    let layers = drafter.config.target_layer_ids.clone();
    let width = drafter.config.block_size;
    let drafter_load_s = load.elapsed().as_secs_f64();

    // Residency audit: exceeding the GPU working-set budget does NOT error on
    // Metal — it silently corrupts resident buffers (seen twice on the 18 GB
    // M3: drafter-weight corruption pre-packed-embed, and f32-1.0 token ids
    // when the full ~5.3 GB of mm2d Q2_0 planes were built). Surface the
    // numbers so an over-budget run is diagnosable from its log.
    if let Device::Metal(dev) = device {
        let allocated = dev.current_allocated_size() as f64 / 1e9;
        let budget = dev.recommended_max_working_set_size() as f64 / 1e9;
        eprintln!("gpu allocated {allocated:.2} GB / working-set budget {budget:.2} GB");
        if allocated > budget {
            eprintln!("warning: GPU allocation exceeds the working-set budget — expect silent buffer corruption, reduce residency (LMBRRR_MM2D_MIN_N, packed embed)");
        }
    }

    let dbg = std::env::var(lmbrrr::env_keys::SPEC_DEBUG).is_ok();
    let oracle_log = std::env::var(lmbrrr::env_keys::ORACLE_LOG).is_ok();
    let accept_probe = std::env::var(lmbrrr::env_keys::ACCEPT_PROBE).is_ok();
    // Per-round records for offline scheduler EV (confidence + exact accept mask).
    let mut oracle_rounds: Vec<serde_json::Value> = Vec::new();
    // Per-position accept-probe rows (checkpoint hidden RMS + labels).
    let mut accept_probe_rows: Vec<serde_json::Value> = Vec::new();
    // Full-attn-ish checkpoints on 64L / every-4th full attn (0-based 15,31,47,63).
    const PROBE_LAYERS: [usize; 4] = [15, 31, 47, 63];

    // Partial-accept rollback strategy: capture-based closed-form
    // reconstruction by default (no re-forward; the capture is one S0 + small
    // per-position vectors per layer, ~150 MB total — NOT the per-position
    // state capture that OOM'd historically). LMBRRR_READVANCE_ROLLBACK=1
    // restores the restore+re-forward path (bit-faithful, ~2x the rollback
    // cost per partial accept).
    model.set_verify_state_capture(!readvance_rollback);

    // Prefill with tap-layer capture, seed the drafter context.
    model.clear_cache();
    drafter.clear_context();
    model.set_device_capture(Some(layers.clone()));
    let input = Tensor::from_slice(ids, (1, ids.len()), device)?;
    let logits = model.forward_all_logits(&input, 0)?;
    device.synchronize()?;
    let caps = model.take_device_captures();
    let ctx_feat = Tensor::cat(&caps, D::Minus1)?;
    drop(caps);
    drafter.append_context(&ctx_feat, 0)?;
    if dbg {
        eprintln!("drafter loaded ({drafter_load_s:.1}s), prefill done, offset={}", ids.len());
    }

    let mut anchor = argmax_row(&logits.narrow(1, ids.len() - 1, 1)?)?;
    drop(logits);
    drop(ctx_feat);
    // The anchor IS the first generated token (the prefill argmax): commit
    // it. It was silently dropped — spec output was greedy's shifted by one,
    // invisible to the eye (token 1 is usually whitespace) and caught only by
    // the teacher-forced score gate (pos-0 logprob -10 on every run).
    // Prompt-lookup drafting (PLD): zero-cost copy proposals from verbatim
    // n-gram matches over prompt+committed text. Fires only when the match
    // is at least drafter-width wide — ungated PLD preempts strong drafter
    // rounds and loses (measured -13% on math in the MiniCPM campaign).
    let mut ngram_index = lmbrrr::ngram_draft::NgramDraftIndex::new(3, 4);
    ngram_index.extend(ids);
    let mut indexed = 0usize;
    let mut pld_rounds = 0usize;
    let mut pld_accepted = 0usize;
    let mut tree_rounds = 0usize;
    let mut alt_wins = 0usize;
    let mut offset = ids.len();
    // The anchor is prefill-produced but counted in the decode window —
    // symmetric with plain decode, whose first token is also the prefill argmax.
    let mut committed: Vec<u32> = vec![anchor];
    let mut committed_raw = 1usize;
    let mut rounds = 0usize;
    let mut accepted_total = 0usize;
    let mut propose_s = 0.0f64;
    let mut verify_s = 0.0f64;
    let mut rollback_s = 0.0f64;
    let mut round_wall_ms: Vec<f64> = Vec::new();
    let mut round_accepted: Vec<usize> = Vec::new();
    let mut flat_rows = 0usize;
    let mut skip_rounds = 0usize;
    let mut skip_tokens = 0usize;
    let mut skip_next = false; // set after accepted==0 when skip_after_reject

    let decode = Instant::now();
    while committed.len() < max_new_tokens && committed.last() != Some(&eos) {
        let round_t = Instant::now();

        // Reactive skip: previous round fully rejected → plain greedy step.
        if skip_next {
            skip_next = false;
            model.set_device_capture(Some(layers.clone()));
            let step = Tensor::from_slice(&[anchor], (1, 1), device)?;
            let tv = Instant::now();
            let logits = model.forward_all_logits(&step, offset)?;
            let next = argmax_row(&logits)?;
            verify_s += tv.elapsed().as_secs_f64();
            drop(logits);
            let caps = model.take_device_captures();
            let ctx_feat = Tensor::cat(&caps, D::Minus1)?;
            drop(caps);
            drafter.append_context(&ctx_feat, offset)?;
            drop(ctx_feat);
            offset += 1;
            committed.push(next);
            committed_raw += 1;
            rounds += 1;
            skip_rounds += 1;
            skip_tokens += 1;
            anchor = next;
            device.synchronize()?;
            round_wall_ms.push(round_t.elapsed().as_secs_f64() * 1000.0);
            round_accepted.push(0);
            if dbg {
                eprintln!("round {rounds}: SKIP after-reject -> greedy {next}");
            }
            if next == eos {
                break;
            }
            continue;
        }

        if dbg {
            eprintln!("round {}: snapshot...", rounds + 1);
        }
        let snapshot = model.snapshot_decode_state();
        if dbg {
            eprintln!("round {}: anchor={anchor} propose...", rounds + 1);
        }
        // Keep the lookup index in sync with committed text, then prefer a
        // wide verbatim copy over a drafter round when one exists.
        if committed.len() > indexed {
            ngram_index.extend(&committed[indexed..]);
            indexed = committed.len();
        }
        let copy_draft = if use_pld {
            ngram_index.propose(8).filter(|d| d.len() >= width)
        } else {
            None
        };
        // Tree round: verify [anchor, a_1..a_tw, b_1..b_tw] as one flattened
        // chunk (tw = 3 keeps 1 + 2*tw = 7 within the flat m<=8 tensor tile)
        // and commit the longer-accepted branch. Fires only when the
        // runner-up root is live (distinct token).
        let mut pre_drafts: Option<Vec<u32>> = None;
        if use_tree && copy_draft.is_none() {
            const TW: usize = 3;
            let tp = Instant::now();
            let p = drafter.propose_branching(anchor, offset, width)?;
            propose_s += tp.elapsed().as_secs_f64();
            if p.alt_tokens.len() >= TW && p.tokens.len() >= TW && p.alt_tokens[0] != p.tokens[0]
            {
                let a = &p.tokens[..TW];
                let b = &p.alt_tokens[..TW];
                let snapshot = model.snapshot_decode_state();
                let mut flat = Vec::with_capacity(1 + 2 * TW);
                flat.push(anchor);
                flat.extend_from_slice(a);
                flat.extend_from_slice(b);
                let flat_input = Tensor::from_slice(&flat, (1, flat.len()), device)?;
                model.set_device_capture(Some(layers.clone()));
                let tv = Instant::now();
                let logits = model.forward_tree_all_logits(&flat_input, offset, TW)?;
                let targets = logits
                    .argmax(D::Minus1)?
                    .to_dtype(DType::U32)?
                    .flatten_all()?
                    .to_vec1::<u32>()?;
                verify_s += tv.elapsed().as_secs_f64();
                drop(logits);
                let caps = model.take_device_captures();
                let ctx_feat = Tensor::cat(&caps, D::Minus1)?;
                drop(caps);

                let main_accepted = a
                    .iter()
                    .zip(targets[..TW].iter())
                    .take_while(|(d, t)| d == t)
                    .count();
                let alt_accepted = if targets[0] == b[0] {
                    1 + b[1..]
                        .iter()
                        .zip(targets[TW + 1..].iter())
                        .take_while(|(d, t)| d == t)
                        .count()
                } else {
                    0
                };
                let on_alt = alt_accepted > main_accepted;
                let accepted = main_accepted.max(alt_accepted);
                let winner: &[u32] = if on_alt { b } else { a };
                let bonus_row = if on_alt { TW + alt_accepted } else { main_accepted };
                let bonus = targets[bonus_row];

                let ctx_rows = if on_alt {
                    Tensor::cat(
                        &[ctx_feat.narrow(1, 0, 1)?, ctx_feat.narrow(1, TW + 1, accepted)?],
                        1,
                    )?
                    .contiguous()?
                } else {
                    ctx_feat.narrow(1, 0, accepted + 1)?.contiguous()?
                };
                drop(ctx_feat);
                drafter.append_context(&ctx_rows, offset)?;
                drop(ctx_rows);

                let tr = Instant::now();
                model.rollback_tree(&snapshot, TW, on_alt, accepted)?;
                device.synchronize()?;
                rollback_s += tr.elapsed().as_secs_f64();

                offset += accepted + 1;
                committed.extend_from_slice(&winner[..accepted]);
                committed.push(bonus);
                committed_raw += accepted + 1;
                accepted_total += accepted;
                rounds += 1;
                tree_rounds += 1;
                round_wall_ms.push(round_t.elapsed().as_secs_f64() * 1000.0);
                round_accepted.push(accepted);
                if on_alt {
                    alt_wins += 1;
                }
                anchor = bonus;
                if committed[committed.len() - (accepted + 1)..]
                    .iter()
                    .any(|&t| t == eos)
                {
                    if let Some(pos) = committed.iter().position(|&t| t == eos) {
                        committed.truncate(pos + 1);
                    }
                    break;
                }
                continue;
            }
            // Alt root dead: fall through to a plain round with the already-
            // proposed main chain.
            pre_drafts = Some(p.tokens);
        }
        let (drafts, used_pld, conf_opt): (Vec<u32>, bool, Option<Vec<f32>>) =
            match (copy_draft, pre_drafts) {
                (Some(d), _) => {
                    pld_rounds += 1;
                    (d, true, None)
                }
                (None, Some(d)) => (d, false, None),
                (None, None) => {
                    let tp = Instant::now();
                    let proposal = if dbg && rounds == 0 {
                        drafter.propose_with_diagnostics(anchor, offset, width)?
                    } else {
                        drafter.propose(anchor, offset, width)?
                    };
                    propose_s += tp.elapsed().as_secs_f64();
                    // Keep confidences whenever skip-low-conf or oracle log needs them.
                    let conf = if oracle_log || skip_low_conf.is_some() {
                        Some(proposal.confidence_logits.clone())
                    } else {
                        None
                    };
                    (proposal.tokens, false, conf)
                }
            };
        let round_width = drafts.len();

        // Conf-gated verify skip (program P3.1 rescope): when mean drafter
        // confidence is below threshold, the offline oracle shows the round is
        // often a total reject under flat mm2d — pay a plain 1-token step
        // (~69 ms) instead of verify (~218 ms). Tree/PLD rounds never skip.
        if let (Some(thr), Some(conf)) = (skip_low_conf, conf_opt.as_ref()) {
            if !used_pld && !conf.is_empty() {
                let mean_c = conf.iter().sum::<f32>() / conf.len() as f32;
                if mean_c < thr {
                    // Target state is still the pre-round snapshot (propose is
                    // drafter-only). Feed the pending anchor, take greedy next.
                    model.set_device_capture(Some(layers.clone()));
                    let step = Tensor::from_slice(&[anchor], (1, 1), device)?;
                    let tv = Instant::now();
                    let logits = model.forward_all_logits(&step, offset)?;
                    let next = argmax_row(&logits)?;
                    verify_s += tv.elapsed().as_secs_f64();
                    drop(logits);
                    let caps = model.take_device_captures();
                    let ctx_feat = Tensor::cat(&caps, D::Minus1)?;
                    drop(caps);
                    drafter.append_context(&ctx_feat, offset)?;
                    drop(ctx_feat);
                    offset += 1;
                    committed.push(next);
                    committed_raw += 1;
                    // Count as a zero-accept drafted round for stats.
                    accepted_total += 0;
                    rounds += 1;
                    skip_rounds += 1;
                    skip_tokens += 1;
                    anchor = next;
                    device.synchronize()?;
                    round_wall_ms.push(round_t.elapsed().as_secs_f64() * 1000.0);
                    round_accepted.push(0);
                    if oracle_log {
                        oracle_rounds.push(serde_json::json!({
                            "round": rounds - 1,
                            "conf": conf,
                            "exact_mask": [],
                            "exact_prefix": 0,
                            "accepted": 0,
                            "skipped_low_conf": true,
                            "mean_conf": mean_c,
                            "wall_ms": round_wall_ms.last().copied(),
                        }));
                    }
                    if dbg {
                        eprintln!(
                            "round {rounds}: SKIP low_conf mean={mean_c:.3} < {thr} -> greedy {next}"
                        );
                    }
                    if next == eos {
                        break;
                    }
                    continue;
                }
            }
        }

        if dbg {
            eprintln!(
                "round {}: proposed {drafts:?} (pld={used_pld}), verify...",
                rounds + 1
            );
        }
        let mut chunk = Vec::with_capacity(round_width + 1);
        chunk.push(anchor);
        chunk.extend_from_slice(&drafts);
        let chunk_input = Tensor::from_slice(&chunk, (1, chunk.len()), device)?;

        // When accept-probing, capture checkpoint layers in addition to drafter taps
        // so mid-stack hiddens are available for offline AUC without a second forward.
        if accept_probe {
            let mut caps = layers.clone();
            for &l in &PROBE_LAYERS {
                if !caps.contains(&l) {
                    caps.push(l);
                }
            }
            caps.sort_unstable();
            caps.dedup();
            model.set_device_capture(Some(caps));
        } else {
            model.set_device_capture(Some(layers.clone()));
        }
        let tv = Instant::now();
        let logits = model.forward_all_logits(&chunk_input, offset)?;
        let targets = logits
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        // Acceptance rule (port of the MiniCPM loop's --accept-margin):
        // exact argmax match (lossless greedy), or typical acceptance — the
        // draft survives while its target logit is within `margin` of the
        // top logit. Computed before the logits drop below.
        let accepted = match accept_margin {
            None => drafts
                .iter()
                .zip(targets.iter())
                .take_while(|(d, t)| d == t)
                .count(),
            Some(margin) if round_width > 0 => {
                let verify_logits = logits.narrow(1, 0, round_width)?;
                let max_vals = verify_logits
                    .max(D::Minus1)?
                    .to_dtype(DType::F32)?
                    .squeeze(0)?
                    .to_vec1::<f32>()?;
                let idx =
                    Tensor::from_slice(&drafts[..round_width], (1, round_width, 1), device)?;
                let draft_vals = verify_logits
                    .gather(&idx, D::Minus1)?
                    .to_dtype(DType::F32)?
                    .squeeze(2)?
                    .squeeze(0)?
                    .to_vec1::<f32>()?;
                // Peakedness gate: top logprob per row = top - logsumexp(row).
                // Only computed when the gate is on (one exp+sum over the
                // verify rows, ~µs against a 190 ms verify).
                let top_lps: Option<Vec<f32>> = match margin_peak {
                    None => None,
                    Some(_) => {
                        let x = verify_logits.to_dtype(DType::F32)?;
                        let mx = x.max_keepdim(D::Minus1)?;
                        let lse = x
                            .broadcast_sub(&mx)?
                            .exp()?
                            .sum(D::Minus1)?
                            .log()?
                            .squeeze(0)?
                            .to_vec1::<f32>()?;
                        // top_lp = max - (max + log sum exp(x - max)) = -log-sum term
                        Some(lse.iter().map(|s| -s).collect())
                    }
                };
                if let (Some(peak), Some(lps)) = (margin_peak, &top_lps) {
                    flat_rows += lps.iter().filter(|lp| **lp < -peak).count();
                }
                (0..round_width)
                    .take_while(|&i| {
                        let within = draft_vals[i] >= max_vals[i] - margin;
                        match (margin_peak, &top_lps) {
                            // Flat row: the margin does not bound logprob
                            // there — require the exact argmax instead.
                            (Some(peak), Some(lps)) if lps[i] < -peak => {
                                drafts[i] == targets[i]
                            }
                            _ => within,
                        }
                    })
                    .count()
            }
            Some(_) => 0,
        };
        verify_s += tv.elapsed().as_secs_f64();
        // Offline scheduler oracle: per-position exact-match mask + confidences.
        // Exact mask is the lossless accept length at each prefix; independent
        // of margin so EV of width truncation is comparable across modes.
        if oracle_log {
            let exact_mask: Vec<bool> = drafts
                .iter()
                .zip(targets.iter())
                .map(|(d, t)| d == t)
                .collect();
            let exact_prefix = exact_mask.iter().take_while(|&&m| m).count();
            oracle_rounds.push(serde_json::json!({
                "round": rounds,
                "conf": conf_opt,
                "exact_mask": exact_mask,
                "exact_prefix": exact_prefix,
                "accepted": accepted,
                "wall_ms": round_t.elapsed().as_secs_f64() * 1000.0,
            }));
        }
        drop(logits); // free the [1, chunk, 248k] verify logits before any re-forward
        let caps = model.take_device_captures();
        // Split captures: drafter taps (original order) vs accept-probe checkpoints.
        let (ctx_feat, probe_by_layer) = if accept_probe {
            let mut all_layers = layers.clone();
            for &l in &PROBE_LAYERS {
                if !all_layers.contains(&l) {
                    all_layers.push(l);
                }
            }
            all_layers.sort_unstable();
            all_layers.dedup();
            if caps.len() != all_layers.len() {
                anyhow::bail!(
                    "accept-probe: expected {} captures, got {}",
                    all_layers.len(),
                    caps.len()
                );
            }
            let mut by_layer: std::collections::HashMap<usize, Tensor> =
                std::collections::HashMap::new();
            for (l, c) in all_layers.iter().zip(caps.into_iter()) {
                by_layer.insert(*l, c);
            }
            let tap_caps: Vec<Tensor> = layers
                .iter()
                .map(|l| {
                    by_layer
                        .get(l)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("missing tap capture layer {l}"))
                })
                .collect::<Result<Vec<_>>>()?;
            let feat = Tensor::cat(&tap_caps, D::Minus1)?;
            drop(tap_caps);
            (feat, Some(by_layer))
        } else {
            (Tensor::cat(&caps, D::Minus1)?, None)
        };

        // Accept-probe rows: for each draft position, checkpoint hidden RMS +
        // exact-match labels (prefix-valid). Kill bar for P5: best single-feature
        // or logistic AUC >= 0.85 on held-out rounds.
        if let Some(by_layer) = probe_by_layer.as_ref() {
            let exact_mask: Vec<bool> = drafts
                .iter()
                .zip(targets.iter())
                .map(|(d, t)| d == t)
                .collect();
            // chunk layout: pos 0 = anchor, pos 1.. = drafts
            for (di, &ok) in exact_mask.iter().enumerate() {
                let prefix_ok = exact_mask.iter().take(di).all(|&m| m);
                let pos = di + 1; // row in capture tensors
                let mut feats = serde_json::Map::new();
                for &l in &PROBE_LAYERS {
                    let Some(h) = by_layer.get(&l) else { continue };
                    // h: [1, seq, hidden] -> row pos -> rms = sqrt(mean sq)
                    let row = h.narrow(1, pos, 1)?.squeeze(0)?.squeeze(0)?; // [hidden]
                    let row_f = row.to_dtype(DType::F32)?;
                    let ms = row_f.sqr()?.mean_all()?.to_scalar::<f32>()?;
                    let rms = ms.sqrt();
                    let mean = row_f.mean_all()?.to_scalar::<f32>()?;
                    let maxabs = row_f.abs()?.max_all()?.to_scalar::<f32>()?;
                    feats.insert(format!("L{l}_rms"), serde_json::json!(rms));
                    feats.insert(format!("L{l}_mean"), serde_json::json!(mean));
                    feats.insert(format!("L{l}_maxabs"), serde_json::json!(maxabs));
                }
                if let Some(conf) = conf_opt.as_ref() {
                    if let Some(c) = conf.get(di) {
                        feats.insert("conf".into(), serde_json::json!(c));
                    }
                }
                feats.insert("pos".into(), serde_json::json!(di));
                feats.insert("exact".into(), serde_json::json!(ok));
                feats.insert("prefix_ok".into(), serde_json::json!(prefix_ok));
                // Useful label for early-exit: "will this position still be on the accepted prefix?"
                feats.insert(
                    "on_accept_prefix".into(),
                    serde_json::json!(prefix_ok && ok),
                );
                feats.insert("round".into(), serde_json::json!(rounds));
                accept_probe_rows.push(serde_json::Value::Object(feats));
            }
        }

        let bonus = targets[accepted];

        // Extend the drafter context with the committed captures (anchor +
        // accepted drafts); the bonus's true hidden is recomputed next round.
        // These verify captures are valid regardless of the rollback below.
        let committed_feat = ctx_feat.narrow(1, 0, accepted + 1)?.contiguous()?;
        drop(ctx_feat);
        drafter.append_context(&committed_feat, offset)?;
        drop(committed_feat);

        if used_pld {
            pld_accepted += accepted;
        }
        if accepted != round_width {
            let tr = Instant::now();
            if readvance_rollback {
                // Readvance rollback: restore the pre-chunk decode state and
                // re-forward only the committed prefix to rebuild the target
                // KV + mixer state at the commit point. The re-forward's
                // captures duplicate the committed ones already appended.
                model.restore_decode_state(&snapshot)?;
                let readvance = &chunk[..accepted + 1];
                let readvance_input =
                    Tensor::from_slice(readvance, (1, readvance.len()), device)?;
                let _ = model.forward_all_logits(&readvance_input, offset)?;
                let _ = model.take_device_captures();
            } else {
                // Capture rollback: closed-form DeltaNet reconstruction at the
                // commit point + KV truncate — no re-forward. Some verify
                // paths never capture (chunk > 32, sequential fallback) —
                // fall back to a readvance instead of dying mid-decode.
                if let Err(err) = model.rollback_to_prefix(&snapshot, accepted + 1) {
                    eprintln!("warning: capture rollback unavailable, readvancing ({err})");
                    model.restore_decode_state(&snapshot)?;
                    let readvance = &chunk[..accepted + 1];
                    let readvance_input =
                        Tensor::from_slice(readvance, (1, readvance.len()), device)?;
                    let _ = model.forward_all_logits(&readvance_input, offset)?;
                    let _ = model.take_device_captures();
                }
            }
            device.synchronize()?;
            rollback_s += tr.elapsed().as_secs_f64();
        }
        offset += accepted + 1;

        committed.extend_from_slice(&drafts[..accepted]);
        committed.push(bonus);
        committed_raw += accepted + 1;
        accepted_total += accepted;
        rounds += 1;
        anchor = bonus;
        device.synchronize()?;
        round_wall_ms.push(round_t.elapsed().as_secs_f64() * 1000.0);
        round_accepted.push(accepted);
        if skip_after_reject && accepted == 0 {
            skip_next = true;
        }
        if dbg {
            eprintln!(
                "round {rounds}: offset={offset} accepted={accepted}/{width} committed={}",
                committed.len()
            );
        }

        if committed[committed.len() - (accepted + 1)..]
            .iter()
            .any(|&t| t == eos)
        {
            if let Some(p) = committed.iter().position(|&t| t == eos) {
                committed.truncate(p + 1);
            }
            break;
        }
    }
    let decode_seconds = decode.elapsed().as_secs_f64();
    let eos_reached = committed.last() == Some(&eos);
    committed.truncate(max_new_tokens);
    // Tokens committed by in-window verify work but dropped by EOS/cap
    // truncation: their GPU cost stays in decode_seconds. Surfaced so the
    // (anti-spec) bias is visible instead of silently absorbed.
    let discarded_tokens = committed_raw - committed.len();
    let text = tok
        .decode(&committed, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;

    println!("--- SPEC GENERATED ({} tokens) ---\n{text}\n---", committed.len());
    println!(
        "{}",
        serde_json::json!({
            "drafter_load_seconds": drafter_load_s,
            "generated_tokens": committed.len(),
            "ids": committed.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","),
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": committed.len() as f64 / decode_seconds.max(f64::EPSILON),
            "rounds": rounds,
            "mean_accepted_per_round": accepted_total as f64 / rounds.max(1) as f64,
            "draft_width": width,
            "propose_seconds": propose_s,
            "verify_seconds": verify_s,
            "rollback_seconds": rollback_s,
            "accept_margin": accept_margin,
            "margin_peak": margin_peak,
            "flat_rows": flat_rows,
            "pld_rounds": pld_rounds,
            "pld_accepted": pld_accepted,
            "tree_rounds": tree_rounds,
            "alt_wins": alt_wins,
            "skip_low_conf": skip_low_conf,
            "skip_after_reject": skip_after_reject,
            "skip_rounds": skip_rounds,
            "skip_tokens": skip_tokens,
            "eos_reached": eos_reached,
            "discarded_tokens": discarded_tokens,
            "round_wall_ms": round_wall_ms,
            "round_accepted": round_accepted,
            "overhead_seconds": decode_seconds - propose_s - verify_s - rollback_s,
            "oracle_rounds": if oracle_log { serde_json::Value::Array(oracle_rounds) } else { serde_json::Value::Null },
            "accept_probe_rows": if accept_probe {
                serde_json::Value::Array(accept_probe_rows)
            } else {
                serde_json::Value::Null
            },
            "provenance": super::run_bench::report_provenance_json(),
        })
    );
    Ok(())
}

pub(crate) fn gguf(args: GgufArgs) -> Result<()> {
    let device = Device::new_metal(0)?;
    match args.cmd {
        GgufCmd::Score(a) => {
            let gen: Vec<u32> = a
                .ids
                .split(',')
                .map(|s| s.trim().parse::<u32>())
                .collect::<Result<_, _>>()
                .map_err(|e| anyhow::anyhow!("bad --ids: {e}"))?;
            let (mut model, prompt_ids, _eos, _ctx, _tok, _load) =
                load_gguf_model(&a.model, &device, None)?;
            score(&mut model, &device, &prompt_ids, &gen)
        }
        GgufCmd::Requant(a) => requant(&a),
        GgufCmd::Pow2Scales(a) => pow2_scales(&a),
        GgufCmd::Repack(a) => repack(&device, &a),
        GgufCmd::BenchGemv => bench_gemv(&device),
        GgufCmd::BenchShapes => bench_shapes(&device),
        GgufCmd::ProfileKernel(a) => profile_kernel(&device, &a.which, a.iters, a.m),
        GgufCmd::BenchDeltanet(a) => bench_deltanet(&device, &a.which, a.l, a.iters),
        GgufCmd::Decode(a) => {
            let (mut model, ids, eos, _ctx, tok, load_seconds) =
                load_gguf_model(&a.model, &device, None)?;
            decode(&mut model, &device, &ids, eos, a.max_new_tokens, &tok, load_seconds, a.warmup)
        }
        GgufCmd::Spec(a) => {
            let spec_run = lmbrrr::runtime_config::SpecRunConfig::from_env();
            // Effective acceptance: --exact / --accept-margin 0 => lossless
            // byte-match (None path); explicit margin honored; --fast => 3.0;
            // default => 1.0 (quality-free). Precedence: exact > margin > fast.
            let accept_margin = if a.exact {
                None
            } else if let Some(m) = a.accept_margin {
                (m != 0.0).then_some(m)
            } else if a.fast {
                Some(3.0)
            } else {
                Some(1.0)
            };
            // spec defaults to the planar mm2d operating point unless --no-mm2d.
            let mm2d_default =
                (!a.no_mm2d).then(lmbrrr::mm2d::Mm2dConfig::spec_planar_default);
            let (mut model, ids, eos, ctx, tok, _load_seconds) =
                load_gguf_model(&a.model, &device, mm2d_default)?;
            spec_decode(
                &mut model,
                &a.drafter,
                &device,
                &ids,
                eos,
                a.max_new_tokens,
                &ctx,
                &tok,
                spec_run.readvance_rollback,
                accept_margin,
                a.margin_peak,
                a.pld,
                a.tree,
                a.skip_low_conf,
                a.skip_after_reject,
            )
        }
        GgufCmd::Profile(a) => {
            let (mut model, ids, _eos, _ctx, _tok, _load_seconds) =
                load_gguf_model(&a.model, &device, None)?;
            profile_decode(&mut model, &device, &ids, a.verify_width, a.capture_taps.as_deref())
        }
    }
}

/// Open the GGUF, build the Qwen35 target, and encode the prompt (ChatML unless
/// `--raw`). Returns the model, prompt ids, eos id, the `ModelCtx` (needed by
/// spec decode), the tokenizer, and the model-load time.
fn load_gguf_model(
    args: &ModelArgs,
    device: &Device,
    mm2d_default: Option<lmbrrr::mm2d::Mm2dConfig>,
) -> Result<(Qwen35CausalLM, Vec<u32>, u32, ModelCtx, Tokenizer, f64)> {
    let dtype = DType::BF16;
    let load = Instant::now();
    let gguf = GgufFile::open(&args.gguf)?;
    let cfg = gguf.config()?;
    // Entrypoint env resolution (LMBRRR_MM2D etc.) — the target's ctx. The
    // drafter keeps its own default ctx (see spec_decode).
    let ctx = {
        let mut m = lmbrrr::runtime_config::RuntimeConfig::from_env().model;
        // Command-scope mm2d default (spec passes the planar operating point):
        // applied ONLY when the user has not explicitly set LMBRRR_MM2D — env
        // always wins over the command default. Planes build below at load.
        if let Some(base) = mm2d_default {
            if std::env::var(lmbrrr::env_keys::MM2D).is_err() {
                m.mm2d = std::sync::Arc::new(base);
            }
        }
        m
    };
    let tok = gguf
        .tokenizer()
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let mut model = {
        let src = gguf.source(&ctx, dtype, device.clone());
        Qwen35CausalLM::new(&cfg, &src, &ctx)?
    };
    model.clear_cache();
    let load_seconds = load.elapsed().as_secs_f64();

    let prompt = if args.raw {
        args.prompt.clone()
    } else {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            args.prompt
        )
    };
    let ids = tok
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    let eos = gguf.eos_token_id().unwrap_or(248046);
    Ok((model, ids, eos, ctx, tok, load_seconds))
}

/// Greedy decode with a per-token sync; prints the generated text and a JSON
/// throughput summary (steady-state = median inter-token gap after 3 warm-ups).
#[allow(clippy::too_many_arguments)]
fn decode(
    model: &mut Qwen35CausalLM,
    device: &Device,
    ids: &[u32],
    eos: u32,
    max_new_tokens: usize,
    tok: &Tokenizer,
    load_seconds: f64,
    warmup: bool,
) -> Result<()> {
    // Cold-compile isolation: the process's FIRST forward JIT-compiles every
    // pipeline it touches, so an unwarmed prefill_seconds is TTFT + shader
    // compile, not steady prefill. One untimed warmup forward (+ one decode
    // step, to compile the seq=1 pipelines too) moves that cost out of the
    // measured window; cold_prefill_seconds records what it was.
    let mut cold_prefill_seconds = None;
    if warmup {
        let cold = Instant::now();
        let winput = Tensor::from_slice(ids, (1, ids.len()), device)?;
        let wlogits = model.forward(&winput, 0)?;
        device.synchronize()?;
        cold_prefill_seconds = Some(cold.elapsed().as_secs_f64());
        let wnext = wlogits.argmax(D::Minus1)?.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?[0];
        let wstep = Tensor::from_slice(&[wnext], (1, 1), device)?;
        let _ = model.forward(&wstep, ids.len())?;
        device.synchronize()?;
        model.clear_cache();
    }
    let prefill = Instant::now();
    let input = Tensor::from_slice(ids, (1, ids.len()), device)?;
    let mut logits = model.forward(&input, 0)?;
    device.synchronize()?;
    let prefill_seconds = prefill.elapsed().as_secs_f64();

    let mut offset = ids.len();
    let mut out = Vec::new();
    let mut gaps = Vec::new();
    let mut forwards = 0usize;
    let mut eos_reached = false;
    let decode = Instant::now();
    for _ in 0..max_new_tokens {
        let t0 = Instant::now();
        // The argmax read-back forces execution of the pending forward — no
        // separate synchronize, so the forward + argmax batch into one point.
        let next = logits
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?[0];
        if next == eos {
            eos_reached = true;
            break;
        }
        out.push(next);
        // Never encode a forward whose logits nothing will read: the old tail
        // forward leaked unsynchronized GPU time into decode_seconds (a ~1/N
        // pro-baseline bias) while doing no useful work.
        if out.len() == max_new_tokens {
            break;
        }
        let step = Tensor::from_slice(&[next], (1, 1), device)?;
        logits = model.forward(&step, offset)?;
        forwards += 1;
        offset += 1;
        gaps.push(t0.elapsed().as_secs_f64());
    }
    let decode_seconds = decode.elapsed().as_secs_f64();
    let text = tok
        .decode(&out, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;

    // Steady-state = median inter-token gap after 3 warm-up tokens (shader
    // compile / cache fill land in the first few steps).
    let mut steady: Vec<f64> = gaps.iter().skip(3.min(out.len())).copied().collect();
    steady.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let steady_tps = steady
        .get(steady.len() / 2)
        .filter(|&&m| m > 0.0)
        .map(|&m| 1.0 / m)
        .unwrap_or(0.0);

    println!("--- GENERATED ({} tokens) ---\n{text}\n---", out.len());
    println!(
        "{}",
        serde_json::json!({
            "load_seconds": load_seconds,
            "prompt_tokens": ids.len(),
            "prefill_seconds": prefill_seconds,
            "prefill_tokens_per_second": ids.len() as f64 / prefill_seconds,
            "cold_prefill_seconds": cold_prefill_seconds,
            "generated_tokens": out.len(),
            "ids": out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","),
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": out.len() as f64 / decode_seconds.max(f64::EPSILON),
            // The first token is prefill-produced (its forward is paid in
            // prefill_seconds); decode_forwards is the in-window forward count
            // so offline analysis can separate per-forward cost from the
            // token-count definition.
            "decode_forwards": forwards,
            "eos_reached": eos_reached,
            "steady_state_tokens_per_second": steady_tps,
            "provenance": super::run_bench::report_provenance_json(),
            "device": format!("{device:?}"),
            "dtype": "BF16",
        })
    );
    Ok(())
}
