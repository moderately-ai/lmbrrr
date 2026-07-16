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
}

#[derive(Args, Debug)]
struct SpecArgs {
    #[command(flatten)]
    model: ModelArgs,

    /// Drafter GGUF (Ternary-Bonsai-27B-dspark-Q4_1.gguf). The draft width is
    /// the drafter's block_size.
    #[arg(long)]
    drafter: PathBuf,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    /// Typical acceptance: a draft survives while its target logit is within
    /// this margin of the top logit (port of the MiniCPM loop's flag).
    /// Committed tokens remain the drafts, so output may legitimately differ
    /// from greedy. Unset = exact argmax match (lossless).
    #[arg(long)]
    accept_margin: Option<f32>,

    /// Prompt-lookup drafting. Default OFF: measured a net LOSS against this
    /// drafter on both prose (18.3 -> 15.2 tok/s, 0 copy tokens accepted) and
    /// code (20.0 -> 16.1 — copy rounds preempt stronger drafter rounds),
    /// reproducing the MiniCPM campaign's ungated-PLD lesson.
    #[arg(long)]
    pld: bool,
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
        for _ in 0..iters {
            let _ = lin.forward(&x)?;
        }
        device.synchronize()?;
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

fn profile_decode(
    model: &mut Qwen35CausalLM,
    device: &Device,
    ids: &[u32],
    width: usize,
) -> Result<()> {
    use candle::D;
    use std::collections::HashMap;

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
        for _ in 0..steps {
            let chunk_input = Tensor::from_slice(&chunk, (1, width), device)?;
            let logits = model.forward_all_logits(&chunk_input, offset)?;
            // Include the verify's argmax readback in the step, as spec does.
            let _ = logits
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            device.synchronize()?;
            offset += width;
        }
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
    use_pld: bool,
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

    let dbg = std::env::var("LMBRRR_SPEC_DEBUG").is_ok();

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
    let mut offset = ids.len();
    let mut committed: Vec<u32> = vec![anchor];
    let mut rounds = 0usize;
    let mut accepted_total = 0usize;
    let mut propose_s = 0.0f64;
    let mut verify_s = 0.0f64;
    let mut rollback_s = 0.0f64;

    let decode = Instant::now();
    while committed.len() < max_new_tokens && committed.last() != Some(&eos) {
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
        let (drafts, used_pld) = match copy_draft {
            Some(d) => {
                pld_rounds += 1;
                (d, true)
            }
            None => {
                let tp = Instant::now();
                let proposal = if dbg && rounds == 0 {
                    drafter.propose_with_diagnostics(anchor, offset, width)?
                } else {
                    drafter.propose(anchor, offset, width)?
                };
                propose_s += tp.elapsed().as_secs_f64();
                let p = if dbg { proposal.tokens.clone() } else { proposal.tokens };
                (p, false)
            }
        };
        let round_width = drafts.len();
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

        model.set_device_capture(Some(layers.clone()));
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
                draft_vals
                    .iter()
                    .zip(max_vals.iter())
                    .take_while(|(draft, top)| **draft >= **top - margin)
                    .count()
            }
            Some(_) => 0,
        };
        verify_s += tv.elapsed().as_secs_f64();
        drop(logits); // free the [1, chunk, 248k] verify logits before any re-forward
        let caps = model.take_device_captures();
        let ctx_feat = Tensor::cat(&caps, D::Minus1)?;
        drop(caps);

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
        accepted_total += accepted;
        rounds += 1;
        anchor = bonus;
        device.synchronize()?;
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
    committed.truncate(max_new_tokens);
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
            "pld_rounds": pld_rounds,
            "pld_accepted": pld_accepted,
            "overhead_seconds": decode_seconds - propose_s - verify_s,
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
                load_gguf_model(&a.model, &device)?;
            score(&mut model, &device, &prompt_ids, &gen)
        }
        GgufCmd::Requant(a) => requant(&a),
        GgufCmd::Repack(a) => repack(&device, &a),
        GgufCmd::BenchGemv => bench_gemv(&device),
        GgufCmd::BenchShapes => bench_shapes(&device),
        GgufCmd::ProfileKernel(a) => profile_kernel(&device, &a.which, a.iters, a.m),
        GgufCmd::Decode(a) => {
            let (mut model, ids, eos, _ctx, tok, load_seconds) = load_gguf_model(&a.model, &device)?;
            decode(&mut model, &device, &ids, eos, a.max_new_tokens, &tok, load_seconds)
        }
        GgufCmd::Spec(a) => {
            let spec_run = lmbrrr::runtime_config::SpecRunConfig::from_env();
            let (mut model, ids, eos, ctx, tok, _load_seconds) = load_gguf_model(&a.model, &device)?;
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
                a.accept_margin,
                a.pld,
            )
        }
        GgufCmd::Profile(a) => {
            let (mut model, ids, _eos, _ctx, _tok, _load_seconds) =
                load_gguf_model(&a.model, &device)?;
            profile_decode(&mut model, &device, &ids, a.verify_width)
        }
    }
}

/// Open the GGUF, build the Qwen35 target, and encode the prompt (ChatML unless
/// `--raw`). Returns the model, prompt ids, eos id, the `ModelCtx` (needed by
/// spec decode), the tokenizer, and the model-load time.
fn load_gguf_model(
    args: &ModelArgs,
    device: &Device,
) -> Result<(Qwen35CausalLM, Vec<u32>, u32, ModelCtx, Tokenizer, f64)> {
    let dtype = DType::BF16;
    let load = Instant::now();
    let gguf = GgufFile::open(&args.gguf)?;
    let cfg = gguf.config()?;
    // Entrypoint env resolution (LMBRRR_MM2D etc.) — the target's ctx. The
    // drafter keeps its own default ctx (see spec_decode).
    let ctx = lmbrrr::runtime_config::RuntimeConfig::from_env().model;
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
fn decode(
    model: &mut Qwen35CausalLM,
    device: &Device,
    ids: &[u32],
    eos: u32,
    max_new_tokens: usize,
    tok: &Tokenizer,
    load_seconds: f64,
) -> Result<()> {
    let prefill = Instant::now();
    let input = Tensor::from_slice(ids, (1, ids.len()), device)?;
    let mut logits = model.forward(&input, 0)?;
    device.synchronize()?;
    let prefill_seconds = prefill.elapsed().as_secs_f64();

    let mut offset = ids.len();
    let mut out = Vec::new();
    let mut gaps = Vec::new();
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
            break;
        }
        out.push(next);
        let step = Tensor::from_slice(&[next], (1, 1), device)?;
        logits = model.forward(&step, offset)?;
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
            "generated_tokens": out.len(),
            "ids": out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","),
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": out.len() as f64 / decode_seconds.max(f64::EPSILON),
            "steady_state_tokens_per_second": steady_tps,
            "device": format!("{device:?}"),
            "dtype": "BF16",
        })
    );
    Ok(())
}
