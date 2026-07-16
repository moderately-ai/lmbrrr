//! Lean text-decode + throughput harness for a qwen35-hybrid GGUF (the ternary
//! Ternary-Bonsai-27B target). Baseline path: host argmax per token with a
//! per-token sync for honest timing — no fused-argmax / async-readback fast
//! paths yet (those land with the generic decode-loop unification). The number
//! this reports is the floor to iterate up from, measured on the M3 referee.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use candle::{DType, Device, Tensor, D};
use clap::Parser;

use lmbrrr::gguf::GgufFile;
use lmbrrr::model_ctx::ModelCtx;
use lmbrrr::qwen35::{CausalTextModel, Qwen35CausalLM, Qwen35Profiler};

#[derive(Parser, Debug)]
pub(crate) struct GgufRunArgs {
    /// Path to the qwen35-hybrid GGUF (e.g. Ternary-Bonsai-27B-Q2_0.gguf).
    #[arg(long)]
    pub gguf: PathBuf,

    #[arg(long, default_value = "Explain quantum computing in simple terms.")]
    pub prompt: String,

    #[arg(long, default_value_t = 128)]
    pub max_new_tokens: usize,

    /// Feed the prompt verbatim instead of wrapping it in the ChatML template.
    #[arg(long)]
    pub raw: bool,

    /// Micro-bench the decode GEMV (Q2_0 vs Q4K) on the ffn shape instead of
    /// decoding — isolates kernel bandwidth from model overhead.
    #[arg(long)]
    pub bench_gemv: bool,

    /// Attach the per-op profiler and dump the decode-step breakdown by
    /// component (deltanet_recurrent_rule / mlp / attention / norms / ...).
    #[arg(long)]
    pub profile: bool,

    /// Enable DSpark speculative decoding with the given drafter GGUF
    /// (Ternary-Bonsai-27B-dspark-Q4_1.gguf). Draft width defaults to the
    /// drafter's block_size.
    #[arg(long)]
    pub spec_drafter: Option<PathBuf>,
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
    Ok(())
}

fn profile_decode(model: &mut Qwen35CausalLM, device: &Device, ids: &[u32]) -> Result<()> {
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

    // Aggregate decode-step events (seq_len == 1) by component.
    let mut agg: HashMap<String, (f64, usize)> = HashMap::new();
    for e in prof.events() {
        if e.seq_len != 1 {
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
        "profiled {steps} decode steps; summed op-time {:.1} ms ({:.2} ms/token)",
        total * 1000.0,
        total * 1000.0 / steps as f64
    );
    println!("{:<40} {:>10} {:>8} {:>6}", "component", "ms/token", "calls", "pct");
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
) -> Result<()> {
    use lmbrrr::dspark::DsparkDrafter;
    let load = Instant::now();
    let dgguf = GgufFile::open(drafter_path)?;
    let mut drafter = DsparkDrafter::load_gguf(&dgguf, device, DType::BF16, ctx, false)?;
    let layers = drafter.config.target_layer_ids.clone();
    let width = drafter.config.block_size;
    let drafter_load_s = load.elapsed().as_secs_f64();

    // Prefill with tap-layer capture, seed the drafter context.
    model.clear_cache();
    drafter.clear_context();
    model.set_device_capture(Some(layers.clone()));
    let input = Tensor::from_slice(ids, (1, ids.len()), device)?;
    let logits = model.forward_all_logits(&input, 0)?;
    device.synchronize()?;
    let caps = model.take_device_captures();
    let ctx_feat = Tensor::cat(&caps, D::Minus1)?;
    drafter.append_context(&ctx_feat, 0)?;

    let mut anchor = argmax_row(&logits.narrow(1, ids.len() - 1, 1)?)?;
    let mut offset = ids.len();
    let mut committed: Vec<u32> = Vec::new();
    let mut rounds = 0usize;
    let mut accepted_total = 0usize;

    let decode = Instant::now();
    while committed.len() < max_new_tokens {
        let snapshot = model.snapshot_decode_state();
        let drafts = drafter.propose(anchor, offset, width)?.tokens;
        let mut chunk = Vec::with_capacity(width + 1);
        chunk.push(anchor);
        chunk.extend_from_slice(&drafts);
        let chunk_input = Tensor::from_slice(&chunk, (1, chunk.len()), device)?;

        model.set_device_capture(Some(layers.clone()));
        let logits = model.forward_all_logits(&chunk_input, offset)?;
        let targets = logits
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let caps = model.take_device_captures();
        let ctx_feat = Tensor::cat(&caps, D::Minus1)?;

        let accepted = drafts
            .iter()
            .zip(targets.iter())
            .take_while(|(d, t)| d == t)
            .count();
        let bonus = targets[accepted];
        if accepted != width {
            model.rollback_to_prefix(&snapshot, accepted + 1)?;
            device.synchronize()?;
        }
        // Extend the drafter context with the committed captures (anchor +
        // accepted drafts); the bonus's true hidden is recomputed next round.
        let committed_feat = ctx_feat.narrow(1, 0, accepted + 1)?.contiguous()?;
        drafter.append_context(&committed_feat, offset)?;
        offset += accepted + 1;

        committed.extend_from_slice(&drafts[..accepted]);
        committed.push(bonus);
        accepted_total += accepted;
        rounds += 1;
        anchor = bonus;

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
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": committed.len() as f64 / decode_seconds.max(f64::EPSILON),
            "rounds": rounds,
            "mean_accepted_per_round": accepted_total as f64 / rounds.max(1) as f64,
            "draft_width": width,
        })
    );
    Ok(())
}

pub(crate) fn gguf_run(args: GgufRunArgs) -> Result<()> {
    let device = Device::new_metal(0)?;
    let dtype = DType::BF16;

    if args.bench_gemv {
        return bench_gemv(&device);
    }

    let load = Instant::now();
    let gguf = GgufFile::open(&args.gguf)?;
    let cfg = gguf.config()?;
    let ctx = ModelCtx::default();
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

    if args.profile {
        return profile_decode(&mut model, &device, &ids);
    }

    if let Some(drafter_path) = args.spec_drafter.as_ref() {
        return spec_decode(
            &mut model,
            drafter_path,
            &device,
            &ids,
            eos,
            args.max_new_tokens,
            &ctx,
            &tok,
        );
    }

    let prefill = Instant::now();
    let input = Tensor::from_slice(&ids, (1, ids.len()), &device)?;
    let mut logits = model.forward(&input, 0)?;
    device.synchronize()?;
    let prefill_seconds = prefill.elapsed().as_secs_f64();

    let mut offset = ids.len();
    let mut out = Vec::new();
    let mut gaps = Vec::new();
    let decode = Instant::now();
    for _ in 0..args.max_new_tokens {
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
        let step = Tensor::from_slice(&[next], (1, 1), &device)?;
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
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": out.len() as f64 / decode_seconds.max(f64::EPSILON),
            "steady_state_tokens_per_second": steady_tps,
            "device": format!("{device:?}"),
            "dtype": "BF16",
        })
    );
    Ok(())
}
