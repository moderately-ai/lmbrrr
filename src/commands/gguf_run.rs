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
use lmbrrr::qwen35::{CausalTextModel, Qwen35CausalLM};

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
}

pub(crate) fn gguf_run(args: GgufRunArgs) -> Result<()> {
    let device = Device::new_metal(0)?;
    let dtype = DType::BF16;

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
        device.synchronize()?;
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
