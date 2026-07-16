//! GGUF loader for qwen35-hybrid models (prism-ml Ternary-Bonsai-27B and any
//! future external qwen35 GGUF). Owns the metadata→`TextConfig` derivation and
//! `GgufSource`, the packed [`LinearSource`] the shared qwen35 constructors
//! build from. See tickets/gguf-loader-qwen35-hybrid.md for the verified name
//! map and the ground-truth GGUF keys.

use std::cell::RefCell;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use candle::quantized::{ggml_file::qtensor_from_ggml, gguf_file, QTensor};
use candle::{DType, Device, Result, Shape, Tensor};
use candle_nn::Activation;

use crate::config::{LayerType, RopeParameters, TextConfig};
use crate::linear_source::{LinearPart, LinearSource};
use crate::model_ctx::ModelCtx;
use crate::quantized_linear::MixedLinear;

/// Opened GGUF: the parsed header + a seekable reader for on-demand tensor
/// loads. Cheap to borrow as many [`GgufSource`] views (one per module prefix).
pub struct GgufFile {
    content: gguf_file::Content,
    reader: RefCell<BufReader<File>>,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path).map_err(candle::Error::wrap)?);
        let content = gguf_file::Content::read(&mut reader)?;
        Ok(Self {
            content,
            reader: RefCell::new(reader),
        })
    }

    pub fn config(&self) -> Result<TextConfig> {
        qwen35_config_from_gguf(&self.content)
    }

    /// Build the gpt2 BPE tokenizer from the embedded `tokenizer.ggml.*` vocab
    /// + merges (byte-level, matching what llama.cpp uses for this GGUF).
    pub fn tokenizer(&self) -> Result<tokenizers::Tokenizer> {
        use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
        use tokenizers::models::bpe::{Vocab, BPE};
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::Tokenizer;

        let err = |m: &str| candle::Error::Msg(m.to_string());
        let tokens = self
            .content
            .metadata
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| err("gguf missing tokenizer.ggml.tokens"))?
            .to_vec()?;
        let vocab: Vocab = tokens
            .iter()
            .enumerate()
            .map(|(i, v)| Ok((v.to_string()?.clone(), i as u32)))
            .collect::<Result<_>>()?;
        let merges = self
            .content
            .metadata
            .get("tokenizer.ggml.merges")
            .ok_or_else(|| err("gguf missing tokenizer.ggml.merges"))?
            .to_vec()?
            .iter()
            .map(|v| {
                let s = v.to_string()?;
                let (a, b) = s
                    .split_once(' ')
                    .ok_or_else(|| err("malformed merge entry"))?;
                Ok((a.to_string(), b.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(|e| candle::Error::Msg(format!("build bpe: {e}")))?;
        let mut tok = Tokenizer::new(bpe);
        // gpt2/qwen byte-level: no prefix space, regex pre-split, byte decoder.
        tok.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
        tok.with_decoder(Some(ByteLevelDecoder::new(true, true, true)));
        Ok(tok)
    }

    /// A source rooted at the model top level (empty module prefix).
    pub fn source<'a>(&'a self, ctx: &'a ModelCtx, dtype: DType, device: Device) -> GgufSource<'a> {
        GgufSource {
            file: self,
            ctx,
            dtype,
            device,
            prefix: String::new(),
        }
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
    }

    pub(crate) fn read_qtensor(&self, gguf_name: &str, device: &Device) -> Result<QTensor> {
        let mut reader = self.reader.borrow_mut();
        self.content.tensor(&mut *reader, gguf_name, device)
    }

    pub(crate) fn content(&self) -> &gguf_file::Content {
        &self.content
    }
}

pub(crate) fn md_bool(content: &gguf_file::Content, key: &str) -> Result<bool> {
    let v = content
        .metadata
        .get(key)
        .ok_or_else(|| candle::Error::Msg(format!("gguf missing metadata key {key}")))?;
    v.to_bool()
        .or_else(|_| v.to_u32().map(|x| x != 0))
        .map_err(candle::Error::from)
}

pub(crate) fn md_u32(content: &gguf_file::Content, key: &str) -> Result<u32> {
    Ok(md_usize(content, key)? as u32)
}

pub(crate) fn md_u32_array(content: &gguf_file::Content, key: &str) -> Result<Vec<u32>> {
    let v = content
        .metadata
        .get(key)
        .ok_or_else(|| candle::Error::Msg(format!("gguf missing metadata key {key}")))?;
    let items = v
        .to_vec()
        .map_err(|_| candle::Error::Msg(format!("gguf metadata {key} is not an array")))?;
    items
        .iter()
        .map(|it| {
            it.to_u32()
                .or_else(|_| it.to_i32().map(|x| x as u32))
                .map_err(candle::Error::from)
        })
        .collect()
}

pub(crate) fn md_usize(content: &gguf_file::Content, key: &str) -> Result<usize> {
    let v = content
        .metadata
        .get(key)
        .ok_or_else(|| candle::Error::Msg(format!("gguf missing metadata key {key}")))?;
    // Counts are stored as u32 in this file; fall back to the wider int types.
    v.to_u32()
        .map(|x| x as usize)
        .or_else(|_| v.to_u64().map(|x| x as usize))
        .or_else(|_| v.to_i32().map(|x| x as usize))
}

pub(crate) fn md_f64(content: &gguf_file::Content, key: &str) -> Result<f64> {
    let v = content
        .metadata
        .get(key)
        .ok_or_else(|| candle::Error::Msg(format!("gguf missing metadata key {key}")))?;
    Ok(v.to_f32()? as f64)
}

/// Derive a [`TextConfig`] from `qwen35.*` GGUF metadata. Fields with no GGUF
/// source are set explicitly (`hidden_act=silu`, `attention_bias=false`,
/// `tie_word_embeddings=false` — token_embd/output ship as separate tensors).
pub fn qwen35_config_from_gguf(content: &gguf_file::Content) -> Result<TextConfig> {
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().cloned())
        .unwrap_or_default();
    if arch != "qwen35" {
        candle::bail!("gguf architecture {arch:?} is not qwen35");
    }

    let head_dim = md_usize(content, "qwen35.attention.key_length")?;
    let num_hidden_layers = md_usize(content, "qwen35.block_count")?;
    let full_attention_interval = md_usize(content, "qwen35.full_attention_interval")?;
    // Full-attention every `interval`-th layer (idx 3,7,... for interval 4),
    // matching the GGUF tensor split (blk.3 ships attn_q/k/v, blk.0 attn_qkv).
    let layer_types = (0..num_hidden_layers)
        .map(|i| {
            if (i + 1) % full_attention_interval == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();

    let vocab_size = content
        .metadata
        .get("tokenizer.ggml.tokens")
        .ok_or_else(|| candle::Error::Msg("gguf missing tokenizer.ggml.tokens".into()))?
        .to_vec()?
        .len();

    // Text-only decode: mrope with all-text positions is standard rope, so the
    // rotary spans `rope.dimension_count` of `head_dim` at `rope.freq_base`.
    let rotary_dims = md_usize(content, "qwen35.rope.dimension_count")?;
    let rope_parameters = RopeParameters {
        partial_rotary_factor: rotary_dims as f64 / head_dim as f64,
        rope_theta: md_f64(content, "qwen35.rope.freq_base")?,
        rope_type: "default".to_string(),
    };

    Ok(TextConfig {
        vocab_size,
        hidden_size: md_usize(content, "qwen35.embedding_length")?,
        intermediate_size: md_usize(content, "qwen35.feed_forward_length")?,
        num_hidden_layers,
        num_attention_heads: md_usize(content, "qwen35.attention.head_count")?,
        num_key_value_heads: md_usize(content, "qwen35.attention.head_count_kv")?,
        head_dim,
        max_position_embeddings: md_usize(content, "qwen35.context_length")?,
        hidden_act: Activation::Silu,
        rms_norm_eps: md_f64(content, "qwen35.attention.layer_norm_rms_epsilon")?,
        attention_bias: false,
        tie_word_embeddings: false,
        layer_types,
        linear_conv_kernel_dim: md_usize(content, "qwen35.ssm.conv_kernel")?,
        linear_key_head_dim: md_usize(content, "qwen35.ssm.state_size")?,
        linear_value_head_dim: md_usize(content, "qwen35.ssm.state_size")?,
        linear_num_key_heads: md_usize(content, "qwen35.ssm.group_count")?,
        linear_num_value_heads: md_usize(content, "qwen35.ssm.time_step_rank")?,
        rope_parameters,
    })
}

/// Translate an lmbrrr module path (prefix `.` name) to the flat GGUF tensor
/// name. The map is verified against the Bonsai GGUF (851 tensors).
fn to_gguf_name(path: &str) -> Result<String> {
    match path {
        "embed_tokens.weight" => return Ok("token_embd.weight".into()),
        "norm.weight" => return Ok("output_norm.weight".into()),
        "lm_head.weight" => return Ok("output.weight".into()),
        _ => {}
    }
    if let Some(rest) = path.strip_prefix("layers.") {
        let (n, sub) = rest
            .split_once('.')
            .ok_or_else(|| candle::Error::Msg(format!("bad layer path {path}")))?;
        let mapped = match sub {
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "post_attention_norm.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "self_attn.q_norm.weight" => "attn_q_norm.weight",
            "self_attn.k_norm.weight" => "attn_k_norm.weight",
            "linear_attn.in_proj_qkv.weight" => "attn_qkv.weight",
            "linear_attn.in_proj_z.weight" => "attn_gate.weight",
            "linear_attn.in_proj_b.weight" => "ssm_beta.weight",
            "linear_attn.in_proj_a.weight" => "ssm_alpha.weight",
            "linear_attn.out_proj.weight" => "ssm_out.weight",
            "linear_attn.norm.weight" => "ssm_norm.weight",
            "linear_attn.dt_bias" => "ssm_dt.bias",
            "linear_attn.A_log" => "ssm_a",
            "linear_attn.conv1d.weight" => "ssm_conv1d.weight",
            other => candle::bail!("no gguf mapping for layers.*.{other}"),
        };
        return Ok(format!("blk.{n}.{mapped}"));
    }
    candle::bail!("no gguf mapping for {path}")
}

/// Packed weight source over a [`GgufFile`] at a given module prefix. Q2_0
/// linears stay packed via `ctx.quantized_linear`; fused linears concatenate
/// the raw block bytes (exact — requant would corrupt the per-block scale);
/// dense tensors (norms/embed/conv/A_log) dequantize to the model dtype.
pub struct GgufSource<'a> {
    file: &'a GgufFile,
    ctx: &'a ModelCtx,
    dtype: DType,
    device: Device,
    prefix: String,
}

impl GgufSource<'_> {
    fn path(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        }
    }
}

impl LinearSource for GgufSource<'_> {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn tensor(&self, shape: Shape, name: &str) -> Result<Tensor> {
        let gname = to_gguf_name(&self.path(name))?;
        let qt = self.file.read_qtensor(&gname, &self.device)?;
        let mut t = qt.dequantize(&self.device)?;
        if gname.ends_with("ssm_a") {
            // The model loads `A_log` and applies `.exp()`; the GGUF stores
            // `ssm_a = -exp(A_log)`, so invert: A_log = ln(-ssm_a).
            t = t.neg()?.log()?;
        }
        t.reshape(shape)?.to_dtype(self.dtype)
    }

    fn linear(&self, parts: &[LinearPart<'_>], in_dim: usize, bias: bool) -> Result<MixedLinear> {
        if bias {
            candle::bail!("gguf packed linear with bias is unsupported (qwen35 has no attn bias)");
        }
        let qts = parts
            .iter()
            .map(|p| {
                let gname = to_gguf_name(&self.path(p.name))?;
                self.file.read_qtensor(&gname, &self.device)
            })
            .collect::<Result<Vec<_>>>()?;
        if qts.len() == 1 {
            return self.ctx.quantized_linear(qts.into_iter().next().unwrap());
        }
        // Fused: cat the packed block bytes along output rows. Row-concatenation
        // never crosses a quant block (blocks run along the input dim), so this
        // is bit-exact — dequant→cat→requant would recompute per-block scales
        // and corrupt codes.
        let dtype = qts[0].dtype();
        let out_total: usize = parts.iter().map(|p| p.out).sum();
        let mut bytes = Vec::new();
        for qt in &qts {
            if qt.dtype() != dtype {
                candle::bail!("fused gguf linear parts differ in ggml dtype");
            }
            bytes.extend_from_slice(&qt.data()?);
        }
        let fused = qtensor_from_ggml(dtype, &bytes, vec![out_total, in_dim], &self.device)?;
        self.ctx.quantized_linear(fused)
    }

    fn sub(&self, prefix: &str) -> Self {
        GgufSource {
            file: self.file,
            ctx: self.ctx,
            dtype: self.dtype,
            device: self.device.clone(),
            prefix: self.path(prefix),
        }
    }

    fn norms_pre_shifted(&self) -> bool {
        true
    }

    fn embedding_qtensor(&self, name: &str) -> Result<Option<QTensor>> {
        let gname = to_gguf_name(&self.path(name))?;
        Ok(Some(self.file.read_qtensor(&gname, &self.device)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Config-mapper smoke against the real Bonsai GGUF (header-only, so it runs
    // on any candle pin). Skips when the model isn't present on this machine.
    #[test]
    fn bonsai_config_from_gguf() -> Result<()> {
        let path = std::path::PathBuf::from(format!(
            "{}/models/Ternary-Bonsai-27B/Ternary-Bonsai-27B-Q2_0.gguf",
            std::env::var("HOME").unwrap_or_default()
        ));
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            return Ok(());
        }
        let gguf = GgufFile::open(&path)?;
        let cfg = gguf.config()?;
        assert_eq!(cfg.vocab_size, 248320);
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.intermediate_size, 17408);
        assert_eq!(cfg.num_hidden_layers, 64);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.max_position_embeddings, 262144);
        assert_eq!(cfg.linear_conv_kernel_dim, 4);
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert_eq!(cfg.linear_value_head_dim, 128);
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_num_value_heads, 48);
        // key_dim=2048, value_dim=6144 -> conv_dim 10240 (= attn_qkv out rows).
        let value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
        let conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2 + value_dim;
        assert_eq!(value_dim, 6144);
        assert_eq!(conv_dim, 10240);
        assert!((cfg.rope_parameters.rope_theta - 1e7).abs() < 1.0);
        assert!((cfg.rope_parameters.partial_rotary_factor - 0.25).abs() < 1e-9);
        assert_eq!(cfg.num_hidden_layers, cfg.layer_types.len());
        assert!(matches!(cfg.layer_types[3], LayerType::FullAttention));
        assert!(matches!(cfg.layer_types[0], LayerType::LinearAttention));
        Ok(())
    }

    // Load real 27B packed weights through GgufSource and verify the byte-cat
    // fusion: a fused linear must equal its parts' outputs concatenated. Proves
    // the name map, packed byte-cat, quantized_linear, and the Q2_0 kernel all
    // wire together. Needs the Metal kernel (candle pin f4eb38b2+).
    #[test]
    fn bonsai_gguf_source_fused_load() -> Result<()> {
        let path = std::path::PathBuf::from(format!(
            "{}/models/Ternary-Bonsai-27B/Ternary-Bonsai-27B-Q2_0.gguf",
            std::env::var("HOME").unwrap_or_default()
        ));
        if !path.exists() {
            eprintln!("skip: model not present");
            return Ok(());
        }
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skip: no metal device");
                return Ok(());
            }
        };
        let gguf = GgufFile::open(&path)?;
        let ctx = ModelCtx::default();
        let root = gguf.source(&ctx, DType::BF16, device.clone());
        // layer 0 is DeltaNet: in_proj_qkv (10240) + in_proj_z (6144) fuse.
        let src = root.sub("layers").sub("0").sub("linear_attn");
        let hidden = 5120usize;
        let x = Tensor::randn(0f32, 1f32, (1, 1, hidden), &device)?.to_dtype(DType::BF16)?;

        let fused = src.linear(
            &[
                LinearPart::new("in_proj_qkv.weight", 10240),
                LinearPart::new("in_proj_z.weight", 6144),
            ],
            hidden,
            false,
        )?;
        let qkv = src.linear(&[LinearPart::new("in_proj_qkv.weight", 10240)], hidden, false)?;
        let z = src.linear(&[LinearPart::new("in_proj_z.weight", 6144)], hidden, false)?;

        let got = fused.forward(&x)?.to_dtype(DType::F32)?;
        let expected = Tensor::cat(&[qkv.forward(&x)?, z.forward(&x)?], 2)?.to_dtype(DType::F32)?;
        assert_eq!(got.dims(), &[1, 1, 16384]);
        let diff = (got - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        eprintln!("fused-vs-split max abs diff: {diff:.3e}");
        assert!(diff < 1e-4, "byte-cat fusion mismatch: {diff:.3e}");
        Ok(())
    }

    // Build the FULL 27B (all 64 hybrid layers, ~7 GB packed) through
    // GgufSource + Qwen35CausalLM and run a prefill forward — the end-to-end
    // model construction + decode on real ternary weights. Heavy; #[ignore] so
    // it runs on demand (`nextest run -E 'test(bonsai_27b_forward)' --run-ignored all`).
    #[test]
    #[ignore = "loads the full 27B (~7 GB); run on demand"]
    fn bonsai_27b_forward() -> Result<()> {
        use crate::qwen35::{CausalTextModel, Qwen35CausalLM};

        let path = std::path::PathBuf::from(format!(
            "{}/models/Ternary-Bonsai-27B/Ternary-Bonsai-27B-Q2_0.gguf",
            std::env::var("HOME").unwrap_or_default()
        ));
        if !path.exists() {
            eprintln!("skip: model not present");
            return Ok(());
        }
        let device = Device::new_metal(0)?;
        let gguf = GgufFile::open(&path)?;
        let cfg = gguf.config()?;
        let ctx = ModelCtx::default();
        let build = std::time::Instant::now();
        let mut model = {
            let src = gguf.source(&ctx, DType::BF16, device.clone());
            Qwen35CausalLM::new(&cfg, &src, &ctx)?
        };
        eprintln!("built 27B in {:.1}s", build.elapsed().as_secs_f32());

        let tokens = Tensor::from_slice(&[100u32, 200, 300, 400, 500], (1, 5), &device)?;
        let logits = model.forward(&tokens, 0)?;
        assert_eq!(logits.dims(), &[1, cfg.vocab_size]);
        let v = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(v.iter().all(|x| x.is_finite()), "non-finite logits");
        let argmax = logits.argmax(candle::D::Minus1)?.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?[0];
        eprintln!("prefill argmax id = {argmax} (vocab {})", cfg.vocab_size);
        assert!((argmax as usize) < cfg.vocab_size);
        Ok(())
    }

    // Diagnostic: are GGUF norm weights stored as w (~1.0, llama.cpp already
    // resolved the shift) or w-1 (~0.0, raw Qwen)? Decides zero_centered.
    #[test]
    fn bonsai_norm_stats() -> Result<()> {
        let path = std::path::PathBuf::from(format!(
            "{}/models/Ternary-Bonsai-27B/Ternary-Bonsai-27B-Q2_0.gguf",
            std::env::var("HOME").unwrap_or_default()
        ));
        if !path.exists() {
            return Ok(());
        }
        let device = Device::new_metal(0)?;
        let gguf = GgufFile::open(&path)?;
        for name in ["blk.0.attn_norm.weight", "output_norm.weight"] {
            let qt = gguf.read_qtensor(name, &device)?;
            let t = qt.dequantize(&device)?.to_dtype(DType::F32)?.flatten_all()?;
            let v = t.to_vec1::<f32>()?;
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let (mn, mx) = v.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
            eprintln!("{name}: mean={mean:.4} min={mn:.4} max={mx:.4}");
        }
        Ok(())
    }

    // Full E2E: build the 27B + tokenizer from the GGUF, greedy-decode a ChatML
    // prompt, print the generated text. Proves coherent on-device generation.
    #[test]
    #[ignore = "loads the full 27B (~7 GB); run on demand"]
    fn bonsai_27b_generate() -> Result<()> {
        use crate::qwen35::{CausalTextModel, Qwen35CausalLM};
        use candle::D;

        let path = std::path::PathBuf::from(format!(
            "{}/models/Ternary-Bonsai-27B/Ternary-Bonsai-27B-Q2_0.gguf",
            std::env::var("HOME").unwrap_or_default()
        ));
        if !path.exists() {
            eprintln!("skip: model not present");
            return Ok(());
        }
        let device = Device::new_metal(0)?;
        let gguf = GgufFile::open(&path)?;
        let cfg = gguf.config()?;
        let ctx = ModelCtx::default();
        let tok = gguf.tokenizer()?;
        let mut model = {
            let src = gguf.source(&ctx, DType::BF16, device.clone());
            Qwen35CausalLM::new(&cfg, &src, &ctx)?
        };
        model.clear_cache();

        let prompt = "<|im_start|>user\nExplain quantum computing in simple terms.<|im_end|>\n<|im_start|>assistant\n";
        let ids = tok
            .encode(prompt, false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        let eos = gguf
            .content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(248046);

        let input = Tensor::from_slice(&ids, (1, ids.len()), &device)?;
        let mut logits = model.forward(&input, 0)?;
        let mut offset = ids.len();
        let mut out = Vec::new();
        for _ in 0..24 {
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
        }
        let text = tok
            .decode(&out, true)
            .map_err(|e| candle::Error::Msg(format!("decode: {e}")))?;
        eprintln!("=== GENERATED ({} tokens) ===\n{text}\n===", out.len());
        assert!(!out.is_empty());
        Ok(())
    }
}
