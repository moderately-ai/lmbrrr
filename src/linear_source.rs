use candle::{DType, Device, Result, Shape, Tensor};
use candle_nn::VarBuilder;

use crate::quantized_linear::MixedLinear;

/// One row-block of a (possibly fused) linear weight: the `.weight` tensor
/// `name` (relative to the source's prefix) contributing `out` output rows.
pub struct LinearPart<'a> {
    pub name: &'a str,
    pub out: usize,
}

impl<'a> LinearPart<'a> {
    pub fn new(name: &'a str, out: usize) -> Self {
        Self { name, out }
    }
}

/// Weight-source seam. The qwen35 constructors build through this so one body
/// serves both the dense safetensors path ([`VarBuilderSource`]) and the
/// packed GGUF path (`GgufSource`, added with the loader). Quantization
/// decisions live in the source: `VarBuilderSource` returns dense linears (the
/// safetensors path quantizes in a later `apply_quantized_text_artifact`
/// pass), the GGUF source returns packed ones directly.
pub trait LinearSource {
    fn dtype(&self) -> DType;
    fn device(&self) -> &Device;
    /// Raw tensor at `name` in the source dtype (norm weights, biases, conv,
    /// A_log, dt_bias, the embedding table). Callers own any dtype handling.
    fn tensor(&self, shape: Shape, name: &str) -> Result<Tensor>;
    /// Linear whose weight rows are `parts` concatenated along dim 0, all with
    /// input dim `in_dim`; with `bias`, the matching `<stem>.bias` tensors are
    /// concatenated the same way.
    fn linear(&self, parts: &[LinearPart<'_>], in_dim: usize, bias: bool) -> Result<MixedLinear>;
    /// Sub-source at a nested prefix (mirrors `VarBuilder::pp`).
    fn sub(&self, prefix: &str) -> Self
    where
        Self: Sized;

    /// Whether norm weights already carry their final value. Raw Qwen
    /// safetensors store `w-1` (the `+1` zero-centring shift is applied at
    /// load); llama.cpp's GGUF conversion folds the `+1` in, so those must NOT
    /// be shifted again. Default false (the safetensors convention).
    fn norms_pre_shifted(&self) -> bool {
        false
    }
}

fn bias_name(weight_name: &str) -> String {
    match weight_name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.bias"),
        None => format!("{weight_name}.bias"),
    }
}

/// Dense path: reproduces the historical `vb.get` + `Tensor::cat` +
/// `MixedLinear::dense` construction exactly (byte-identical to the
/// pre-seam constructors).
pub struct VarBuilderSource<'a> {
    vb: VarBuilder<'a>,
}

impl<'a> VarBuilderSource<'a> {
    pub fn new(vb: VarBuilder<'a>) -> Self {
        Self { vb }
    }
}

impl LinearSource for VarBuilderSource<'_> {
    fn dtype(&self) -> DType {
        self.vb.dtype()
    }

    fn device(&self) -> &Device {
        self.vb.device()
    }

    fn tensor(&self, shape: Shape, name: &str) -> Result<Tensor> {
        self.vb.get(shape, name)
    }

    fn linear(&self, parts: &[LinearPart<'_>], in_dim: usize, bias: bool) -> Result<MixedLinear> {
        let weights = parts
            .iter()
            .map(|p| self.vb.get((p.out, in_dim), p.name))
            .collect::<Result<Vec<_>>>()?;
        let weight = if weights.len() == 1 {
            weights.into_iter().next().unwrap()
        } else {
            Tensor::cat(&weights.iter().collect::<Vec<_>>(), 0)?
        };
        let bias = if bias {
            let biases = parts
                .iter()
                .map(|p| self.vb.get(p.out, bias_name(p.name).as_str()))
                .collect::<Result<Vec<_>>>()?;
            Some(if biases.len() == 1 {
                biases.into_iter().next().unwrap()
            } else {
                Tensor::cat(&biases.iter().collect::<Vec<_>>(), 0)?
            })
        } else {
            None
        };
        Ok(MixedLinear::dense(candle_nn::Linear::new(weight, bias)))
    }

    fn sub(&self, prefix: &str) -> Self {
        Self {
            vb: self.vb.pp(prefix),
        }
    }
}
