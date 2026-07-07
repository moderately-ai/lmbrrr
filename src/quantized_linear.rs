use std::{
    collections::HashMap,
    fmt,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result};
use candle::{quantized::QMatMul, DType, Device, Module, Tensor};
use candle_nn::Linear;
use serde::Deserialize;

#[derive(Clone)]
pub enum MixedLinear {
    Dense(Linear),
    QMatMul(Arc<QMatMul>),
}

impl MixedLinear {
    pub fn dense(linear: Linear) -> Self {
        Self::Dense(linear)
    }

    pub fn from_dequantized_weight(weight: Tensor) -> Self {
        Self::QMatMul(Arc::new(QMatMul::Tensor(weight)))
    }

    pub fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(xs),
            Self::QMatMul(qmatmul) => qmatmul.forward(xs),
        }
    }
}

impl fmt::Debug for MixedLinear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense(_) => f.write_str("MixedLinear::Dense"),
            Self::QMatMul(_) => f.write_str("MixedLinear::QMatMul"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct QuantizedTextArtifact {
    manifest_path: PathBuf,
    data_path: Option<PathBuf>,
    tensors: HashMap<String, QuantizedTensor>,
    device: Device,
    dtype: DType,
}

impl QuantizedTextArtifact {
    pub fn from_manifest(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("open quantized manifest {}", path.display()))?;
        let manifest: QuantizedManifest = serde_json::from_reader(file)
            .with_context(|| format!("parse quantized manifest {}", path.display()))?;
        if manifest.kind != "lmbrrr_mixed_precision_weights" {
            anyhow::bail!(
                "quantized manifest kind is {}, expected lmbrrr_mixed_precision_weights",
                manifest.kind
            );
        }
        let manifest_dir = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let data_path = manifest
            .output
            .data_file
            .as_ref()
            .map(|file| manifest_dir.join(file));
        let tensors = manifest
            .tensors
            .into_iter()
            .filter_map(|tensor| {
                let data = tensor.data?;
                (tensor.format != "source").then_some((
                    tensor.name,
                    QuantizedTensor {
                        format: tensor.format,
                        shape: tensor.shape,
                        num_elements: tensor.num_elements,
                        data,
                    },
                ))
            })
            .collect::<HashMap<_, _>>();
        if !tensors.is_empty() && data_path.is_none() {
            anyhow::bail!("quantized manifest has tensor data entries but no output.data_file");
        }
        Ok(Self {
            manifest_path: path.to_path_buf(),
            data_path,
            tensors,
            device: device.clone(),
            dtype,
        })
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn quantized_tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn load_linear(&self, name: &str) -> Result<Option<MixedLinear>> {
        let Some(tensor) = self.tensors.get(name) else {
            return Ok(None);
        };
        let values = self.load_values(name, tensor)?;
        let weight =
            Tensor::from_vec(values, tensor.shape.clone(), &self.device)?.to_dtype(self.dtype)?;
        Ok(Some(MixedLinear::from_dequantized_weight(weight)))
    }

    fn load_values(&self, name: &str, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
        let data_path = self
            .data_path
            .as_ref()
            .context("quantized artifact has no data file")?;
        let mut file = File::open(data_path)
            .with_context(|| format!("open quantized data {}", data_path.display()))?;
        file.seek(SeekFrom::Start(tensor.data.offset))
            .with_context(|| format!("seek quantized tensor {name}"))?;
        let mut bytes = vec![0u8; tensor.data.length as usize];
        file.read_exact(&mut bytes)
            .with_context(|| format!("read quantized tensor {name}"))?;
        match tensor.format.as_str() {
            "q8_symmetric" => decode_q8(name, &bytes, tensor.num_elements),
            "q4k_block64_symmetric" => decode_q4_block64(name, &bytes, tensor.num_elements),
            other => anyhow::bail!("unsupported quantized tensor format {other} for {name}"),
        }
    }
}

#[derive(Clone, Debug)]
struct QuantizedTensor {
    format: String,
    shape: Vec<usize>,
    num_elements: usize,
    data: QuantizedData,
}

#[derive(Clone, Debug, Deserialize)]
struct QuantizedManifest {
    kind: String,
    output: QuantizedOutput,
    tensors: Vec<QuantizedTensorManifest>,
}

#[derive(Clone, Debug, Deserialize)]
struct QuantizedOutput {
    data_file: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct QuantizedTensorManifest {
    name: String,
    format: String,
    shape: Vec<usize>,
    num_elements: usize,
    data: Option<QuantizedData>,
}

#[derive(Clone, Debug, Deserialize)]
struct QuantizedData {
    offset: u64,
    length: u64,
}

fn decode_q8(name: &str, bytes: &[u8], num_elements: usize) -> Result<Vec<f32>> {
    if bytes.len() != 4 + num_elements {
        anyhow::bail!(
            "q8 tensor {name} has {} bytes, expected {}",
            bytes.len(),
            4 + num_elements
        );
    }
    let scale = f32::from_le_bytes(bytes[..4].try_into().expect("slice has length 4"));
    Ok(bytes[4..]
        .iter()
        .map(|byte| (*byte as i8) as f32 * scale)
        .collect())
}

fn decode_q4_block64(name: &str, bytes: &[u8], num_elements: usize) -> Result<Vec<f32>> {
    const BLOCK: usize = 64;
    let expected = num_elements.div_ceil(BLOCK) * 4 + num_elements.div_ceil(2);
    if bytes.len() != expected {
        anyhow::bail!(
            "q4k tensor {name} has {} bytes, expected {expected}",
            bytes.len()
        );
    }
    let mut out = Vec::with_capacity(num_elements);
    let mut cursor = 0usize;
    for _ in 0..num_elements.div_ceil(BLOCK) {
        let scale = f32::from_le_bytes(
            bytes[cursor..cursor + 4]
                .try_into()
                .expect("slice has length 4"),
        );
        cursor += 4;
        let values_in_block = (num_elements - out.len()).min(BLOCK);
        for _ in 0..values_in_block.div_ceil(2) {
            let packed = bytes[cursor];
            cursor += 1;
            let left = ((packed & 0x0f) as i8) - 8;
            out.push(left as f32 * scale);
            if out.len() < num_elements && out.len() % BLOCK != 0 {
                let right = ((packed >> 4) as i8) - 8;
                out.push(right as f32 * scale);
            }
        }
    }
    out.truncate(num_elements);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_q8_values() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        bytes.extend_from_slice(&[254u8, 0u8, 2u8]);
        let values = decode_q8("test", &bytes, 3).unwrap();
        assert_eq!(values, vec![-1.0, 0.0, 1.0]);
    }
}
