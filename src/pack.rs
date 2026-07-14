//! GGML-ready weight pack: a GGUF sidecar that caches the exact quantized
//! bytes the loader would otherwise recompute on EVERY process start.
//!
//! Receipts (2026-07-14 Metal System Trace autopsy): the artifact loader
//! decodes each lmbq blob to f32 and re-runs `make_qkx1_quants` over ~350 MB
//! of text weights plus the 254M-value lm_head, single-threaded, at every
//! launch — 8 s on the M3, 10-16 s on the M4 — and no report field measured
//! it. The pack stores the finished q4_K/q8_0 blocks once; a warm start is
//! read + upload.
//!
//! Bitwise contract: pack contents are produced by the same CPU quantizer
//! the slow path uses (`QTensor::quantize`), so a pack hit uploads the exact
//! bytes a cold load would produce — goldens must not move.
//!
//! Invalidation: a fingerprint (manifest sha256 + data-file length + head
//! tier + schema version) is stored in the GGUF metadata; any mismatch is
//! treated as no-pack and the file is rewritten after the slow load.
//! LMBRRR_PACK=0 disables both use and writing.

use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
    sync::Mutex,
};

use anyhow::{Context, Result};
use candle::{
    quantized::{ggml_file::qtensor_from_ggml, gguf_file, GgmlDType, QTensor},
    Device,
};
use sha2::{Digest, Sha256};

const PACK_SCHEMA: u32 = 1;
const FINGERPRINT_KEY: &str = "lmbrrr.pack.fingerprint";

/// One quantized tensor awaiting the pack write: the CPU-side ggml bytes
/// plus enough structure to rebuild a QTensor for the GGUF writer.
struct PendingTensor {
    key: String,
    dtype: GgmlDType,
    dims: Vec<usize>,
    bytes: Vec<u8>,
}

pub struct PackStore {
    path: PathBuf,
    fingerprint: String,
    device: Device,
    /// Tensors eagerly loaded from a valid pack, consumed by `take`.
    loaded: Mutex<HashMap<String, QTensor>>,
    /// Slow-path results awaiting `finish` (only populated on a miss).
    pending: Mutex<Vec<PendingTensor>>,
    hit: bool,
    enabled: bool,
}

impl std::fmt::Debug for PackStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackStore")
            .field("path", &self.path)
            .field("status", &self.status())
            .finish()
    }
}

impl PackStore {
    /// Opens (or plans) the pack next to the manifest. A valid existing pack
    /// is eagerly loaded to the device here — that IS the fast start.
    pub fn open(
        manifest_path: &Path,
        head_tier: Option<GgmlDType>,
        device: &Device,
    ) -> Result<Self> {
        let enabled = std::env::var("LMBRRR_PACK").map_or(true, |v| v != "0");
        let manifest_dir = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let path = manifest_dir.join("packed.gguf");
        let fingerprint = Self::fingerprint(manifest_path, head_tier)?;
        let mut store = Self {
            path,
            fingerprint,
            device: device.clone(),
            loaded: Mutex::new(HashMap::new()),
            pending: Mutex::new(Vec::new()),
            hit: false,
            enabled,
        };
        if store.enabled && store.path.exists() {
            match store.try_load() {
                Ok(true) => store.hit = true,
                Ok(false) => {}
                Err(err) => {
                    // A corrupt/stale pack must never block a load; the slow
                    // path rebuilds and overwrites it.
                    eprintln!(
                        "warning: ignoring weight pack {} ({err:#})",
                        store.path.display()
                    );
                }
            }
        }
        Ok(store)
    }

    /// Disabled store: every lookup misses and nothing is recorded.
    pub fn disabled(device: &Device) -> Self {
        Self {
            path: PathBuf::new(),
            fingerprint: String::new(),
            device: device.clone(),
            loaded: Mutex::new(HashMap::new()),
            pending: Mutex::new(Vec::new()),
            hit: false,
            enabled: false,
        }
    }

    fn fingerprint(manifest_path: &Path, head_tier: Option<GgmlDType>) -> Result<String> {
        let manifest_bytes = std::fs::read(manifest_path)
            .with_context(|| format!("read manifest {}", manifest_path.display()))?;
        let mut hasher = Sha256::new();
        hasher.update(&manifest_bytes);
        // The lmbq blob is written together with the manifest; its length is
        // a cheap corruption/mismatch tripwire without hashing 280 MB.
        let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&manifest_bytes) {
            if let Some(data_file) = json
                .pointer("/output/data_file")
                .and_then(serde_json::Value::as_str)
            {
                let len = std::fs::metadata(manifest_dir.join(data_file))
                    .map(|m| m.len())
                    .unwrap_or(0);
                hasher.update(len.to_le_bytes());
            }
        }
        let head = match head_tier {
            Some(t) => format!("{t:?}"),
            None => "none".to_string(),
        };
        Ok(format!(
            "v{PACK_SCHEMA}:{:x}:head={head}",
            hasher.finalize()
        ))
    }

    fn try_load(&self) -> Result<bool> {
        let file = File::open(&self.path)
            .with_context(|| format!("open pack {}", self.path.display()))?;
        let mut reader = BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader).context("read pack gguf")?;
        let stored = match content.metadata.get(FINGERPRINT_KEY) {
            Some(gguf_file::Value::String(s)) => s.clone(),
            _ => anyhow::bail!("pack has no fingerprint metadata"),
        };
        if stored != self.fingerprint {
            anyhow::bail!(
                "pack fingerprint mismatch (stored {stored}, want {})",
                self.fingerprint
            );
        }
        let mut loaded = self.loaded.lock().expect("pack loaded lock poisoned");
        for name in content.tensor_infos.keys() {
            let tensor = content
                .tensor(&mut reader, name, &self.device)
                .with_context(|| format!("load pack tensor {name}"))?;
            loaded.insert(name.clone(), tensor);
        }
        Ok(true)
    }

    pub fn hit(&self) -> bool {
        self.hit
    }

    pub fn status(&self) -> &'static str {
        if !self.enabled {
            "disabled"
        } else if self.hit {
            "hit"
        } else {
            "miss"
        }
    }

    /// Consumes a packed tensor (each weight is used exactly once per load).
    pub fn take(&self, key: &str) -> Option<QTensor> {
        self.loaded
            .lock()
            .expect("pack loaded lock poisoned")
            .remove(key)
    }

    /// Quantizes on CPU, records the bytes for the pack write, and returns
    /// the device QTensor built from those same bytes (bit-identical to
    /// `QTensor::quantize_onto`, which runs the same CPU quantizer).
    pub fn quantize_and_record(
        &self,
        key: &str,
        cpu_f32: &candle::Tensor,
        dtype: GgmlDType,
    ) -> Result<QTensor> {
        let cpu_q = QTensor::quantize(cpu_f32, dtype).context("cpu quantize")?;
        let dims = cpu_q.shape().dims().to_vec();
        let bytes = cpu_q.data().context("read quantized bytes")?.into_owned();
        let device_q = qtensor_from_ggml(dtype, &bytes, dims.clone(), &self.device)
            .with_context(|| format!("upload packed tensor {key}"))?;
        if self.enabled && !self.hit {
            self.pending
                .lock()
                .expect("pack pending lock poisoned")
                .push(PendingTensor {
                    key: key.to_string(),
                    dtype,
                    dims,
                    bytes,
                });
        }
        Ok(device_q)
    }

    /// Writes the pack after a miss-load (atomic: temp file + rename).
    /// Returns the written path, or None when nothing needed writing.
    pub fn finish(&self) -> Result<Option<PathBuf>> {
        if !self.enabled || self.hit {
            return Ok(None);
        }
        let pending = std::mem::take(
            &mut *self.pending.lock().expect("pack pending lock poisoned"),
        );
        if pending.is_empty() {
            return Ok(None);
        }
        let cpu_tensors = pending
            .iter()
            .map(|p| {
                let q = qtensor_from_ggml(p.dtype, &p.bytes, p.dims.clone(), &Device::Cpu)
                    .with_context(|| format!("rebuild pack tensor {}", p.key))?;
                Ok((p.key.as_str(), q))
            })
            .collect::<Result<Vec<_>>>()?;
        let refs = cpu_tensors
            .iter()
            .map(|(name, q)| (*name, q))
            .collect::<Vec<_>>();
        let fingerprint = gguf_file::Value::String(self.fingerprint.clone());
        let metadata = [(FINGERPRINT_KEY, &fingerprint)];
        let tmp = self.path.with_extension("gguf.tmp");
        {
            let file = File::create(&tmp)
                .with_context(|| format!("create pack temp {}", tmp.display()))?;
            let mut writer = BufWriter::new(file);
            gguf_file::write(&mut writer, &metadata, &refs).context("write pack gguf")?;
        }
        std::fs::rename(&tmp, &self.path)
            .with_context(|| format!("rename pack into place {}", self.path.display()))?;
        Ok(Some(self.path.clone()))
    }
}
