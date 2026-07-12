//! Exact per-row argmax plus top-k candidates from verify logits, for
//! verify-logit token recycling.

use anyhow::{Context, Result};
use candle::{DType, Device, Tensor, D};

/// Exact per-row argmax plus top-k (id, logit) candidates from [1, l, V]
/// logits via two-stage chunk reduction, for verify-logit token recycling.
///
/// Stage 1 reduces each row to per-chunk maxima (chunks of 256) on device —
/// a tiny readback instead of the full vocab row. Stage 2 gathers only the
/// rows' top-k chunks (one batched index_select, one readback) and finishes
/// exactly on host. Exactness: the k-th largest global value is exceeded by
/// at most k−1 others, so at most k chunks can have a maximum ≥ it — the
/// top-k chunk-maxima chunks provably contain the true top-k values.
/// Tie semantics match candle argmax (lowest index wins at both stages).
const RECYCLE_CHUNK: usize = 256;

pub fn logits_argmax_and_topk(
    logits: &Tensor,
    k: usize,
) -> Result<Vec<(u32, Vec<(u32, f32)>)>> {
    let (_, l, vocab) = logits.dims3()?;
    let chunks = vocab.div_ceil(RECYCLE_CHUNK);
    let padded = chunks * RECYCLE_CHUNK;
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let padded_logits = if padded == vocab {
        logits_f32
    } else {
        let pad = Tensor::full(
            f32::NEG_INFINITY,
            (1, l, padded - vocab),
            logits.device(),
        )?;
        Tensor::cat(&[&logits_f32, &pad], D::Minus1)?
    };
    let rows = padded_logits.reshape((l, chunks, RECYCLE_CHUNK))?;
    let chunk_maxima = rows.max(D::Minus1)?.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

    // Per row, pick the top-k chunks by maxima (lowest chunk index wins ties
    // to preserve argmax tie semantics).
    let take = k.min(chunks);
    let mut selected = Vec::with_capacity(l * take);
    let mut selected_per_row = Vec::with_capacity(l);
    for (row, maxima) in chunk_maxima.iter().enumerate() {
        let mut order: Vec<usize> = (0..chunks).collect();
        order.sort_by(|&a, &b| {
            maxima[b]
                .partial_cmp(&maxima[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let row_chunks: Vec<usize> = order[..take].to_vec();
        for &chunk in &row_chunks {
            selected.push((row * chunks + chunk) as u32);
        }
        selected_per_row.push(row_chunks);
    }

    // One batched gather of every selected chunk, one readback.
    let flat = rows.reshape((l * chunks, RECYCLE_CHUNK))?;
    let idx = Tensor::from_slice(&selected, selected.len(), logits.device())?;
    let gathered = flat
        .index_select(&idx, 0)?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?;

    let mut out = Vec::with_capacity(l);
    for (row, row_chunks) in selected_per_row.iter().enumerate() {
        // Exact top-k over the union of the selected chunks' values.
        let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(take * RECYCLE_CHUNK);
        for (slot, &chunk) in row_chunks.iter().enumerate() {
            let values = &gathered[row * take + slot];
            let base = chunk * RECYCLE_CHUNK;
            for (offset, &value) in values.iter().enumerate() {
                let id = base + offset;
                if id < vocab {
                    candidates.push((id as u32, value));
                }
            }
        }
        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        candidates.truncate(k);
        let argmax = candidates
            .first()
            .map(|(id, _)| *id)
            .context("logits row produced no candidates")?;
        out.push((argmax, candidates));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-stage top-k must agree exactly with candle argmax and a naive
    /// full-row top-k, including lowest-index tie semantics, across padded /
    /// unpadded vocab sizes and duplicate values.
    #[test]
    fn chunked_topk_matches_naive_exactly() {
        let device = Device::Cpu;
        // Deterministic pseudo-random values; includes exact duplicates.
        for (l, vocab) in [(1usize, 1000usize), (3, 4096), (2, 248094 / 64)] {
            let values: Vec<f32> = (0..l * vocab)
                .map(|i| (((i * 2654435761) % 1013) as f32) / 7.0)
                .collect();
            let logits = Tensor::from_slice(&values, (1, l, vocab), &device).unwrap();
            let got = logits_argmax_and_topk(&logits, 8).unwrap();

            let reference_argmax = logits
                .squeeze(0)
                .unwrap()
                .argmax(D::Minus1)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            for row in 0..l {
                assert_eq!(got[row].0, reference_argmax[row], "argmax row {row}");
                // Naive exact top-8 with the same tie rule.
                let mut naive: Vec<(u32, f32)> = values[row * vocab..(row + 1) * vocab]
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i as u32, v))
                    .collect();
                naive.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap()
                        .then(a.0.cmp(&b.0))
                });
                naive.truncate(8);
                assert_eq!(got[row].1, naive, "top-8 row {row}");
            }
        }
    }
}
