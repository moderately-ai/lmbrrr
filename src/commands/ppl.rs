//! Quality reference battery: corpus perplexity for the deployed
//! configuration plus per-position mean/max KL divergence and top-1
//! agreement against the BF16 dense reference — the standing quality gate
//! for quantization policy changes (imatrix, mixed rungs, head tiers).

use crate::*;

pub(crate) fn ppl(args: PplArgs) -> Result<()> {
    if args.chunk_tokens < 2 {
        anyhow::bail!("--chunk-tokens must be at least 2 (need one prediction per chunk)");
    }
    let text = fs::read_to_string(&args.text_file)
        .with_context(|| format!("read ppl corpus {}", args.text_file.display()))?;

    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let ids = tokenize_prompt(&tokenizer, text)?;
    let chunks = corpus_chunks(&ids, args.chunk_tokens, args.max_chunks);
    if chunks.is_empty() {
        anyhow::bail!(
            "corpus tokenized to {} tokens; no chunk of {} fits",
            ids.len(),
            args.chunk_tokens
        );
    }

    // Deployed arm: exactly the ModelArgs configuration (manifest, lm-head
    // tier, restriction). Reference arm: the same weights fully dense BF16.
    let deployed_is_dense = args.model.quantized_manifest.is_none()
        && args.model.quantize_lm_head.is_none()
        && args.model.target_head_vocab_size.is_none();
    let with_reference = !args.no_reference && !deployed_is_dense;
    let (mut deployed, _, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
        head_loader_quant(&args.model),
    )?;
    maybe_restrict_head(&mut deployed, &args.model)?;
    let mut reference = if with_reference {
        Some(load_model(&bundle, dtype, &device)?.0)
    } else {
        None
    };
    if deployed_is_dense && !args.no_reference {
        eprintln!("deployed arm is already dense BF16; skipping the reference arm (KLD vs itself is 0)");
    }

    let mut deployed_nll = 0.0f64;
    let mut reference_nll = 0.0f64;
    let mut kld_sum = 0.0f64;
    let mut kld_max = 0.0f64;
    let mut top1_agree = 0usize;
    let mut predicted = 0usize;
    let mut chunk_rows = Vec::with_capacity(chunks.len());
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let dep = arm_log_softmax(&mut deployed, chunk, &device, &args.model.downsample_mode)?;
        let targets = target_ids_tensor(chunk, &device)?;
        let dep_nll = chunk_nll(&dep, &targets)?;
        deployed_nll += dep_nll;

        let mut row = serde_json::json!({
            "chunk": chunk_index,
            "tokens": chunk.len(),
            "deployed_nll_per_token": dep_nll / (chunk.len() - 1) as f64,
        });
        if let Some(reference) = reference.as_mut() {
            let refr = arm_log_softmax(reference, chunk, &device, &args.model.downsample_mode)?;
            let ref_nll = chunk_nll(&refr, &targets)?;
            reference_nll += ref_nll;
            // Per position: KL(ref || dep) = sum_v p_ref (logp_ref - logp_dep),
            // and top-1 agreement between the two argmaxes.
            let kl = refr
                .exp()?
                .mul(&refr.sub(&dep)?)?
                .sum(D::Minus1)?
                .to_vec2::<f32>()?;
            let agree = arm_argmax_agreement(&refr, &dep)?;
            top1_agree += agree;
            let chunk_kld: f64 = kl[0].iter().map(|v| *v as f64).sum();
            let chunk_kld_max = kl[0].iter().fold(0.0f64, |m, v| m.max(*v as f64));
            kld_sum += chunk_kld;
            kld_max = kld_max.max(chunk_kld_max);
            row["reference_nll_per_token"] =
                serde_json::json!(ref_nll / (chunk.len() - 1) as f64);
            row["mean_kld"] = serde_json::json!(chunk_kld / (chunk.len() - 1) as f64);
            row["max_kld"] = serde_json::json!(chunk_kld_max);
            row["top1_agreement"] =
                serde_json::json!(agree as f64 / (chunk.len() - 1) as f64);
        }
        predicted += chunk.len() - 1;
        chunk_rows.push(row);
        eprintln!(
            "chunk {}/{}: deployed ppl so far {:.4}",
            chunk_index + 1,
            chunks.len(),
            (deployed_nll / predicted as f64).exp()
        );
    }

    let report = serde_json::json!({
        "kind": "lmbrrr_quality_reference_battery",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "corpus": args.text_file,
        "chunk_tokens": args.chunk_tokens,
        "chunks": chunks.len(),
        "predicted_tokens": predicted,
        "quantized_load": quantized_load_json(&quantized_load),
        "deployed_ppl": (deployed_nll / predicted as f64).exp(),
        "reference_ppl": reference
            .is_some()
            .then(|| (reference_nll / predicted as f64).exp()),
        "mean_kld": reference.is_some().then(|| kld_sum / predicted as f64),
        "max_kld": reference.is_some().then_some(kld_max),
        "top1_agreement": reference
            .is_some()
            .then(|| top1_agree as f64 / predicted as f64),
        "per_chunk": chunk_rows,
    });
    write_json_report(args.output.as_ref(), &report)
}

/// Fixed non-overlapping evaluation windows; a trailing remainder shorter
/// than the chunk size is dropped so every chunk sees identical context
/// shape (both arms MUST use the same chunking for the KLD to be aligned).
fn corpus_chunks(ids: &[u32], chunk_tokens: usize, max_chunks: Option<usize>) -> Vec<Vec<u32>> {
    let mut chunks: Vec<Vec<u32>> = ids
        .chunks_exact(chunk_tokens)
        .map(|chunk| chunk.to_vec())
        .collect();
    if let Some(max) = max_chunks {
        chunks.truncate(max);
    }
    chunks
}

/// One arm's log-probabilities for a chunk: fresh state, full-chunk forward
/// through every lm_head position (bypasses the last-position narrowing),
/// F32 log-softmax over the vocab, narrowed to the L-1 predicting rows.
fn arm_log_softmax(
    model: &mut MiniCpmForConditionalGeneration,
    chunk: &[u32],
    device: &Device,
    downsample_mode: &str,
) -> Result<Tensor> {
    model.clear_cache();
    let input = Tensor::from_slice(chunk, (1, chunk.len()), device)?;
    let logits = model.forward_all_logits(&input, None::<&ProcessedImages>, downsample_mode, 0)?;
    let predicting = logits.narrow(1, 0, chunk.len() - 1)?.to_dtype(DType::F32)?;
    Ok(candle_nn::ops::log_softmax(&predicting, D::Minus1)?)
}

/// The next-token targets for a chunk as a gather index: [1, L-1, 1].
fn target_ids_tensor(chunk: &[u32], device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_slice(&chunk[1..], (1, chunk.len() - 1, 1), device)?)
}

/// Total negative log-likelihood of the targets under [1, L-1, V] log-probs.
fn chunk_nll(log_probs: &Tensor, targets: &Tensor) -> Result<f64> {
    let picked = log_probs.gather(targets, 2)?.to_vec3::<f32>()?;
    Ok(-picked[0]
        .iter()
        .map(|row| row[0] as f64)
        .sum::<f64>())
}

/// Positions where both arms' argmax agree (exact greedy-token agreement).
fn arm_argmax_agreement(reference: &Tensor, deployed: &Tensor) -> Result<usize> {
    let ref_ids = reference.argmax(D::Minus1)?.to_vec2::<u32>()?;
    let dep_ids = deployed.argmax(D::Minus1)?.to_vec2::<u32>()?;
    Ok(ref_ids[0]
        .iter()
        .zip(dep_ids[0].iter())
        .filter(|(a, b)| a == b)
        .count())
}

#[cfg(test)]
mod tests {
    use super::corpus_chunks;

    #[test]
    fn chunks_are_fixed_size_and_drop_the_tail() {
        let ids: Vec<u32> = (0..10).collect();
        let chunks = corpus_chunks(&ids, 4, None);
        assert_eq!(chunks, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
    }

    #[test]
    fn max_chunks_caps_the_window_count() {
        let ids: Vec<u32> = (0..12).collect();
        let chunks = corpus_chunks(&ids, 4, Some(2));
        assert_eq!(chunks.len(), 2);
    }
}
