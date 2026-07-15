//! Speculative-verification harnesses: the single-round spec-verify command,
//! greedy chunk verification + acceptance analysis, the tree-equivalence
//! check, and the drafter parity gate.

use crate::*;

#[derive(Clone, Debug)]
struct SpecVerifyPosition {
    index: usize,
    draft_token_id: u32,
    target_token_id: u32,
    token_match: bool,
    accepted: bool,
    first_rejected: bool,
}

#[derive(Clone, Debug)]
struct SpecVerifyAnalysis {
    positions: Vec<SpecVerifyPosition>,
    accepted_tokens: usize,
    first_rejected_index: Option<usize>,
    bonus_token_id: u32,
    reconstructed_token_ids: Vec<u32>,
}

impl SpecVerifyAnalysis {
    fn verified_tokens(&self) -> usize {
        self.positions.len()
    }

    fn bonus_tokens(&self) -> usize {
        1
    }

    fn accepted_length(&self) -> usize {
        self.accepted_tokens + self.bonus_tokens()
    }

    fn acceptance_rate(&self) -> Option<f64> {
        let verified = self.verified_tokens();
        (verified > 0).then(|| self.accepted_tokens as f64 / verified as f64)
    }

    fn verifier_waste_tokens(&self) -> usize {
        self.first_rejected_index
            .map(|idx| self.verified_tokens().saturating_sub(idx + 1))
            .unwrap_or(0)
    }

    fn verifier_waste_share(&self) -> Option<f64> {
        let verified = self.verified_tokens();
        (verified > 0).then(|| self.verifier_waste_tokens() as f64 / verified as f64)
    }
}

#[derive(Clone, Debug)]
struct SpecVerifyStats {
    analysis: SpecVerifyAnalysis,
    target_token_ids: Vec<u32>,
    prefill_elapsed: Duration,
    verify_elapsed: Duration,
    argmax_elapsed: Duration,
}

#[derive(Clone, Debug)]
struct ConfidenceSchedule {
    threshold: f64,
    original_draft_tokens: usize,
    scheduled_draft_tokens: usize,
    dropped_draft_tokens: usize,
    scheduled_cumulative_confidence: f64,
    next_rejected_cumulative_confidence: Option<f64>,
    confidences: Vec<f64>,
}

#[derive(Clone, Debug)]
struct DraftCorruption {
    index: usize,
    original_token_id: u32,
    corrupted_token_id: u32,
}

fn corrupt_draft_token(
    draft_tokens: &mut [u32],
    index: usize,
    vocab_size: usize,
) -> Result<DraftCorruption> {
    let token = draft_tokens
        .get_mut(index)
        .with_context(|| format!("--corrupt-draft-at {index} is outside the draft sequence"))?;
    let original = *token;
    let corrupted = if (original as usize) + 1 < vocab_size {
        original + 1
    } else {
        original.saturating_sub(1)
    };
    if corrupted == original {
        anyhow::bail!("cannot corrupt draft token {original} with vocab size {vocab_size}");
    }
    *token = corrupted;
    Ok(DraftCorruption {
        index,
        original_token_id: original,
        corrupted_token_id: corrupted,
    })
}

fn apply_confidence_schedule(
    draft_tokens: &mut Vec<u32>,
    confidences: &[f64],
    threshold: Option<f64>,
) -> Result<Option<ConfidenceSchedule>> {
    let Some(threshold) = threshold else {
        if !confidences.is_empty() {
            anyhow::bail!("--draft-confidence requires --schedule-confidence-threshold");
        }
        return Ok(None);
    };
    if !(0.0..=1.0).contains(&threshold) {
        anyhow::bail!("--schedule-confidence-threshold must be between 0 and 1");
    }
    if confidences.len() < draft_tokens.len() {
        anyhow::bail!(
            "got {} draft confidences for {} draft tokens",
            confidences.len(),
            draft_tokens.len()
        );
    }
    if let Some(confidence) = confidences
        .iter()
        .find(|confidence| !(0.0..=1.0).contains(*confidence))
    {
        anyhow::bail!("draft confidence {confidence} is outside 0..=1");
    }

    let original_draft_tokens = draft_tokens.len();
    let (scheduled_draft_tokens, scheduled_cumulative_confidence, next_rejected) =
        schedule_prefix_len(&confidences[..original_draft_tokens], threshold);
    draft_tokens.truncate(scheduled_draft_tokens);
    Ok(Some(ConfidenceSchedule {
        threshold,
        original_draft_tokens,
        scheduled_draft_tokens,
        dropped_draft_tokens: original_draft_tokens.saturating_sub(scheduled_draft_tokens),
        scheduled_cumulative_confidence,
        next_rejected_cumulative_confidence: next_rejected,
        confidences: confidences[..original_draft_tokens].to_vec(),
    }))
}

fn schedule_prefix_len(confidences: &[f64], threshold: f64) -> (usize, f64, Option<f64>) {
    let mut cumulative = 1.0f64;
    let mut accepted = 0usize;
    for confidence in confidences {
        let next = cumulative * confidence;
        if next < threshold {
            return (accepted, cumulative, Some(next));
        }
        cumulative = next;
        accepted += 1;
    }
    (accepted, cumulative, None)
}

fn confidence_schedule_json(schedule: &Option<ConfidenceSchedule>) -> serde_json::Value {
    match schedule {
        Some(schedule) => serde_json::json!({
            "threshold": schedule.threshold,
            "original_draft_tokens": schedule.original_draft_tokens,
            "scheduled_draft_tokens": schedule.scheduled_draft_tokens,
            "dropped_draft_tokens": schedule.dropped_draft_tokens,
            "scheduled_cumulative_confidence": schedule.scheduled_cumulative_confidence,
            "next_rejected_cumulative_confidence": schedule.next_rejected_cumulative_confidence,
            "confidences": schedule.confidences,
        }),
        None => serde_json::Value::Null,
    }
}

fn verify_greedy_draft(
    model: &mut MiniCpmForConditionalGeneration,
    device: &Device,
    prompt_tokens: &[u32],
    draft_tokens: &[u32],
    downsample_mode: &str,
) -> Result<SpecVerifyStats> {
    model.clear_cache();
    let prompt_input = Tensor::from_slice(prompt_tokens, (1, prompt_tokens.len()), device)?;
    let prefill_start = Instant::now();
    let prompt_logits =
        model.forward(&prompt_input, None::<&ProcessedImages>, downsample_mode, 0)?;
    device.synchronize()?;
    let prefill_elapsed = prefill_start.elapsed();

    let (first_target_token, first_argmax_elapsed) = argmax_token(&prompt_logits, device)?;
    let mut argmax_elapsed = first_argmax_elapsed;
    let mut target_token_ids = Vec::with_capacity(draft_tokens.len());
    if draft_tokens.is_empty() {
        let analysis = analyze_verification(draft_tokens, &target_token_ids, first_target_token)?;
        return Ok(SpecVerifyStats {
            analysis,
            target_token_ids,
            prefill_elapsed,
            verify_elapsed: Duration::ZERO,
            argmax_elapsed,
        });
    }
    target_token_ids.push(first_target_token);

    let draft_input = Tensor::from_slice(draft_tokens, (1, draft_tokens.len()), device)?;
    let verify_start = Instant::now();
    let draft_logits = model.forward_all_logits(
        &draft_input,
        None::<&ProcessedImages>,
        downsample_mode,
        prompt_tokens.len(),
    )?;
    device.synchronize()?;
    let verify_elapsed = verify_start.elapsed();
    let (chunk_target_tokens, chunk_argmax_elapsed) = argmax_tokens(&draft_logits, device)?;
    argmax_elapsed += chunk_argmax_elapsed;

    if chunk_target_tokens.len() != draft_tokens.len() {
        anyhow::bail!(
            "verifier chunk returned {} target tokens for {} draft tokens",
            chunk_target_tokens.len(),
            draft_tokens.len()
        );
    }
    target_token_ids.extend(
        chunk_target_tokens
            .iter()
            .take(draft_tokens.len().saturating_sub(1))
            .copied(),
    );
    let bonus_after_all = chunk_target_tokens
        .last()
        .copied()
        .context("missing bonus token after draft chunk")?;
    let analysis = analyze_verification(draft_tokens, &target_token_ids, bonus_after_all)?;

    Ok(SpecVerifyStats {
        analysis,
        target_token_ids,
        prefill_elapsed,
        verify_elapsed,
        argmax_elapsed,
    })
}

fn analyze_verification(
    draft_tokens: &[u32],
    target_token_ids: &[u32],
    bonus_after_all: u32,
) -> Result<SpecVerifyAnalysis> {
    if draft_tokens.len() != target_token_ids.len() {
        anyhow::bail!(
            "draft length {} does not match target token length {}",
            draft_tokens.len(),
            target_token_ids.len()
        );
    }

    let accepted_tokens = draft_tokens
        .iter()
        .zip(target_token_ids.iter())
        .take_while(|(draft, target)| draft == target)
        .count();
    let first_rejected_index = (accepted_tokens < draft_tokens.len()).then_some(accepted_tokens);
    let bonus_token_id = first_rejected_index
        .map(|idx| target_token_ids[idx])
        .unwrap_or(bonus_after_all);
    let mut reconstructed_token_ids = draft_tokens[..accepted_tokens].to_vec();
    reconstructed_token_ids.push(bonus_token_id);

    let positions = draft_tokens
        .iter()
        .zip(target_token_ids.iter())
        .enumerate()
        .map(|(index, (draft, target))| SpecVerifyPosition {
            index,
            draft_token_id: *draft,
            target_token_id: *target,
            token_match: draft == target,
            accepted: index < accepted_tokens,
            first_rejected: first_rejected_index == Some(index),
        })
        .collect();

    Ok(SpecVerifyAnalysis {
        positions,
        accepted_tokens,
        first_rejected_index,
        bonus_token_id,
        reconstructed_token_ids,
    })
}

pub(crate) fn spec_verify(args: SpecVerifyArgs) -> Result<()> {
    if args.baseline_draft_tokens.is_some() && !args.draft_tokens.is_empty() {
        anyhow::bail!("use either --draft-token or --baseline-draft-tokens, not both");
    }
    if let Some(count) = args.baseline_draft_tokens {
        if count == 0 {
            anyhow::bail!("--baseline-draft-tokens must be greater than zero");
        }
    } else if args.draft_tokens.is_empty() {
        anyhow::bail!("provide at least one --draft-token or use --baseline-draft-tokens");
    }

    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let prompt_text = chat_prompt(&args.prompt, 0, args.enable_thinking);
    let prompt_tokens = tokenize_prompt(&tokenizer, prompt_text)?;
    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
        args.model.quantize_lm_head,
        &lmbrrr::runtime_config::RuntimeConfig::from_env(),
    )?;
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());

    let (mut draft_tokens, draft_source, baseline_tokens) =
        if let Some(count) = args.baseline_draft_tokens {
            let baseline_generation = greedy_generation_args(count + 1, args.enable_thinking);
            let baseline = generate_tokens(
                &mut model,
                &device,
                &baseline_generation,
                &prompt_tokens,
                None::<&ProcessedImages>,
                &args.model.downsample_mode,
                &eos_ids,
                |_, _, _, _| Ok(()),
            )?;
            if baseline.generated_token_ids.len() < count + 1 {
                anyhow::bail!(
                    "baseline generation produced {} tokens before EOS; need {}",
                    baseline.generated_token_ids.len(),
                    count + 1
                );
            }
            (
                baseline.generated_token_ids[..count].to_vec(),
                "baseline".to_string(),
                Some(baseline.generated_token_ids),
            )
        } else {
            (args.draft_tokens.clone(), "explicit".to_string(), None)
        };

    let corruption = if let Some(index) = args.corrupt_draft_at {
        Some(corrupt_draft_token(
            &mut draft_tokens,
            index,
            bundle.config.text_config.vocab_size,
        )?)
    } else {
        None
    };

    let confidence_schedule = apply_confidence_schedule(
        &mut draft_tokens,
        &args.draft_confidences,
        args.schedule_confidence_threshold,
    )?;

    let stats = verify_greedy_draft(
        &mut model,
        &device,
        &prompt_tokens,
        &draft_tokens,
        &args.model.downsample_mode,
    )?;
    let analysis = &stats.analysis;
    let accepted_token_ids = draft_tokens[..analysis.accepted_tokens].to_vec();
    let rejected_token_ids = analysis
        .first_rejected_index
        .map(|idx| draft_tokens[idx..].to_vec())
        .unwrap_or_default();
    let baseline_prefix_match = baseline_tokens.as_ref().map(|tokens| {
        tokens
            .get(..analysis.reconstructed_token_ids.len())
            .map(|prefix| prefix == analysis.reconstructed_token_ids.as_slice())
            .unwrap_or(false)
    });
    let expected_rejection_index = corruption
        .as_ref()
        .and_then(|corruption| (draft_source == "baseline").then_some(corruption.index));
    let rejection_matched_expectation =
        expected_rejection_index.map(|expected| analysis.first_rejected_index == Some(expected));

    let positions = analysis
        .positions
        .iter()
        .map(|position| {
            serde_json::json!({
                "index": position.index,
                "draft_token_id": position.draft_token_id,
                "target_token_id": position.target_token_id,
                "draft_token": decode_token_lossy(&tokenizer, position.draft_token_id),
                "target_token": decode_token_lossy(&tokenizer, position.target_token_id),
                "token_match": position.token_match,
                "accepted": position.accepted,
                "first_rejected": position.first_rejected,
            })
        })
        .collect::<Vec<_>>();

    let report = serde_json::json!({
        "kind": "lmbrrr_greedy_spec_verify",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "downsample_mode": args.model.downsample_mode.as_str(),
        "enable_thinking": args.enable_thinking,
        "artifact_seconds": secs(bundle.elapsed),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "draft_source": draft_source,
        "prompt": args.prompt.as_str(),
        "prompt_tokens": prompt_tokens.len(),
        "draft_tokens": draft_tokens.len(),
        "confidence_schedule": confidence_schedule_json(&confidence_schedule),
        "verified_tokens": analysis.verified_tokens(),
        "accepted_tokens": analysis.accepted_tokens,
        "bonus_tokens": analysis.bonus_tokens(),
        "accepted_length": analysis.accepted_length(),
        "acceptance_rate": analysis.acceptance_rate(),
        "first_rejected_index": analysis.first_rejected_index,
        "verifier_waste_tokens": analysis.verifier_waste_tokens(),
        "verifier_waste_share": analysis.verifier_waste_share(),
        "prefill_seconds": secs(stats.prefill_elapsed),
        "prefill_tokens_per_second": tokens_per_second(prompt_tokens.len(), stats.prefill_elapsed),
        "verify_seconds": secs(stats.verify_elapsed),
        "verify_tokens_per_second": tokens_per_second(draft_tokens.len(), stats.verify_elapsed),
        "argmax_seconds": secs(stats.argmax_elapsed),
        "round_seconds": secs(stats.prefill_elapsed + stats.verify_elapsed + stats.argmax_elapsed),
        "draft_token_ids": &draft_tokens,
        "target_token_ids": &stats.target_token_ids,
        "accepted_token_ids": &accepted_token_ids,
        "rejected_token_ids": rejected_token_ids,
        "bonus_token_id": analysis.bonus_token_id,
        "bonus_token": decode_token_lossy(&tokenizer, analysis.bonus_token_id),
        "reconstructed_token_ids": &analysis.reconstructed_token_ids,
        "draft_text": decode_tokens(&tokenizer, &draft_tokens)?,
        "accepted_text": decode_tokens(&tokenizer, &accepted_token_ids)?,
        "reconstructed_text": decode_tokens(&tokenizer, &analysis.reconstructed_token_ids)?,
        "baseline_token_ids": baseline_tokens,
        "baseline_prefix_match": baseline_prefix_match,
        "expected_rejection_index": expected_rejection_index,
        "rejection_matched_expectation": rejection_matched_expectation,
        "corruption": corruption.map(|corruption| serde_json::json!({
            "index": corruption.index,
            "original_token_id": corruption.original_token_id,
            "corrupted_token_id": corruption.corrupted_token_id,
            "original_token": decode_token_lossy(&tokenizer, corruption.original_token_id),
            "corrupted_token": decode_token_lossy(&tokenizer, corruption.corrupted_token_id),
        })),
        "positions": positions,
    });

    let failed_expectation = baseline_prefix_match == Some(false)
        || rejection_matched_expectation == Some(false)
        || (args.fail_on_mismatch && analysis.first_rejected_index.is_some());
    write_json_report(args.output.as_ref(), &report)?;
    if failed_expectation {
        anyhow::bail!("speculative verifier expectation failed");
    }
    Ok(())
}

pub(crate) fn dspark_drafter_parity(args: DsparkDrafterParityArgs) -> Result<()> {
    use lmbrrr::dspark::DsparkDrafter;

    let device = select_device(args.cpu)?;
    let dtype = if device.is_cpu() { DType::F32 } else { DType::BF16 };
    let runtime = lmbrrr::runtime_config::RuntimeConfig::from_env();
    let mut drafter =
        DsparkDrafter::load(&args.checkpoint, &device, dtype, runtime.mm2d.clone())?;
    let gamma = drafter.config.block_size;

    let fixture = candle::safetensors::load(&args.fixture, &device)
        .with_context(|| format!("load fixture {}", args.fixture.display()))?;
    let ctx = fixture
        .get("target_hidden_states")
        .context("fixture missing target_hidden_states")?
        .clone();
    let draft_ids = fixture
        .get("draft_input_ids")
        .context("fixture missing draft_input_ids")?
        .to_dtype(DType::U32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<u32>()?;
    let anchor = draft_ids[0][0];
    let ctx_len = ctx.dim(1)?;

    drafter.append_context(&ctx, 0)?;
    let proposal = drafter.propose_with_diagnostics(anchor, ctx_len, gamma)?;

    let expected_tokens = fixture
        .get("sampled_tokens")
        .context("fixture missing sampled_tokens")?
        .to_dtype(DType::U32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<u32>()?[0]
        .clone();
    let expected_conf = fixture
        .get("confidence_logits")
        .context("fixture missing confidence_logits")?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?[0]
        .clone();

    let max_abs = |ours: &Tensor, name: &str| -> Result<f64> {
        let theirs = fixture
            .get(name)
            .with_context(|| format!("fixture missing {name}"))?;
        let diff = (ours.to_dtype(DType::F32)? - theirs.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        Ok(diff as f64)
    };
    let diag = |t: &Option<Tensor>, name: &str| -> Result<Tensor> {
        t.clone()
            .with_context(|| format!("diagnostics missing {name}"))
    };
    let hidden_diff = max_abs(&diag(&proposal.block_hidden, "block_hidden")?, "block_hidden")?;
    let base_diff = max_abs(&diag(&proposal.base_logits, "base_logits")?, "base_logits")?;
    let corrected_diff = max_abs(
        &diag(&proposal.corrected_logits, "corrected_logits")?,
        "corrected_logits",
    )?;

    let token_matches = proposal
        .tokens
        .iter()
        .zip(expected_tokens.iter())
        .filter(|(a, b)| a == b)
        .count();
    let conf_max_diff = proposal
        .confidence_logits
        .iter()
        .zip(expected_conf.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let passed = token_matches == gamma && hidden_diff < 0.25 && conf_max_diff < 0.25;
    let report = serde_json::json!({
        "kind": "lmbrrr_dspark_drafter_parity",
        "schema_version": 1,
        "checkpoint": args.checkpoint,
        "fixture": args.fixture,
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "gamma": gamma,
        "ctx_len": ctx_len,
        "sampled_tokens": proposal.tokens,
        "expected_tokens": expected_tokens,
        "token_matches": token_matches,
        "max_abs_block_hidden_diff": hidden_diff,
        "max_abs_base_logits_diff": base_diff,
        "max_abs_corrected_logits_diff": corrected_diff,
        "confidence_logits": proposal.confidence_logits,
        "expected_confidence_logits": expected_conf,
        "max_abs_confidence_diff": conf_max_diff,
        "passed": passed,
    });
    write_json_report(args.output.as_ref(), &report)?;
    if !passed {
        anyhow::bail!("drafter parity failed");
    }
    Ok(())
}

pub(crate) fn tree_check(args: TreeCheckArgs) -> Result<()> {
    let w = args.branch_width;
    if w == 0 || w > 5 {
        anyhow::bail!("--branch-width must be in 1..=5 (flattened chunk must fit the l <= 12 kernel)");
    }
    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let (mut model, _, _) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
        args.model.quantize_lm_head,
        &lmbrrr::runtime_config::RuntimeConfig::from_env(),
    )?;

    let prompt_text = chat_prompt(&args.prompt, 0, false);
    let prompt_tokens = tokenize_prompt(&tokenizer, prompt_text)?;
    let prompt_input = Tensor::from_slice(&prompt_tokens, (1, prompt_tokens.len()), &device)?;
    let prefill_logits = model.forward(
        &prompt_input,
        None::<&ProcessedImages>,
        &args.model.downsample_mode,
        0,
    )?;
    let top2 = |logits: &Tensor| -> Result<(u32, u32)> {
        let v = logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let mut best = (0usize, f32::NEG_INFINITY);
        let mut second = (0usize, f32::NEG_INFINITY);
        for (i, &x) in v.iter().enumerate() {
            if x > best.1 {
                second = best;
                best = (i, x);
            } else if x > second.1 {
                second = (i, x);
            }
        }
        Ok((best.0 as u32, second.0 as u32))
    };
    let max_abs_delta = |a: &Tensor, b: &Tensor| -> Result<f32> {
        let d = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        Ok(d)
    };
    let forward_chain = |model: &mut MiniCpmForConditionalGeneration,
                         tokens: &[u32],
                         offset: usize|
     -> Result<Tensor> {
        let input = Tensor::from_slice(tokens, (1, tokens.len()), &device)?;
        Ok(model.forward_all_logits(
            &input,
            None::<&ProcessedImages>,
            &args.model.downsample_mode,
            offset,
        )?)
    };

    let (mut anchor, _) = top2(&prefill_logits)?;
    let mut offset = prompt_tokens.len();
    let mut worst_main = 0f32;
    let mut worst_alt = 0f32;
    let mut worst_rollback = 0f32;
    let probe_token = prompt_tokens[prompt_tokens.len() / 2];

    for round in 0..args.rounds {
        let snapshot = model.snapshot_decode_state();

        // Greedy main branch a_1..a_w and the runner-up alternate root b_1.
        let mut a_tokens = Vec::with_capacity(w);
        let mut b_tokens = Vec::with_capacity(w);
        let mut cur = anchor;
        let mut b_root = 0u32;
        for i in 0..w {
            let logits = forward_chain(&mut model, &[cur], offset + i)?
                .narrow(1, 0, 1)?
                .squeeze(1)?;
            let (best, second) = top2(&logits)?;
            if i == 0 {
                b_root = second;
            }
            a_tokens.push(best);
            cur = best;
        }
        // Alternate branch: runner-up root, then that path's own greedy
        // continuation (built as a chain from the snapshot).
        model.restore_decode_state(&snapshot)?;
        b_tokens.push(b_root);
        let mut cur = b_root;
        let _ = forward_chain(&mut model, &[anchor], offset)?;
        for i in 1..w {
            let logits = forward_chain(&mut model, &[cur], offset + i)?
                .narrow(1, 0, 1)?
                .squeeze(1)?;
            let (best, _) = top2(&logits)?;
            b_tokens.push(best);
            cur = best;
        }

        // Chain references over the exact tree tokens.
        model.restore_decode_state(&snapshot)?;
        let mut chain_a = vec![anchor];
        chain_a.extend(&a_tokens);
        let ref_a = forward_chain(&mut model, &chain_a, offset)?;
        model.restore_decode_state(&snapshot)?;
        let mut chain_b = vec![anchor];
        chain_b.extend(&b_tokens);
        let ref_b = forward_chain(&mut model, &chain_b, offset)?;

        // Tree forward on the flattened layout.
        model.restore_decode_state(&snapshot)?;
        let mut flat = vec![anchor];
        flat.extend(&a_tokens);
        flat.extend(&b_tokens);
        let flat_input = Tensor::from_slice(&flat, (1, flat.len()), &device)?;
        let tree_logits = model.forward_tree_all_logits(&flat_input, offset, w)?;

        let d_main = max_abs_delta(
            &tree_logits.narrow(1, 0, w + 1)?,
            &ref_a.narrow(1, 0, w + 1)?,
        )?;
        let d_alt = max_abs_delta(
            &tree_logits.narrow(1, w + 1, w)?,
            &ref_b.narrow(1, 1, w)?,
        )?;
        worst_main = worst_main.max(d_main);
        worst_alt = worst_alt.max(d_alt);

        // Rollback probes: install each winner path and compare a probe
        // token's logits against the same state built as a plain chain.
        let p = w.div_ceil(2);
        for on_alt in [false, true] {
            model.restore_decode_state(&snapshot)?;
            let _ = model.forward_tree_all_logits(&flat_input, offset, w)?;
            model.rollback_tree(&snapshot, w, on_alt, p)?;
            let probe =
                forward_chain(&mut model, &[probe_token], offset + 1 + p)?;
            model.restore_decode_state(&snapshot)?;
            let mut chain = vec![anchor];
            chain.extend(if on_alt { &b_tokens } else { &a_tokens }[..p].iter().copied());
            let _ = forward_chain(&mut model, &chain, offset)?;
            let ref_probe =
                forward_chain(&mut model, &[probe_token], offset + 1 + p)?;
            let d = max_abs_delta(&probe, &ref_probe)?;
            worst_rollback = worst_rollback.max(d);
            println!(
                "round {round} rollback on_alt={on_alt} p={p}: max |Δlogit| {d:.4}"
            );
        }
        println!(
            "round {round}: main rows max |Δ| {d_main:.5}, alt rows max |Δ| {d_alt:.4} (tokens a={a_tokens:?} b={b_tokens:?})"
        );

        // Advance: commit the main chain and continue from its last logits.
        model.restore_decode_state(&snapshot)?;
        let ref_a = forward_chain(&mut model, &chain_a, offset)?;
        let last = ref_a.narrow(1, w, 1)?.squeeze(1)?;
        let (next, _) = top2(&last)?;
        offset += w + 1;
        anchor = next;
    }

    println!(
        "tree-check: worst main {worst_main:.5} (eps {}), worst alt {worst_alt:.4} (eps {}), worst rollback {worst_rollback:.4} (eps {})",
        args.main_eps, args.alt_eps, args.alt_eps
    );
    if worst_main > args.main_eps || worst_alt > args.alt_eps || worst_rollback > args.alt_eps {
        anyhow::bail!("tree-check FAILED");
    }
    println!("tree-check PASSED");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verifier_analysis_accepts_full_draft_and_bonus() {
        let analysis = analyze_verification(&[10, 11, 12], &[10, 11, 12], 13).unwrap();

        assert_eq!(analysis.accepted_tokens, 3);
        assert_eq!(analysis.first_rejected_index, None);
        assert_eq!(analysis.bonus_token_id, 13);
        assert_eq!(analysis.reconstructed_token_ids, [10, 11, 12, 13]);
        assert_eq!(analysis.verifier_waste_tokens(), 0);
        assert_eq!(analysis.acceptance_rate(), Some(1.0));
    }

    #[test]
    fn verifier_analysis_rejects_at_first_mismatch() {
        let analysis = analyze_verification(&[10, 99, 12, 13], &[10, 11, 55, 56], 57).unwrap();

        assert_eq!(analysis.accepted_tokens, 1);
        assert_eq!(analysis.first_rejected_index, Some(1));
        assert_eq!(analysis.bonus_token_id, 11);
        assert_eq!(analysis.reconstructed_token_ids, [10, 11]);
        assert_eq!(analysis.verifier_waste_tokens(), 2);
        assert_eq!(analysis.verifier_waste_share(), Some(0.5));
        assert!(analysis.positions[1].first_rejected);
        assert!(!analysis.positions[2].accepted);
    }

    #[test]
    fn confidence_scheduler_truncates_before_threshold_drop() {
        let (len, cumulative, next) = schedule_prefix_len(&[0.9, 0.9, 0.9], 0.75);

        assert_eq!(len, 2);
        assert!((cumulative - 0.81).abs() < 1e-9);
        assert!(next.is_some_and(|value| (value - 0.729).abs() < 1e-9));
    }

    #[test]
    fn confidence_scheduler_can_drop_all_tokens() {
        let (len, cumulative, next) = schedule_prefix_len(&[0.7, 0.9], 0.8);

        assert_eq!(len, 0);
        assert_eq!(cumulative, 1.0);
        assert_eq!(next, Some(0.7));
    }

    #[test]
    fn draft_corruption_changes_selected_token() {
        let mut draft = vec![7, 8, 9];
        let corruption = corrupt_draft_token(&mut draft, 1, 16).unwrap();

        assert_eq!(corruption.index, 1);
        assert_eq!(corruption.original_token_id, 8);
        assert_eq!(corruption.corrupted_token_id, 9);
        assert_eq!(draft, [7, 9, 9]);
    }

}
