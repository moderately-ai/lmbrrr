//! The dspark-run command: stub-drafter state-rollback oracle (blocking
//! trajectory-invariance gate) and dispatch to the trained-drafter runner.

use crate::*;

#[derive(Clone, Debug)]
struct SpecStubRun {
    corrupt_every: usize,
    committed: Vec<u32>,
    /// Top-K logit values (descending) at the position that committed each
    /// token. The invariance gate compares these trajectories across
    /// corruption patterns: shared positions must agree within kernel noise
    /// (state-integrity check), and a token divergence is benign only when
    /// both runs' top-2 margins sit inside the noise (tie-flip).
    committed_top_k: Vec<Vec<f32>>,
    rounds: usize,
    rollbacks: usize,
    accepted_histogram: Vec<usize>,
    prefill_seconds: f64,
    verify_seconds: f64,
    readvance_seconds: f64,
    argmax_seconds: f64,
    wall_seconds: f64,
}

/// One full multi-round speculative pass with a stub drafter. Chunks follow
/// the DeepSpec convention: [anchor, d1..dw] is fed, the logits at position i
/// verify draft i+1, and the token after the last accepted draft is the bonus
/// (= next round's anchor). On partial acceptance the decode state is
/// restored from the pre-verify snapshot and the accepted prefix re-advanced
/// in one chunk; on full acceptance the advanced state is kept as-is.
#[allow(clippy::too_many_arguments)]
/// Top-K logit values per sequence position, descending. Logits may be [v],
/// [l, v] or [b, l, v]; the batch dim, when present, must be 1. CPU
/// reduction — oracle-mode only, never on the production path.
const ORACLE_TOP_K: usize = 8;
fn top_k_values(logits: &Tensor) -> Result<Vec<Vec<f32>>> {
    let logits = match logits.dims().len() {
        3 => logits.squeeze(0)?,
        1 => logits.unsqueeze(0)?,
        _ => logits.clone(),
    };
    let rows = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    Ok(rows
        .iter()
        .map(|row| {
            let mut top = [f32::NEG_INFINITY; ORACLE_TOP_K];
            for &v in row {
                if v > top[ORACLE_TOP_K - 1] {
                    let mut i = ORACLE_TOP_K - 1;
                    while i > 0 && v > top[i - 1] {
                        top[i] = top[i - 1];
                        i -= 1;
                    }
                    top[i] = v;
                }
            }
            top.to_vec()
        })
        .collect())
}

/// Logit-scale bound on legitimate chunk-split numerics. The target's logits
/// at a committed position depend only on the prefix, not on how verify
/// chunks split it, so across corruption patterns the top-K values at every
/// shared position must agree to within kernel noise — measured at ~3 BF16
/// ulps of a top logit near 32 (observed divergence-point margins 0.0 /
/// 0.25 / 0.375) — while a real rollback bug perturbs the whole trajectory.
/// Reports carry the observed maxima so this stays calibrated by evidence.
const LOGIT_NOISE_BOUND: f32 = 0.75;

fn dspark_stub_run(
    model: &mut MiniCpmForConditionalGeneration,
    device: &Device,
    prompt_tokens: &[u32],
    stub_tokens: &[u32],
    gamma: usize,
    max_new_tokens: usize,
    corrupt_every: usize,
    vocab_size: usize,
    downsample_mode: &str,
    eos_ids: &[u32],
) -> Result<SpecStubRun> {
    let wall_start = Instant::now();
    model.clear_cache();
    let prompt_input = Tensor::from_slice(prompt_tokens, (1, prompt_tokens.len()), device)?;
    let prefill_start = Instant::now();
    let prompt_logits =
        model.forward(&prompt_input, None::<&ProcessedImages>, downsample_mode, 0)?;
    device.synchronize()?;
    let prefill_seconds = secs(prefill_start.elapsed());
    let (first_token, mut argmax_elapsed) = argmax_token(&prompt_logits, device)?;
    model.set_verify_state_capture(!readvance_rollback());

    let mut committed = vec![first_token];
    let mut committed_top_k = top_k_values(&prompt_logits)?;
    let mut anchor = first_token;
    let mut offset = prompt_tokens.len();
    let mut rounds = 0usize;
    let mut rollbacks = 0usize;
    let mut accepted_histogram = vec![0usize; gamma + 1];
    let mut verify_seconds = 0.0f64;
    let mut readvance_seconds = 0.0f64;

    if !eos_ids.contains(&first_token) {
        while committed.len() < max_new_tokens {
            let available = stub_tokens.len().saturating_sub(committed.len());
            let width = gamma.min(available);
            let mut drafts =
                stub_tokens[committed.len()..committed.len() + width].to_vec();
            if corrupt_every > 0 {
                for (j, draft) in drafts.iter_mut().enumerate() {
                    let position = committed.len() + j;
                    if (position + 1) % corrupt_every == 0 {
                        let mut corrupted = (*draft + 1) % vocab_size as u32;
                        if eos_ids.contains(&corrupted) {
                            corrupted = (corrupted + 1) % vocab_size as u32;
                        }
                        *draft = corrupted;
                    }
                }
            }

            let snapshot = model.snapshot_decode_state();
            let mut chunk = Vec::with_capacity(width + 1);
            chunk.push(anchor);
            chunk.extend_from_slice(&drafts);
            let chunk_input = Tensor::from_slice(&chunk, (1, chunk.len()), device)?;
            let verify_start = Instant::now();
            let logits = model.forward_all_logits(
                &chunk_input,
                None::<&ProcessedImages>,
                downsample_mode,
                offset,
            )?;
            device.synchronize()?;
            verify_seconds += secs(verify_start.elapsed());
            let (targets, chunk_argmax) = argmax_tokens(&logits, device)?;
            argmax_elapsed += chunk_argmax;
            let chunk_top_k = top_k_values(&logits)?;

            let accepted = drafts
                .iter()
                .zip(targets.iter())
                .take_while(|(draft, target)| draft == target)
                .count();
            let bonus = targets[accepted];

            if accepted == width {
                offset += width + 1;
            } else {
                rollbacks += 1;
                let readvance_start = Instant::now();
                if readvance_rollback() {
                    model.restore_decode_state(&snapshot)?;
                    let readvance = &chunk[..accepted + 1];
                    let readvance_input =
                        Tensor::from_slice(readvance, (1, readvance.len()), device)?;
                    let _ = model.forward_all_logits(
                        &readvance_input,
                        None::<&ProcessedImages>,
                        downsample_mode,
                        offset,
                    )?;
                    device.synchronize()?;
                } else {
                    model.rollback_to_prefix(&snapshot, accepted + 1)?;
                    device.synchronize()?;
                }
                readvance_seconds += secs(readvance_start.elapsed());
                offset += accepted + 1;
            }

            committed.extend_from_slice(&drafts[..accepted]);
            committed.push(bonus);
            committed_top_k.extend_from_slice(&chunk_top_k[..=accepted]);
            accepted_histogram[accepted] += 1;
            rounds += 1;
            anchor = bonus;

            if committed[committed.len() - (accepted + 1)..]
                .iter()
                .any(|token| eos_ids.contains(token))
            {
                if let Some(eos_at) = committed.iter().position(|token| eos_ids.contains(token)) {
                    committed.truncate(eos_at + 1);
                }
                break;
            }
        }
    }
    committed.truncate(max_new_tokens);
    committed_top_k.truncate(committed.len());
    model.set_verify_state_capture(false);

    Ok(SpecStubRun {
        corrupt_every,
        committed,
        committed_top_k,
        rounds,
        rollbacks,
        accepted_histogram,
        prefill_seconds,
        verify_seconds,
        readvance_seconds,
        argmax_seconds: secs(argmax_elapsed),
        wall_seconds: secs(wall_start.elapsed()),
    })
}

pub(crate) fn dspark_run(args: DsparkRunArgs) -> Result<()> {
    if args.gamma == 0 {
        anyhow::bail!("--gamma must be greater than zero");
    }
    if let Some(drafter_dir) = args.drafter.clone() {
        return dspark_drafter_run(&args, &drafter_dir);
    }
    if args.corrupt_every.is_empty() {
        anyhow::bail!("provide at least one --corrupt-every value");
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
    )?;
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());
    let vocab_size = bundle.config.text_config.vocab_size;

    // Baseline greedy pass: stub source, advisory oracle, and the speed
    // comparator (chunk-path logits can tie-flip vs decode-path logits, so
    // equality with the baseline is advisory; run-invariance below blocks).
    let baseline_start = Instant::now();
    let baseline = generate_tokens(
        &mut model,
        &device,
        &greedy_generation_args(args.max_new_tokens + args.gamma + 8, args.enable_thinking),
        &prompt_tokens,
        None::<&ProcessedImages>,
        &args.model.downsample_mode,
        &eos_ids,
        |_, _, _, _| Ok(()),
    )?;
    let baseline_wall = secs(baseline_start.elapsed());
    let stub_tokens = baseline.generated_token_ids.clone();
    if stub_tokens.is_empty() {
        anyhow::bail!("baseline generation produced no tokens to drive the stub drafter");
    }

    let mut runs = Vec::with_capacity(args.corrupt_every.len());
    for &corrupt_every in &args.corrupt_every {
        runs.push(dspark_stub_run(
            &mut model,
            &device,
            &prompt_tokens,
            &stub_tokens,
            args.gamma,
            args.max_new_tokens,
            corrupt_every,
            vocab_size,
            &args.model.downsample_mode,
            &eos_ids,
        )?);
    }

    // BLOCKING oracle, state-integrity form. The target's logits at a
    // committed position depend only on the prefix, never on how verify
    // chunks split it, so across corruption patterns the top-K logit values
    // at every shared committed position must agree to within kernel noise
    // iff state rollback is sound — a real restore bug perturbs the whole
    // trajectory, argmax flip or not. A committed-token divergence is benign
    // only when both runs' top-2 margins sit inside the noise (a tie the
    // chunk-split numerics may legitimately flip; root-caused 2026-07-10 —
    // every kernel change re-rolls which prompts carry such ties, so bitwise
    // stream equality can never be the gate). Streams legitimately fork
    // after a benign tie, so comparison for that pair stops there.
    let reference = &runs[0];
    let mut invariance_passed = true;
    let mut first_divergence: Option<serde_json::Value> = None;
    let mut benign_tie_divergences: Vec<serde_json::Value> = Vec::new();
    let mut max_trajectory_deviation = 0.0f32;
    for run in &runs[1..] {
        let shared = reference
            .committed
            .iter()
            .zip(run.committed.iter())
            .position(|(a, b)| a != b);
        let shared_len =
            shared.unwrap_or_else(|| reference.committed.len().min(run.committed.len()));
        // State-integrity: top-K trajectories over the shared prefix.
        let mut worst: Option<(usize, f32)> = None;
        for i in 0..shared_len {
            let (Some(a), Some(b)) = (
                reference.committed_top_k.get(i),
                run.committed_top_k.get(i),
            ) else {
                continue;
            };
            let dev = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            max_trajectory_deviation = max_trajectory_deviation.max(dev);
            if dev > LOGIT_NOISE_BOUND && worst.map_or(true, |(_, w)| dev > w) {
                worst = Some((i, dev));
            }
        }
        if let Some((index, dev)) = worst {
            invariance_passed = false;
            first_divergence = Some(serde_json::json!({
                "kind": "trajectory",
                "corrupt_every": run.corrupt_every,
                "index": index,
                "top_k_deviation": dev,
                "logit_noise_bound": LOGIT_NOISE_BOUND,
            }));
            break;
        }
        // Token divergence at the end of the shared prefix: benign iff tie.
        let Some(index) = shared else { continue };
        let margin = |r: &SpecStubRun| {
            r.committed_top_k
                .get(index)
                .map(|top| top[0] - top[1])
        };
        let (ref_margin, run_margin) = (margin(reference), margin(run));
        let benign = matches!((ref_margin, run_margin), (Some(a), Some(b))
            if a <= LOGIT_NOISE_BOUND && b <= LOGIT_NOISE_BOUND);
        let detail = serde_json::json!({
            "kind": "token",
            "corrupt_every": run.corrupt_every,
            "index": index,
            "reference_len": reference.committed.len(),
            "run_len": run.committed.len(),
            "reference_margin": ref_margin,
            "run_margin": run_margin,
            "logit_noise_bound": LOGIT_NOISE_BOUND,
        });
        if benign {
            benign_tie_divergences.push(detail);
        } else {
            invariance_passed = false;
            first_divergence = Some(detail);
            break;
        }
    }

    let advisory_prefix = stub_tokens
        .iter()
        .zip(reference.committed.iter())
        .take_while(|(a, b)| a == b)
        .count();

    let report = serde_json::json!({
        "kind": "lmbrrr_dspark_stub_run",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "prompt": args.prompt.as_str(),
        "prompt_tokens": prompt_tokens.len(),
        "gamma": args.gamma,
        "max_new_tokens": args.max_new_tokens,
        "baseline": {
            "generated_tokens": stub_tokens.len(),
            "wall_seconds": baseline_wall,
            "decode_tokens_per_second": baseline.decode_tokens_per_second(),
            "steady_state_tokens_per_second": baseline.steady_state_tokens_per_second(),
        },
        "runs": runs.iter().map(|run| serde_json::json!({
            "corrupt_every": run.corrupt_every,
            "committed_tokens": run.committed.len(),
            "rounds": run.rounds,
            "rollbacks": run.rollbacks,
            "accepted_histogram": run.accepted_histogram,
            "mean_accepted_length": mean_committed_per_round(&run.accepted_histogram, run.rounds),
            "prefill_seconds": run.prefill_seconds,
            "verify_seconds": run.verify_seconds,
            "readvance_seconds": run.readvance_seconds,
            "argmax_seconds": run.argmax_seconds,
            "wall_seconds": run.wall_seconds,
            "tokens_per_second": run.committed.len() as f64 / run.wall_seconds.max(f64::EPSILON),
        })).collect::<Vec<_>>(),
        "invariance_oracle_passed": invariance_passed,
        "first_divergence": first_divergence,
        "benign_tie_divergences": benign_tie_divergences,
        "max_trajectory_deviation": max_trajectory_deviation,
        "logit_noise_bound": LOGIT_NOISE_BOUND,
        "advisory_baseline_prefix_match": advisory_prefix,
        "advisory_note": "prefix match vs decode-path baseline; tie-flips are expected occasionally, the blocking gate is run-invariance",
        "committed_text": decode_tokens(&tokenizer, &reference.committed)?,
    });
    write_json_report(args.output.as_ref(), &report)?;
    if !invariance_passed {
        anyhow::bail!("state-rollback invariance oracle failed");
    }
    Ok(())
}
