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

/// Multi-round speculative decoding with the trained Candle drafter inside
/// the rollback-verified loop. Context updates use on-device capture of the
/// target's capture layers: the verify chunk's captured states for the
/// anchor + accepted drafts are valid regardless of rollback (they were
/// computed under correct state), so they extend the drafter context each
/// round.
fn dspark_drafter_run(args: &DsparkRunArgs, drafter_dir: &Path) -> Result<()> {
    use lmbrrr::dspark::DsparkDrafter;

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

    // The drafter dir is an artifact bundle (same pattern as sts.json):
    // draft_vocab.json and cost_model.json auto-load when present so a
    // deployed drafter is one directory, not a flag soup. Explicit flags
    // override.
    let draft_vocab_path = args.draft_vocab.clone().or_else(|| {
        let bundled = drafter_dir.join("draft_vocab.json");
        bundled.exists().then_some(bundled)
    });
    let draft_vocab_ids: Option<Vec<u32>> = match &draft_vocab_path {
        None => None,
        Some(path) => {
            #[derive(serde::Deserialize)]
            struct DraftVocabFile {
                ids: Vec<u32>,
            }
            let file = std::fs::File::open(path)
                .with_context(|| format!("open draft vocab {}", path.display()))?;
            let parsed: DraftVocabFile = serde_json::from_reader(file)
                .with_context(|| format!("parse draft vocab {}", path.display()))?;
            Some(parsed.ids)
        }
    };
    let mut drafter = DsparkDrafter::load_with_draft_vocab(
        drafter_dir,
        &device,
        dtype,
        args.drafter_quantize.map(DrafterQuantArg::ggml),
        draft_vocab_ids.as_deref(),
    )?;
    let gamma = args.gamma.min(drafter.config.block_size);
    let capture_layers = drafter.config.target_layer_ids.clone();

    // Config artifacts load before any timed window opens.
    let sts = StsCalibration::load(drafter_dir)?;
    let cost_model_path = args.cost_model.clone().or_else(|| {
        let bundled = drafter_dir.join("cost_model.json");
        bundled.exists().then_some(bundled)
    });
    let mut cost_model = match &cost_model_path {
        Some(path) => RoundCostModel::load(path)?,
        None => RoundCostModel::measured_default(),
    };
    if let Some(fixed_ms) = args.cost_model_fixed_ms {
        cost_model.fixed_ms = fixed_ms;
    }

    // Untimed warmup: the first forwards of the process pay pipeline-state
    // creation and Metal heap growth; without this the greedy baseline runs
    // cold while the spec loop runs warm (behind the baseline), biasing the
    // in-report speedup comparison pro-spec.
    generate_tokens(
        &mut model,
        &device,
        &greedy_generation_args(8, args.enable_thinking),
        &prompt_tokens,
        None::<&ProcessedImages>,
        &args.model.downsample_mode,
        &eos_ids,
        |_, _, _, _| Ok(()),
    )?;

    // Greedy baseline for speed comparison and advisory text check.
    let baseline_start = Instant::now();
    let baseline = generate_tokens(
        &mut model,
        &device,
        &greedy_generation_args(args.max_new_tokens, args.enable_thinking),
        &prompt_tokens,
        None::<&ProcessedImages>,
        &args.model.downsample_mode,
        &eos_ids,
        |_, _, _, _| Ok(()),
    )?;
    let baseline_wall = secs(baseline_start.elapsed());

    let wall_start = Instant::now();
    model.clear_cache();
    model.set_device_capture(Some(capture_layers));
    drafter.clear_context();

    let prompt_input = Tensor::from_slice(&prompt_tokens, (1, prompt_tokens.len()), &device)?;
    let prefill_start = Instant::now();
    let prompt_logits =
        model.forward(&prompt_input, None::<&ProcessedImages>, &args.model.downsample_mode, 0)?;
    device.synchronize()?;
    let prefill_seconds = secs(prefill_start.elapsed());
    let captures = model.take_device_captures();
    let capture_refs = captures.iter().collect::<Vec<_>>();
    let ctx = Tensor::cat(&capture_refs, D::Minus1)?;
    drafter.append_context(&ctx, 0)?;
    let (first_token, _) = argmax_token(&prompt_logits, &device)?;
    model.set_verify_state_capture(!readvance_rollback());

    let mut committed = vec![first_token];
    let mut anchor = first_token;
    let mut start = prompt_tokens.len();
    let mut rounds = 0usize;
    let mut rollbacks = 0usize;
    let mut accepted_histogram = vec![0usize; gamma + 1];
    let mut position_proposed = vec![0usize; gamma];
    let mut position_accepted = vec![0usize; gamma];
    // Per ROUND, per verified position: (position index, raw confidence
    // logit, calibrated p, accepted). Round grouping preserves the prefix
    // structure the cumulative-survival calibration fits on.
    let mut confidence_records: Vec<Vec<(usize, f32, f32, bool)>> = Vec::new();
    // Full gamma-length confidence vector + chosen width per DRAFTED round,
    // in round order. confidence_records truncates at accepted+1 (the
    // calibration population), so offline scheduling studies (e.g. the
    // one-round-lag EV probe for on-device chunk assembly) need the raw
    // vectors the scheduler actually saw.
    let mut proposal_confidence_log: Vec<(Vec<f32>, usize)> = Vec::new();
    // Last drafted round's full confidence vector — the width prior for
    // --lag-schedule rounds (one-round-lag scheduling; EV probe on the
    // ticket: 68.8% width agreement, ~2% mean regret vs fresh confidences).
    let mut prev_draft_confidences: Option<Vec<f32>> = None;
    let mut proposed_width_histogram = vec![0usize; gamma + 1];
    let mut draft_seconds = 0.0f64;
    let mut verify_seconds = 0.0f64;
    let mut readvance_seconds = 0.0f64;

    // Greedy-fallback hysteresis: when the scheduler keeps choosing width 0
    // (speculation structurally unprofitable at current costs, e.g. the
    // quantized target's 3x chunk intercept), stop paying for drafts and
    // probe again periodically instead of degrading to greedy-minus-draft.
    let mut consecutive_zero_widths = 0usize;
    let mut skipped_drafts = 0usize;
    let mut tree_rounds = 0usize;
    let mut alt_wins = 0usize;
    let mut alt_tokens_gained = 0usize;
    let skip_draft_after = args.skip_draft_after;
    let probe_every = args.probe_every.max(1);

    // Prompt-lookup index over prompt + committed tokens; synced at the top
    // of every round so all round types (drafter/tree/lookup) feed it.
    let pld_span = args.pld_span.min(11);
    let mut ngram_index = lmbrrr::ngram_draft::NgramDraftIndex::new(args.pld_min_ngram, 4);
    if args.pld {
        ngram_index.extend(&prompt_tokens);
    }
    let mut pld_rounds = 0usize;
    let mut pld_proposed_tokens = 0usize;
    let mut pld_accepted_tokens = 0usize;
    let mut recycle_table = lmbrrr::token_recycle::RecycleTable::new();
    let mut recycle_rounds = 0usize;
    let mut recycle_proposed_tokens = 0usize;
    let mut recycle_accepted_tokens = 0usize;

    // Per-round host-cost residuals: round wall time minus the cost model's
    // kernel-time prediction for that round's shape. The round boundary is a
    // natural sync (the targets readback), so no extra synchronize() taints
    // the measurement. The median over drafter rounds is the fixed_round_ms
    // the scheduler contract needs; tree rounds are excluded (two segment
    // dispatches per layer — not represented by the chunk table).
    // (scheduled width, draft ran, residual ms, raw round wall ms); width is
    // the verify chunk len minus 1 for copy rounds. Raw walls make residual
    // reanalysis under a different cost model free — residuals alone are
    // relative to the model that happened to be loaded for this run.
    let mut drafter_round_residuals_ms: Vec<(usize, bool, f64, f64)> = Vec::new();
    let mut copy_round_residuals_ms: Vec<(usize, bool, f64, f64)> = Vec::new();

    // Spec-decode span: everything after prefill + drafter-context setup.
    // committed[0] came from the prefill logits, so the decode-rate
    // denominator pairs with committed.len() - 1 tokens.
    let decode_wall_start = Instant::now();
    if !eos_ids.contains(&first_token) {
        while committed.len() < args.max_new_tokens {
            let round_start = Instant::now();
            // Keep the lookup index in sync with everything committed so
            // far, regardless of which round type committed it.
            if args.pld {
                let indexed = ngram_index.len() - prompt_tokens.len();
                if committed.len() > indexed {
                    ngram_index.extend(&committed[indexed..]);
                }
            }

            // Copy-draft round (prompt-lookup, then token recycling): a
            // zero-draft-cost proposal verified as the same exact-argmax
            // chunk. Gated by default to rounds where the scheduler has
            // judged drafting unprofitable (skip-hysteresis engaged) —
            // firing on every match preempts strong drafter rounds and
            // loses (measured -13% on math). Ungated is available for
            // ablation. Lookup outranks recycling: contextual verbatim
            // copies beat statistical table guesses.
            let copy_gate_open = (args.pld || args.recycle)
                && (args.pld_ungated
                    || !args.schedule
                    || consecutive_zero_widths >= skip_draft_after);
            let copy_draft: Option<(Vec<u32>, bool)> = if copy_gate_open {
                let from_pld = if args.pld {
                    ngram_index.propose(pld_span)
                } else {
                    None
                };
                match from_pld {
                    Some(draft) => Some((draft, false)),
                    None if args.recycle => recycle_table
                        .propose(anchor, args.recycle_depth, args.recycle_margin)
                        .map(|draft| (draft, true)),
                    None => None,
                }
            } else {
                None
            };
            {
                if let Some((pld_draft, from_recycle)) = copy_draft {
                    let w = pld_draft.len();
                    let snapshot = model.snapshot_decode_state();
                    let mut chunk = Vec::with_capacity(w + 1);
                    chunk.push(anchor);
                    chunk.extend_from_slice(&pld_draft);
                    let chunk_input = Tensor::from_slice(&chunk, (1, chunk.len()), &device)?;
                    let verify_start = Instant::now();
                    let logits = model.forward_all_logits(
                        &chunk_input,
                        None::<&ProcessedImages>,
                        &args.model.downsample_mode,
                        start,
                    )?;
                    if loop_timing() {
                        device.synchronize()?;
                    }
                    verify_seconds += secs(verify_start.elapsed());
                    let targets: Vec<u32> = if args.recycle {
                        let summary = logits_argmax_and_topk(&logits, args.recycle_topk)?;
                        for (i, (_, candidates)) in summary.iter().enumerate() {
                            recycle_table.update(chunk[i], candidates);
                        }
                        summary.into_iter().map(|(argmax, _)| argmax).collect()
                    } else {
                        argmax_tokens(&logits, &device)?.0
                    };
                    let chunk_captures = model.take_device_captures();

                    let accepted = pld_draft
                        .iter()
                        .zip(targets.iter())
                        .take_while(|(draft, target)| draft == target)
                        .count();
                    let bonus = targets[accepted];

                    let capture_refs = chunk_captures.iter().collect::<Vec<_>>();
                    let chunk_ctx = Tensor::cat(&capture_refs, D::Minus1)?;
                    drafter.append_context(&chunk_ctx.narrow(1, 0, accepted + 1)?, start)?;

                    if accepted == w {
                        start += w + 1;
                    } else {
                        rollbacks += 1;
                        let readvance_start = Instant::now();
                        if readvance_rollback() {
                            model.restore_decode_state(&snapshot)?;
                            let readvance = &chunk[..accepted + 1];
                            let readvance_input =
                                Tensor::from_slice(readvance, (1, readvance.len()), &device)?;
                            let _ = model.forward_all_logits(
                                &readvance_input,
                                None::<&ProcessedImages>,
                                &args.model.downsample_mode,
                                start,
                            )?;
                            device.synchronize()?;
                            let _ = model.take_device_captures();
                        } else {
                            model.rollback_to_prefix(&snapshot, accepted + 1)?;
                            if loop_timing() {
                                device.synchronize()?;
                            }
                        }
                        readvance_seconds += secs(readvance_start.elapsed());
                        start += accepted + 1;
                    }

                    let round_wall_ms = secs(round_start.elapsed()) * 1000.0;
                    copy_round_residuals_ms.push((
                        w,
                        false,
                        round_wall_ms - cost_model.verify_kernel_ms(w + 1),
                        round_wall_ms,
                    ));
                    committed.extend_from_slice(&pld_draft[..accepted]);
                    committed.push(bonus);
                    rounds += 1;
                    if from_recycle {
                        recycle_rounds += 1;
                        recycle_proposed_tokens += w;
                        recycle_accepted_tokens += accepted;
                    } else {
                        pld_rounds += 1;
                        pld_proposed_tokens += w;
                        pld_accepted_tokens += accepted;
                    }
                    anchor = bonus;

                    if committed[committed.len() - (accepted + 1)..]
                        .iter()
                        .any(|token| eos_ids.contains(token))
                    {
                        if let Some(eos_at) =
                            committed.iter().position(|token| eos_ids.contains(token))
                        {
                            committed.truncate(eos_at + 1);
                        }
                        break;
                    }
                    continue;
                }
            }

            let skip_draft = args.schedule
                && consecutive_zero_widths >= skip_draft_after
                && (rounds % probe_every) != 0;
            // One-round-lag rounds keep the proposal on device: width comes
            // from the PREVIOUS drafted round's confidences, the verify
            // chunk is assembled device-side, and the proposal readback
            // rides the verify drain (2 pipeline drains per round -> 1).
            // The first drafted round has no lag prior and stays
            // synchronous, as does everything under --tree (alt chains need
            // host tokens before verify).
            let lag_round = args.lag_schedule
                && args.schedule
                && !args.tree
                && !skip_draft
                && prev_draft_confidences.is_some();
            let (mut proposal, device_proposal) = if skip_draft {
                skipped_drafts += 1;
                (None, None)
            } else if lag_round {
                let draft_start = Instant::now();
                let dp = drafter.propose_device(anchor, start, gamma)?;
                if loop_timing() {
                    device.synchronize()?;
                }
                draft_seconds += secs(draft_start.elapsed());
                (None, Some(dp))
            } else {
                let draft_start = Instant::now();
                let p = if args.tree {
                    drafter.propose_branching(anchor, start, gamma)?
                } else {
                    drafter.propose(anchor, start, gamma)?
                };
                if loop_timing() {
                    device.synchronize()?;
                }
                draft_seconds += secs(draft_start.elapsed());
                (Some(p), None)
            };

            // Width selection: the Appendix-A scheduler when --schedule,
            // else static confidence truncation (floored at 1: a width-0
            // round pays the full draft for one committed token), else full
            // gamma. Lag rounds schedule from the previous drafted round's
            // vector — the proposal's own confidences are still on device.
            let width = match (&proposal, &device_proposal) {
                (None, Some(_)) => schedule_prefix_width(
                    prev_draft_confidences
                        .as_ref()
                        .expect("lag rounds require a previous confidence vector")
                        .iter()
                        .enumerate()
                        .map(|(pos, logit)| sts.position_probability(pos, *logit) as f64),
                    |w| cost_model.t_round_ms(w),
                    gamma,
                ),
                (None, None) => 0,
                (Some(proposal), _) if args.schedule => schedule_prefix_width(
                    proposal
                        .confidence_logits
                        .iter()
                        .enumerate()
                        .map(|(pos, logit)| sts.position_probability(pos, *logit) as f64),
                    |w| cost_model.t_round_ms(w),
                    gamma,
                ),
                (Some(proposal), _) => match args.confidence_threshold {
                    Some(threshold) => proposal
                        .confidence_logits
                        .iter()
                        .take_while(|logit| sts.probability(**logit) >= threshold)
                        .count()
                        .max(1),
                    None => gamma,
                },
            };
            // Hysteresis bookkeeping for zero widths happens here; the reset
            // for nonzero widths is evidence-based and happens after verify:
            // a width>0 round whose draft is fully rejected is a realized
            // zero and must count toward skip mode, not reset it. (Measured
            // on a weak-drafter class: schedule-time resets turned 12
            // fully-rejected rounds into 2x drafter invocations, because
            // every reset buys >=skip_draft_after more probes.)
            if args.schedule && !skip_draft && width == 0 {
                consecutive_zero_widths += 1;
            }
            if let Some(p) = &proposal {
                proposal_confidence_log.push((p.confidence_logits.clone(), width));
            }
            proposed_width_histogram[width] += 1;
            let draft_tokens: &[u32] = proposal.as_ref().map_or(&[], |p| &p.tokens);
            let draft_confidences: &[f32] =
                proposal.as_ref().map_or(&[], |p| &p.confidence_logits);

            // Two-branch tree round: verify [anchor, a_1..a_w, b_1..b_w] in
            // one flattened forward and commit the longer-accepted path. Only
            // worth branching when the runner-up is live (distinct token, and
            // position-0 survival inside the configured band).
            let tree_width = width.min(5);
            let tree_round = args.tree
                && tree_width >= 1
                && proposal.as_ref().is_some_and(|p| {
                    p.alt_tokens.len() >= tree_width && p.alt_tokens[0] != p.tokens[0]
                })
                && draft_confidences
                    .first()
                    .map(|logit| sts.position_probability(0, *logit) as f32)
                    .is_some_and(|p0| p0 >= args.tree_band[0] && p0 <= args.tree_band[1]);
            if tree_round {
                let w = tree_width;
                let p = proposal.as_ref().expect("tree round requires a proposal");
                let a = &p.tokens[..w];
                let b = &p.alt_tokens[..w];
                let snapshot = model.snapshot_decode_state();
                let mut flat = Vec::with_capacity(1 + 2 * w);
                flat.push(anchor);
                flat.extend_from_slice(a);
                flat.extend_from_slice(b);
                let flat_input = Tensor::from_slice(&flat, (1, flat.len()), &device)?;
                let verify_start = Instant::now();
                let logits = model.forward_tree_all_logits(&flat_input, start, w)?;
                if loop_timing() {
                    device.synchronize()?;
                }
                verify_seconds += secs(verify_start.elapsed());
                let (targets, _) = argmax_tokens(&logits, &device)?;
                let chunk_captures = model.take_device_captures();

                let main_accepted = a
                    .iter()
                    .zip(targets[..w].iter())
                    .take_while(|(draft, target)| draft == target)
                    .count();
                // The alternate root is checked against the same anchor-row
                // target; its continuation rows sit after the main branch's.
                let alt_accepted = if targets[0] == b[0] {
                    1 + b[1..]
                        .iter()
                        .zip(targets[w + 1..].iter())
                        .take_while(|(draft, target)| draft == target)
                        .count()
                } else {
                    0
                };
                let on_alt = alt_accepted > main_accepted;
                let accepted = main_accepted.max(alt_accepted);
                if args.schedule {
                    if accepted == 0 {
                        consecutive_zero_widths += 1;
                    } else {
                        consecutive_zero_widths = 0;
                    }
                }
                let winner: &[u32] = if on_alt { b } else { a };
                let bonus_row = if on_alt { w + alt_accepted } else { main_accepted };
                let bonus = targets[bonus_row];

                // Calibration records stay on the main chain (the fit's
                // population); tau_eff shows up in the accepted histogram.
                let mut round_records = Vec::new();
                for j in 0..w {
                    position_proposed[j] += 1;
                    if j < main_accepted {
                        position_accepted[j] += 1;
                    }
                    if j <= main_accepted {
                        let logit = draft_confidences[j];
                        round_records.push((j, logit, sts.probability(logit), j < main_accepted));
                    }
                }
                confidence_records.push(round_records);

                let capture_refs = chunk_captures.iter().collect::<Vec<_>>();
                let chunk_ctx = Tensor::cat(&capture_refs, D::Minus1)?;
                let ctx_rows = if on_alt {
                    Tensor::cat(
                        &[
                            chunk_ctx.narrow(1, 0, 1)?,
                            chunk_ctx.narrow(1, w + 1, accepted)?,
                        ],
                        1,
                    )?
                    .contiguous()?
                } else {
                    chunk_ctx.narrow(1, 0, accepted + 1)?
                };
                drafter.append_context(&ctx_rows, start)?;

                // Winner install is unconditional: even a full main accept
                // must drop the alternate's KV rows.
                model.rollback_tree(&snapshot, w, on_alt, accepted)?;
                if loop_timing() {
                    device.synchronize()?;
                }
                if accepted < w {
                    rollbacks += 1;
                }
                start += accepted + 1;
                committed.extend_from_slice(&winner[..accepted]);
                committed.push(bonus);
                accepted_histogram[accepted] += 1;
                rounds += 1;
                tree_rounds += 1;
                if on_alt {
                    alt_wins += 1;
                    alt_tokens_gained += alt_accepted.saturating_sub(main_accepted);
                }
                anchor = bonus;

                if committed[committed.len() - (accepted + 1)..]
                    .iter()
                    .any(|token| eos_ids.contains(token))
                {
                    if let Some(eos_at) =
                        committed.iter().position(|token| eos_ids.contains(token))
                    {
                        committed.truncate(eos_at + 1);
                    }
                    break;
                }
                continue;
            }

            let snapshot = model.snapshot_decode_state();
            let mut chunk = Vec::with_capacity(width + 1);
            chunk.push(anchor);
            if device_proposal.is_none() {
                // Lag rounds fill the draft suffix after materialization —
                // their ids are still on device here.
                chunk.extend_from_slice(&draft_tokens[..width]);
            }
            let chunk_input = match &device_proposal {
                // On-device chunk assembly: draft ids never visit the host
                // before verification.
                Some(dp) if width > 0 => {
                    let anchor_dev = Tensor::from_slice(&[anchor], 1, &device)?;
                    Tensor::cat(&[&anchor_dev, &dp.tokens_dev.narrow(0, 0, width)?], 0)?
                        .reshape((1, width + 1))?
                }
                _ => Tensor::from_slice(&chunk, (1, chunk.len()), &device)?,
            };
            let verify_start = Instant::now();
            let logits = model.forward_all_logits(
                &chunk_input,
                None::<&ProcessedImages>,
                &args.model.downsample_mode,
                start,
            )?;
            if loop_timing() {
                device.synchronize()?;
            }
            verify_seconds += secs(verify_start.elapsed());
            let mut lag_materialized: Option<(Vec<u32>, Vec<f32>)> = None;
            let targets: Vec<u32> = if let Some(dp) = &device_proposal {
                // The round's single drain: verify targets and the packed
                // proposal confidences/ids ride one readback (ids < 2^24
                // are exact in f32).
                let argmax_f32 = logits
                    .argmax(D::Minus1)?
                    .flatten_all()?
                    .to_dtype(DType::F32)?;
                let combined = Tensor::cat(&[&argmax_f32, &dp.packed], 0)?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?;
                let (target_vals, rest) = combined.split_at(width + 1);
                let (conf, token_vals) = rest.split_at(dp.gamma);
                lag_materialized = Some((
                    token_vals.iter().map(|v| *v as u32).collect(),
                    conf.to_vec(),
                ));
                target_vals.iter().map(|v| *v as u32).collect()
            } else if args.recycle && copy_gate_open {
                // Harvest candidates from drafter-round verifies too, but
                // only while the copy gate is open: the two-stage top-k costs
                // a second device round-trip per round (measured -2.1%/-3.5%
                // end-to-end with zero proposals fired), so rounds where the
                // scheduler considers drafting profitable skip the harvest.
                let summary = logits_argmax_and_topk(&logits, args.recycle_topk)?;
                for (i, (_, candidates)) in summary.iter().enumerate() {
                    recycle_table.update(chunk[i], candidates);
                }
                summary.into_iter().map(|(argmax, _)| argmax).collect()
            } else {
                argmax_tokens(&logits, &device)?.0
            };
            let chunk_captures = model.take_device_captures();
            if let Some((tokens, confidence_logits)) = lag_materialized {
                proposal_confidence_log.push((confidence_logits.clone(), width));
                proposal = Some(lmbrrr::dspark::DraftProposal {
                    tokens,
                    confidence_logits,
                    alt_tokens: Vec::new(),
                    alt_confidence_logits: Vec::new(),
                    block_hidden: None,
                    base_logits: None,
                    corrected_logits: None,
                });
            }
            // Rebind for the shared tail: lag rounds materialized their
            // proposal only now, and the host chunk needs its draft suffix
            // for the rollback/readvance paths.
            let draft_tokens: &[u32] = proposal.as_ref().map_or(&[], |p| &p.tokens);
            let draft_confidences: &[f32] =
                proposal.as_ref().map_or(&[], |p| &p.confidence_logits);
            if device_proposal.is_some() {
                chunk.extend_from_slice(&draft_tokens[..width]);
            }

            let accepted = match args.accept_margin {
                // Exact: draft must equal the target argmax (lossless greedy).
                None => draft_tokens[..width]
                    .iter()
                    .zip(targets.iter())
                    .take_while(|(draft, target)| draft == target)
                    .count(),
                // Typical: draft survives while its target logit is within
                // `margin` of the top logit. Committed tokens remain the
                // drafts, so outputs may legitimately differ from greedy.
                Some(margin) if width > 0 => {
                    let verify_logits = logits.narrow(1, 0, width)?;
                    let max_vals = verify_logits
                        .max(D::Minus1)?
                        .to_dtype(DType::F32)?
                        .squeeze(0)?
                        .to_vec1::<f32>()?;
                    let idx =
                        Tensor::from_slice(&draft_tokens[..width], (1, width, 1), &device)?;
                    let draft_vals = verify_logits
                        .gather(&idx, D::Minus1)?
                        .to_dtype(DType::F32)?
                        .squeeze(2)?
                        .squeeze(0)?
                        .to_vec1::<f32>()?;
                    draft_vals
                        .iter()
                        .zip(max_vals.iter())
                        .take_while(|(draft, top)| **draft >= **top - margin)
                        .count()
                }
                Some(_) => 0,
            };
            let bonus = targets[accepted];
            if args.schedule && width > 0 {
                // Rate-based hysteresis evidence: the width choice is made
                // with the draft cost sunk, so a round can be the best
                // in-round option yet still run below the greedy pace once
                // the draft is included — and a drafter strong enough to
                // keep width > 0 would otherwise never let skip mode engage
                // (measured: -7% on weak Spec-Bench classes). A round only
                // resets the counter when its realized rate, draft
                // included, beat a plain greedy step.
                let round_ms = cost_model.kernel_ms(width);
                let greedy_ms = cost_model.greedy_step_ms;
                if ((accepted + 1) as f64) * greedy_ms < round_ms {
                    consecutive_zero_widths += 1;
                } else {
                    consecutive_zero_widths = 0;
                }
            }
            let mut round_records = Vec::with_capacity(width.min(accepted + 1));
            for j in 0..width {
                position_proposed[j] += 1;
                if j < accepted {
                    position_accepted[j] += 1;
                }
                // A verified position is a labeled calibration sample; only
                // the first rejection is a true negative for prefix
                // acceptance, positions past it were never target-checked
                // against a correct prefix, so stop at accepted + 1.
                if j <= accepted {
                    let logit = draft_confidences[j];
                    round_records.push((j, logit, sts.probability(logit), j < accepted));
                }
            }
            confidence_records.push(round_records);

            // Drafter context grows by the anchor + accepted drafts; those
            // captured positions are valid regardless of rollback.
            let capture_refs = chunk_captures.iter().collect::<Vec<_>>();
            let chunk_ctx = Tensor::cat(&capture_refs, D::Minus1)?;
            drafter.append_context(&chunk_ctx.narrow(1, 0, accepted + 1)?, start)?;

            if accepted == width {
                start += width + 1;
            } else {
                rollbacks += 1;
                let readvance_start = Instant::now();
                if readvance_rollback() {
                    model.restore_decode_state(&snapshot)?;
                    let readvance = &chunk[..accepted + 1];
                    let readvance_input =
                        Tensor::from_slice(readvance, (1, readvance.len()), &device)?;
                    let _ = model.forward_all_logits(
                        &readvance_input,
                        None::<&ProcessedImages>,
                        &args.model.downsample_mode,
                        start,
                    )?;
                    device.synchronize()?;
                    let _ = model.take_device_captures();
                } else {
                    model.rollback_to_prefix(&snapshot, accepted + 1)?;
                    // No sync: the reconstruction orders behind the next
                    // round's work on the queue; only timing mode waits.
                    if loop_timing() {
                        device.synchronize()?;
                    }
                }
                readvance_seconds += secs(readvance_start.elapsed());
                start += accepted + 1;
            }

            let kernel_est_ms = if skip_draft {
                cost_model.verify_kernel_ms(1)
            } else {
                cost_model.kernel_ms(width)
            };
            let round_wall_ms = secs(round_start.elapsed()) * 1000.0;
            drafter_round_residuals_ms.push((
                width,
                !skip_draft,
                round_wall_ms - kernel_est_ms,
                round_wall_ms,
            ));
            if let Some(p) = &proposal {
                prev_draft_confidences = Some(p.confidence_logits.clone());
            }
            committed.extend_from_slice(&draft_tokens[..accepted]);
            committed.push(bonus);
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
    committed.truncate(args.max_new_tokens);
    let wall_seconds = secs(wall_start.elapsed());
    let decode_wall_seconds = secs(decode_wall_start.elapsed());
    model.set_device_capture(None);
    model.set_verify_state_capture(false);

    // Exact per-round committed tokens (accepted + bonus) from the histogram
    // plus lookup-round tokens (tracked separately: lookup spans exceed the
    // gamma-sized histogram); committed.len()/rounds counts the prefill token
    // and loses EOS-truncated tokens, a ~1/rounds bias the scheduler's
    // break-even margin can't afford.
    let drafter_committed: usize = accepted_histogram
        .iter()
        .enumerate()
        .map(|(accepted, count)| (accepted + 1) * count)
        .sum();
    let mean_tau = if rounds == 0 {
        0.0
    } else {
        (drafter_committed
            + pld_accepted_tokens
            + pld_rounds
            + recycle_accepted_tokens
            + recycle_rounds) as f64
            / rounds as f64
    };
    let advisory_prefix = baseline
        .generated_token_ids
        .iter()
        .zip(committed.iter())
        .take_while(|(a, b)| a == b)
        .count();

    let summarize_residuals = |values: &[(usize, bool, f64, f64)]| -> serde_json::Value {
        if values.is_empty() {
            return serde_json::Value::Null;
        }
        let mut sorted = values.iter().map(|(_, _, r, _)| *r).collect::<Vec<_>>();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let pick = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
        serde_json::json!({
            "count": sorted.len(),
            "median_ms": pick(0.5),
            "p10_ms": pick(0.1),
            "p90_ms": pick(0.9),
            "mean_ms": sorted.iter().sum::<f64>() / sorted.len() as f64,
            "samples": values.iter()
                .map(|(w, drafted, r, wall)| serde_json::json!([w, drafted, r, wall]))
                .collect::<Vec<_>>(),
        })
    };

    let report = serde_json::json!({
        "kind": "lmbrrr_dspark_drafter_run",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "drafter": drafter_dir,
        "draft_vocab_path": draft_vocab_path,
        "cost_model_path": cost_model_path,
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "prompt": args.prompt.as_str(),
        "prompt_tokens": prompt_tokens.len(),
        "gamma": gamma,
        "max_new_tokens": args.max_new_tokens,
        "committed_tokens": committed.len(),
        "rounds": rounds,
        "rollbacks": rollbacks,
        "mean_accepted_length": mean_tau,
        "accepted_histogram": accepted_histogram,
        "position_acceptance": position_proposed.iter().zip(position_accepted.iter())
            .map(|(p, a)| if *p > 0 { *a as f64 / *p as f64 } else { 0.0 })
            .collect::<Vec<_>>(),
        "confidence_threshold": args.confidence_threshold,
        "accept_margin": args.accept_margin,
        "acceptance_note": if args.accept_margin.is_some() {
            "typical acceptance: outputs may diverge from greedy; confidence_records reflect the relaxed rule (recalibrate before mixing with exact-rule fits)"
        } else {
            "exact argmax acceptance (lossless greedy)"
        },
        "sts_calibration": { "scale": sts.scale, "shift": sts.shift },
        "cost_model": {
            "fixed_round_ms": cost_model.fixed_ms,
            "default_draft_ms": cost_model.draft_ms,
            "greedy_step_ms": cost_model.greedy_step_ms,
            "verify_ms_by_chunk_len": cost_model.verify_ms.clone(),
        },
        // Round wall minus the model's kernel-time prediction; the drafter
        // median is the measured fixed_round_ms for this stack (tree rounds
        // excluded — two segment dispatches per layer are outside the chunk
        // table's contract).
        "round_residual_ms": {
            "drafter_rounds": summarize_residuals(&drafter_round_residuals_ms),
            "copy_rounds": summarize_residuals(&copy_round_residuals_ms),
        },
        "proposed_width_histogram": proposed_width_histogram,
        "skipped_drafts": skipped_drafts,
        "tree": args.tree,
        "tree_rounds": tree_rounds,
        "tree_alt_wins": alt_wins,
        "tree_alt_tokens_gained": alt_tokens_gained,
        "pld": args.pld,
        "pld_rounds": pld_rounds,
        "pld_proposed_tokens": pld_proposed_tokens,
        "pld_accepted_tokens": pld_accepted_tokens,
        "recycle": args.recycle,
        "recycle_rounds": recycle_rounds,
        "recycle_proposed_tokens": recycle_proposed_tokens,
        "recycle_accepted_tokens": recycle_accepted_tokens,
        "recycle_table_rows": recycle_table.len(),
        "recycle_table_updates": recycle_table.updates(),
        "confidence_records": confidence_records.iter()
            .map(|round| round.iter()
                .map(|(pos, logit, p, acc)| serde_json::json!([pos, logit, p, acc]))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        // Full per-drafted-round confidence vectors + chosen widths, in
        // round order (offline scheduling studies; see the declaration).
        "proposal_confidences": proposal_confidence_log.iter()
            .map(|(logits, width)| serde_json::json!({"logits": logits, "width": width}))
            .collect::<Vec<_>>(),
        "prefill_seconds": prefill_seconds,
        "draft_seconds": draft_seconds,
        "verify_seconds": verify_seconds,
        "readvance_seconds": readvance_seconds,
        "wall_seconds": wall_seconds,
        "decode_wall_seconds": decode_wall_seconds,
        // Decode-only rate: the spec-round span and the tokens it produced.
        // committed[0] is the prefill token — different classes have very
        // different prompt lengths, so folding prefill in here confounded
        // cross-class comparisons and moved with quantization via a channel
        // unrelated to speculative-round economics.
        "tokens_per_second": committed.len().saturating_sub(1) as f64
            / decode_wall_seconds.max(f64::EPSILON),
        // End-to-end rate a caller experiences for this request (prefill +
        // drafter setup + decode), the honest single-shot number.
        "effective_tokens_per_second": committed.len() as f64 / wall_seconds.max(f64::EPSILON),
        "provenance": {
            "lmbrrr_git_rev": env!("LMBRRR_GIT_REV"),
            "candle_pin": env!("LMBRRR_CANDLE_PIN"),
        },
        "baseline": {
            "generated_tokens": baseline.generated_token_ids.len(),
            "wall_seconds": baseline_wall,
            "decode_tokens_per_second": baseline.decode_tokens_per_second(),
            "steady_state_tokens_per_second": baseline.steady_state_tokens_per_second(),
        },
        "advisory_baseline_prefix_match": advisory_prefix,
        "committed_text": decode_tokens(&tokenizer, &committed)?,
        "break_even_note": "measured break-even is tau ~= 4-5 with per-round rollback (docs/research/speculative-state-rollback.md)",
    });
    write_json_report(args.output.as_ref(), &report)
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
