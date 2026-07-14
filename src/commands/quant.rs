//! Quantization tooling commands: sensitivity scan, mixed-precision
//! conversion, matmul micro-bench, and the quality gate harness.

use crate::*;

#[derive(Clone, Debug)]
struct QuantQualityGeneration {
    stats: GenerationStats,
    raw_text: String,
    reasoning_text: String,
    answer_text: String,
}

#[derive(Clone, Debug)]
struct QuantQualityPolicyRun {
    label: String,
    manifest: Option<PathBuf>,
    load_elapsed: Duration,
    run_elapsed: Duration,
    quantized_load: Option<QuantizedLoadStats>,
    generations: Vec<QuantQualityGeneration>,
}

#[derive(Clone, Debug)]
struct QualityThresholds {
    min_prefix_ratio: f64,
    min_token_jaccard: f64,
    min_lexical_jaccard: f64,
    max_length_ratio_delta: f64,
}

#[derive(Clone, Debug)]
struct QualityComparison {
    exact_token_match: bool,
    common_prefix_tokens: usize,
    divergence_index: Option<usize>,
    prefix_ratio: f64,
    token_jaccard: f64,
    lexical_jaccard: f64,
    length_ratio: f64,
    length_ratio_delta: f64,
    passed_gate: bool,
}

pub(crate) fn quant_sensitivity(args: QuantSensitivityArgs) -> Result<()> {
    if args.top_k_logits == 0 {
        anyhow::bail!("--top-k-logits must be greater than zero");
    }
    let formats = if args.candidate_quants.is_empty() {
        vec![
            QuantFormat::SymmetricInt4,
            QuantFormat::SymmetricInt5,
            QuantFormat::SymmetricInt8,
        ]
    } else {
        args.candidate_quants
            .iter()
            .map(|format| format.resolve())
            .collect::<Vec<_>>()
    };

    let calibration_rows = read_calibration_jsonl(&args.calibration)?;
    let text_rows = calibration_rows
        .iter()
        .filter(|row| row.modality == "text")
        .take(args.max_cases.unwrap_or(usize::MAX))
        .collect::<Vec<_>>();
    if text_rows.is_empty() {
        anyhow::bail!(
            "calibration file {} contains no text rows",
            args.calibration.display()
        );
    }

    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
        args.model.quantize_lm_head,
    )?;

    let baseline_started = Instant::now();
    let mut baseline_cases = Vec::with_capacity(text_rows.len());
    for row in text_rows {
        baseline_cases.push(run_quant_baseline_case(
            &mut model,
            &device,
            &tokenizer,
            row,
            &args.model.downsample_mode,
            args.top_k_logits,
        )?);
    }
    let baseline_elapsed = baseline_started.elapsed();

    let weight_report = score_weight_sensitivity(
        &bundle.artifacts.weights,
        &formats,
        args.max_modules,
        args.include_protected,
    )?;

    let report = serde_json::json!({
        "kind": "lmbrrr_quantization_sensitivity",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "downsample_mode": args.model.downsample_mode.as_str(),
        "calibration_set": args.calibration,
        "calibration": aggregate_calibration(&calibration_rows),
        "candidate_quants": formats.iter().map(|format| format.name()).collect::<Vec<_>>(),
        "include_protected": args.include_protected,
        "artifact_seconds": secs(bundle.elapsed),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "tensor_count": bundle.weight_report.tensor_count,
        "has_lm_head": bundle.weight_report.has_lm_head,
        "baseline": {
            "status": "measured",
            "rows_measured": baseline_cases.len(),
            "text_rows_available": calibration_rows.iter().filter(|row| row.modality == "text").count(),
            "seconds": secs(baseline_elapsed),
            "top_k_logits": args.top_k_logits,
            "cases": baseline_cases,
        },
        "weights": weight_report,
        "measurement_limits": {
            "activation_error": "not collected yet because MiniCPM module activation hooks are not implemented",
            "per_module_logit_drift": "not collected yet because this command does not run perturbed quantized module forwards",
            "latency_delta": "weight quantization simulation is timed; runtime matmul latency awaits quantized loader/kernel tickets",
        },
    });

    write_json_report(args.output.as_ref(), &report)
}

pub(crate) fn quant_convert(args: QuantConvertArgs) -> Result<()> {
    let bundle = resolve_artifacts(&args.model)?;
    let manifest = convert_mixed_precision(ConversionOptions {
        model_id: args.model.model_id.clone(),
        revision: args.model.revision.clone(),
        policy: args.policy.resolve(),
        source_weights: bundle.artifacts.weights.clone(),
        sensitivity_artifact: args.sensitivity.clone(),
        output_dir: args.output_dir.clone(),
        max_tensors: args.max_tensors,
        manifest_only: args.manifest_only,
        fallback_overrides: args
            .fallback
            .iter()
            .map(|spec| {
                spec.split_once('=')
                    .map(|(suffix, rung)| (suffix.to_string(), rung.to_string()))
                    .ok_or_else(|| anyhow::anyhow!("--fallback wants <suffix>=<rung>, got {spec}"))
            })
            .collect::<Result<Vec<_>>>()?,
    })?;
    let manifest_path = args.output_dir.join("manifest.json");
    let summary = serde_json::json!({
        "kind": "lmbrrr_quant_convert_complete",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "policy": args.policy.resolve().name(),
        "manifest": manifest_path,
        "artifact_seconds": secs(bundle.elapsed),
        "manifest_only": args.manifest_only,
        "summary": manifest["summary"].clone(),
    });
    write_json_report(None, &summary)
}

pub(crate) fn quant_matmul_bench(args: QuantMatmulBenchArgs) -> Result<()> {
    if args.chunk_tokens == 0 {
        anyhow::bail!("--chunk-tokens must be greater than zero");
    }
    if args.iterations == 0 {
        anyhow::bail!("--iterations must be greater than zero");
    }

    let capture_cell = args
        .gpu_capture_cell
        .as_deref()
        .map(parse_capture_cell)
        .transpose()?;
    if capture_cell.is_some() && std::env::var("METAL_CAPTURE_ENABLED").is_err() {
        anyhow::bail!(
            "--gpu-capture-cell needs METAL_CAPTURE_ENABLED=1 in the environment \
             (undocumented Metal requirement)"
        );
    }

    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let shapes = quant_matmul_shapes(&bundle.config, args.include_lm_head);
    let activation_dtypes = [DType::F32, DType::F16, DType::BF16];
    let quant_dtypes = [
        GgmlDType::Q8_0,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ];

    let token_counts = if args.token_counts.is_empty() {
        vec![1, args.chunk_tokens]
    } else {
        args.token_counts.clone()
    };
    if token_counts.contains(&0) {
        anyhow::bail!("--token-counts entries must be greater than zero");
    }

    let mut rows = Vec::new();
    for shape in shapes {
        let weight_values = deterministic_values(shape.out_dim * shape.in_dim, 0.013);
        let weight_cpu =
            Tensor::from_vec(weight_values, (shape.out_dim, shape.in_dim), &Device::Cpu)?;
        for &tokens in &token_counts {
            let input_values = deterministic_values(tokens * shape.in_dim, 0.017);
            let input_cpu =
                Tensor::from_vec(input_values, (1, tokens, shape.in_dim), &Device::Cpu)?;

            let ctx = MatmulBenchCtx {
                weight_cpu: &weight_cpu,
                input_cpu: &input_cpu,
                device: &device,
                warmup: args.warmup,
                iterations: args.iterations,
                capture: capture_cell.as_ref(),
            };
            for activation_dtype in activation_dtypes {
                rows.push(bench_dense_matmul(&shape, tokens, activation_dtype, &ctx));
            }

            for quant_dtype in quant_dtypes {
                for activation_dtype in activation_dtypes {
                    rows.push(bench_quant_matmul(
                        &shape,
                        tokens,
                        quant_dtype,
                        activation_dtype,
                        &ctx,
                    ));
                }
            }
        }
    }

    let report = serde_json::json!({
        "kind": "lmbrrr_quantized_matmul_benchmark",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "chunk_tokens": args.chunk_tokens,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "include_lm_head": args.include_lm_head,
        "artifact_seconds": secs(bundle.elapsed),
        "rows": rows,
        "note": "Dense baselines use generated weights with Candle matmul. Quantized rows use Candle QTensor::quantize_onto and QMatMul::forward; failures are recorded because activation dtype support is part of the measurement.",
    });
    write_json_report(args.output.as_ref(), &report)
}

pub(crate) fn quant_quality(args: QuantQualityArgs) -> Result<()> {
    if !is_greedy_generation(&args.generation) {
        anyhow::bail!("quant-quality requires greedy generation; leave --temperature at 0");
    }
    validate_quality_threshold("min-prefix-ratio", args.min_prefix_ratio, 0.0, 1.0)?;
    validate_quality_threshold("min-token-jaccard", args.min_token_jaccard, 0.0, 1.0)?;
    validate_quality_threshold("min-lexical-jaccard", args.min_lexical_jaccard, 0.0, 1.0)?;
    validate_quality_threshold(
        "max-length-ratio-delta",
        args.max_length_ratio_delta,
        0.0,
        10.0,
    )?;

    let thresholds = QualityThresholds {
        min_prefix_ratio: args.min_prefix_ratio,
        min_token_jaccard: args.min_token_jaccard,
        min_lexical_jaccard: args.min_lexical_jaccard,
        max_length_ratio_delta: args.max_length_ratio_delta,
    };
    let calibration_rows = read_calibration_jsonl(&args.calibration)?;
    let mut text_rows = calibration_rows
        .iter()
        .filter(|row| row.modality == "text")
        .filter(|row| args.case_ids.is_empty() || args.case_ids.contains(&row.id))
        .collect::<Vec<_>>();
    if let Some(max_cases) = args.max_cases {
        text_rows.truncate(max_cases);
    }
    if text_rows.is_empty() {
        anyhow::bail!(
            "calibration file {} contains no selected text rows",
            args.calibration.display()
        );
    }

    for manifest in [
        &args.q8_manifest,
        &args.q4_mlp_manifest,
        &args.q4_text_safe_manifest,
        &args.mixed_manifest,
    ] {
        if !manifest.exists() {
            anyhow::bail!(
                "quantized manifest {} does not exist; run quant-convert for this policy first",
                manifest.display()
            );
        }
    }

    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());
    let mut policy_specs = vec![
        ("dense", None::<&PathBuf>),
        ("q8-text-linears", Some(&args.q8_manifest)),
        ("q4k-mlp-only", Some(&args.q4_mlp_manifest)),
        ("q4k-text-safe", Some(&args.q4_text_safe_manifest)),
        ("q4k-mlp-q8-text", Some(&args.mixed_manifest)),
    ];
    if args.full_text_manifest.exists() {
        policy_specs.push(("q4k-full-text", Some(&args.full_text_manifest)));
    }

    let ctx = QuantQualityCtx {
        bundle: &bundle,
        device: &device,
        dtype,
        tokenizer: &tokenizer,
        eos_ids: &eos_ids,
        args: &args,
    };
    let mut policy_runs = Vec::new();
    for (label, manifest) in policy_specs {
        eprintln!("running quant-quality policy {label}");
        let started = Instant::now();
        let (generations, load_elapsed, quantized_load) =
            run_quant_quality_policy(&ctx, manifest, &text_rows)
                .with_context(|| format!("run quant-quality policy {label}"))?;
        policy_runs.push(QuantQualityPolicyRun {
            label: label.to_string(),
            manifest: manifest.cloned(),
            load_elapsed,
            run_elapsed: started.elapsed(),
            quantized_load,
            generations,
        });
    }

    let dense_generations = policy_runs
        .first()
        .context("missing dense quant-quality generations")?;
    let mut cases = Vec::new();
    let mut summaries = HashMap::<String, Vec<QualityComparison>>::new();
    for (case_index, row) in text_rows.iter().enumerate() {
        let dense_generation = dense_generations
            .generations
            .get(case_index)
            .context("dense generation count did not match selected rows")?;
        let mut candidates = Vec::new();
        for policy_run in policy_runs.iter().skip(1) {
            let candidate_generation = policy_run
                .generations
                .get(case_index)
                .context("candidate generation count did not match selected rows")?;
            let comparison = compare_quality_outputs(
                &dense_generation.stats.generated_token_ids,
                &dense_generation.raw_text,
                &candidate_generation.stats.generated_token_ids,
                &candidate_generation.raw_text,
                &thresholds,
            );
            summaries
                .entry(policy_run.label.clone())
                .or_default()
                .push(comparison.clone());
            candidates.push(serde_json::json!({
                "policy": policy_run.label,
                "generation": quant_quality_generation_json(candidate_generation),
                "comparison": quality_comparison_json(&comparison),
            }));
        }
        cases.push(serde_json::json!({
            "id": row.id,
            "category": row.category,
            "expected_behavior": row.expected_behavior,
            "enable_thinking": row.enable_thinking,
            "prompt_tokens": row.token_ids.len(),
            "max_new_tokens": quality_generation_args(&args.generation, row).max_new_tokens,
            "dense": quant_quality_generation_json(dense_generation),
            "candidates": candidates,
        }));
    }

    let mut policy_summaries = summaries
        .iter()
        .map(|(policy, comparisons)| quality_summary_json(policy, comparisons))
        .collect::<Vec<_>>();
    policy_summaries.sort_by(|left, right| {
        left["policy"]
            .as_str()
            .unwrap_or_default()
            .cmp(right["policy"].as_str().unwrap_or_default())
    });
    let passed = policy_summaries
        .iter()
        .all(|summary| summary["passed"].as_bool().unwrap_or(false));

    let report = serde_json::json!({
        "kind": "lmbrrr_quantization_quality_eval",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "downsample_mode": args.model.downsample_mode.as_str(),
        "calibration_set": args.calibration,
        "selected_cases": text_rows.len(),
        "artifact_seconds": secs(bundle.elapsed),
        "thresholds": {
            "min_prefix_ratio": thresholds.min_prefix_ratio,
            "min_token_jaccard": thresholds.min_token_jaccard,
            "min_lexical_jaccard": thresholds.min_lexical_jaccard,
            "max_length_ratio_delta": thresholds.max_length_ratio_delta,
        },
        "generation": {
            "max_new_tokens_cap": args.generation.max_new_tokens,
            "temperature": args.generation.temperature,
            "top_p": args.generation.top_p,
            "top_k": args.generation.top_k,
            "seed": args.generation.seed,
            "repeat_penalty": args.generation.repeat_penalty,
            "repeat_last_n": args.generation.repeat_last_n,
        },
        "policy_runs": policy_runs.iter().map(|run| {
            serde_json::json!({
                "label": run.label,
                "manifest": run.manifest,
                "load_seconds": secs(run.load_elapsed),
                "run_seconds": secs(run.run_elapsed),
                "load": quantized_load_json(&run.quantized_load),
            })
        }).collect::<Vec<_>>(),
        "policy_summaries": policy_summaries,
        "passed": passed,
        "cases": cases,
        "gate_note": "A candidate passes a case when it exactly matches dense tokens or meets every configured prefix, token-overlap, lexical-overlap, and length-delta threshold.",
    });
    if args.fail_on_gate && !passed {
        write_json_report(args.output.as_ref(), &report)?;
        anyhow::bail!("one or more quantization quality gates failed");
    }
    write_json_report(args.output.as_ref(), &report)
}

/// Everything shared across the per-policy quality runs: the resolved model
/// bundle, execution device/dtype, and decode references.
struct QuantQualityCtx<'a> {
    bundle: &'a ArtifactBundle,
    device: &'a Device,
    dtype: DType,
    tokenizer: &'a Tokenizer,
    eos_ids: &'a [u32],
    args: &'a QuantQualityArgs,
}

fn run_quant_quality_policy(
    ctx: &QuantQualityCtx,
    quantized_manifest: Option<&PathBuf>,
    rows: &[&CalibrationRow],
) -> Result<(
    Vec<QuantQualityGeneration>,
    Duration,
    Option<QuantizedLoadStats>,
)> {
    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        ctx.bundle,
        ctx.dtype,
        ctx.device,
        quantized_manifest,
        None,
    )?;
    let mut generations = Vec::with_capacity(rows.len());
    for row in rows {
        let generation = quality_generation_args(&ctx.args.generation, row);
        let stats = generate_tokens(
            &mut model,
            ctx.device,
            &generation,
            &row.token_ids,
            None::<&ProcessedImages>,
            &ctx.args.model.downsample_mode,
            ctx.eos_ids,
            |_, _, _, _| Ok(()),
        )?;
        let raw_text = decode_tokens(ctx.tokenizer, &stats.generated_token_ids)?;
        let parts = split_reasoning_text(&raw_text, row.enable_thinking);
        generations.push(QuantQualityGeneration {
            stats,
            raw_text,
            reasoning_text: parts.reasoning_text,
            answer_text: parts.answer_text,
        });
    }
    Ok((generations, load_elapsed, quantized_load))
}

fn quality_generation_args(base: &GenerationArgs, row: &CalibrationRow) -> GenerationArgs {
    let mut generation = base.clone();
    generation.enable_thinking = row.enable_thinking;
    if let Some(row_max_new_tokens) = row.max_new_tokens {
        generation.max_new_tokens = generation.max_new_tokens.min(row_max_new_tokens);
    }
    generation
}

fn quant_quality_generation_json(generation: &QuantQualityGeneration) -> serde_json::Value {
    serde_json::json!({
        "generated_tokens": generation.stats.generated_tokens,
        "generated_token_ids": generation.stats.generated_token_ids,
        "eos_reached": generation.stats.eos_reached,
        "prefill_seconds": secs(generation.stats.prefill_elapsed),
        "prefill_tokens_per_second": generation.stats.prefill_tokens_per_second(),
        "decode_seconds": secs(generation.stats.decode_elapsed),
        "decode_model_seconds": secs(generation.stats.decode_model_elapsed),
        "decode_tokens_per_second": generation.stats.decode_tokens_per_second(),
        "steady_state_tokens_per_second": generation.stats.steady_state_tokens_per_second(),
        "text": {
            "raw": generation.raw_text,
            "reasoning": generation.reasoning_text,
            "answer": generation.answer_text,
        },
    })
}

fn compare_quality_outputs(
    dense_token_ids: &[u32],
    dense_text: &str,
    candidate_token_ids: &[u32],
    candidate_text: &str,
    thresholds: &QualityThresholds,
) -> QualityComparison {
    let exact_token_match = dense_token_ids == candidate_token_ids;
    let common_prefix_tokens = common_prefix_len(dense_token_ids, candidate_token_ids);
    let divergence_index = (!exact_token_match).then_some(common_prefix_tokens);
    let dense_len = dense_token_ids.len().max(1);
    let prefix_ratio = common_prefix_tokens as f64 / dense_len as f64;
    let token_jaccard = token_multiset_jaccard(dense_token_ids, candidate_token_ids);
    let lexical_jaccard = lexical_multiset_jaccard(dense_text, candidate_text);
    let length_ratio = if dense_token_ids.is_empty() {
        if candidate_token_ids.is_empty() {
            1.0
        } else {
            f64::INFINITY
        }
    } else {
        candidate_token_ids.len() as f64 / dense_token_ids.len() as f64
    };
    let length_ratio_delta = (length_ratio - 1.0).abs();
    let passed_gate = exact_token_match
        || (prefix_ratio >= thresholds.min_prefix_ratio
            && token_jaccard >= thresholds.min_token_jaccard
            && lexical_jaccard >= thresholds.min_lexical_jaccard
            && length_ratio_delta <= thresholds.max_length_ratio_delta);
    QualityComparison {
        exact_token_match,
        common_prefix_tokens,
        divergence_index,
        prefix_ratio,
        token_jaccard,
        lexical_jaccard,
        length_ratio,
        length_ratio_delta,
        passed_gate,
    }
}

fn quality_comparison_json(comparison: &QualityComparison) -> serde_json::Value {
    serde_json::json!({
        "exact_token_match": comparison.exact_token_match,
        "common_prefix_tokens": comparison.common_prefix_tokens,
        "divergence_index": comparison.divergence_index,
        "prefix_ratio": comparison.prefix_ratio,
        "token_jaccard": comparison.token_jaccard,
        "lexical_jaccard": comparison.lexical_jaccard,
        "length_ratio": comparison.length_ratio,
        "length_ratio_delta": comparison.length_ratio_delta,
        "passed_gate": comparison.passed_gate,
    })
}

fn quality_summary_json(policy: &str, comparisons: &[QualityComparison]) -> serde_json::Value {
    let cases = comparisons.len();
    let exact_token_matches = comparisons
        .iter()
        .filter(|comparison| comparison.exact_token_match)
        .count();
    let passed_cases = comparisons
        .iter()
        .filter(|comparison| comparison.passed_gate)
        .count();
    serde_json::json!({
        "policy": policy,
        "cases": cases,
        "exact_token_matches": exact_token_matches,
        "passed_cases": passed_cases,
        "failed_cases": cases.saturating_sub(passed_cases),
        "mean_prefix_ratio": mean(comparisons.iter().map(|comparison| comparison.prefix_ratio)),
        "mean_token_jaccard": mean(comparisons.iter().map(|comparison| comparison.token_jaccard)),
        "mean_lexical_jaccard": mean(comparisons.iter().map(|comparison| comparison.lexical_jaccard)),
        "mean_length_ratio_delta": mean(comparisons.iter().map(|comparison| comparison.length_ratio_delta)),
        "passed": cases > 0 && passed_cases == cases,
    })
}

fn validate_quality_threshold(name: &str, value: f64, min: f64, max: f64) -> Result<()> {
    if !value.is_finite() || value < min || value > max {
        anyhow::bail!("{name} must be finite and within [{min}, {max}], got {value}");
    }
    Ok(())
}

fn common_prefix_len<T: Eq>(left: &[T], right: &[T]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(left, right)| left == right)
        .count()
}

fn token_multiset_jaccard(left: &[u32], right: &[u32]) -> f64 {
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    let mut left_counts = HashMap::<u32, usize>::new();
    let mut right_counts = HashMap::<u32, usize>::new();
    for token in left {
        *left_counts.entry(*token).or_default() += 1;
    }
    for token in right {
        *right_counts.entry(*token).or_default() += 1;
    }
    let mut intersection = 0usize;
    let mut union = 0usize;
    for (token, left_count) in &left_counts {
        let right_count = right_counts.get(token).copied().unwrap_or(0);
        intersection += (*left_count).min(right_count);
        union += (*left_count).max(right_count);
    }
    for (token, right_count) in &right_counts {
        if !left_counts.contains_key(token) {
            union += *right_count;
        }
    }
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

fn lexical_multiset_jaccard(left: &str, right: &str) -> f64 {
    let left_terms = lexical_terms(left);
    let right_terms = lexical_terms(right);
    if left_terms.is_empty() && right_terms.is_empty() {
        return 1.0;
    }
    let mut left_counts = HashMap::<String, usize>::new();
    let mut right_counts = HashMap::<String, usize>::new();
    for term in left_terms {
        *left_counts.entry(term).or_default() += 1;
    }
    for term in right_terms {
        *right_counts.entry(term).or_default() += 1;
    }
    let mut intersection = 0usize;
    let mut union = 0usize;
    for (term, left_count) in &left_counts {
        let right_count = right_counts.get(term).copied().unwrap_or(0);
        intersection += (*left_count).min(right_count);
        union += (*left_count).max(right_count);
    }
    for (term, right_count) in &right_counts {
        if !left_counts.contains_key(term) {
            union += *right_count;
        }
    }
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

fn lexical_terms(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            current.extend(ch.to_lowercase());
        } else if !current.is_empty() {
            terms.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        terms.push(current);
    }
    terms
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for value in values {
        total += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// Report label for one token-count arm: m=1 is the decode matvec; larger m
/// is the batched (verify-chunk / prefill) path.
fn matmul_mode_name(tokens: usize) -> String {
    if tokens == 1 {
        "decode_mv".to_string()
    } else {
        format!("mm_{tokens}")
    }
}

#[derive(Clone, Debug)]
struct MatmulShape {
    name: &'static str,
    family: &'static str,
    in_dim: usize,
    out_dim: usize,
}

/// One quantized bench cell selected for a bounded Metal capture.
#[derive(Clone, Debug)]
struct CaptureCell {
    shape: String,
    weight: String,
    activation: String,
    tokens: usize,
}

fn parse_capture_cell(spec: &str) -> Result<CaptureCell> {
    let parts = spec.split(':').collect::<Vec<_>>();
    let [shape, weight, activation, tokens] = parts.as_slice() else {
        anyhow::bail!(
            "--gpu-capture-cell must be shape:weight:activation:tokens, e.g. lm_head:Q4K:BF16:1"
        );
    };
    Ok(CaptureCell {
        shape: shape.to_string(),
        weight: weight.to_ascii_uppercase(),
        activation: activation.to_ascii_uppercase(),
        tokens: tokens
            .parse()
            .with_context(|| format!("--gpu-capture-cell tokens field {tokens:?} must be an integer"))?,
    })
}

fn quant_matmul_shapes(cfg: &MiniCpmConfig, include_lm_head: bool) -> Vec<MatmulShape> {
    let text = &cfg.text_config;
    let key_dim = text.linear_key_head_dim * text.linear_num_key_heads;
    let value_dim = text.linear_value_head_dim * text.linear_num_value_heads;
    let conv_dim = key_dim * 2 + value_dim;
    let mut shapes = vec![
        MatmulShape {
            name: "deltanet_in_proj_qkv",
            family: "text.deltanet",
            in_dim: text.hidden_size,
            out_dim: conv_dim,
        },
        MatmulShape {
            name: "deltanet_out_proj",
            family: "text.deltanet",
            in_dim: value_dim,
            out_dim: text.hidden_size,
        },
        MatmulShape {
            name: "mlp_up_or_gate_proj",
            family: "text.mlp",
            in_dim: text.hidden_size,
            out_dim: text.intermediate_size,
        },
        MatmulShape {
            name: "mlp_down_proj",
            family: "text.mlp",
            in_dim: text.intermediate_size,
            out_dim: text.hidden_size,
        },
        MatmulShape {
            name: "full_attention_q_proj",
            family: "text.full_attention",
            in_dim: text.hidden_size,
            out_dim: text.num_attention_heads * text.head_dim * 2,
        },
        MatmulShape {
            name: "full_attention_o_proj",
            family: "text.full_attention",
            in_dim: text.num_attention_heads * text.head_dim,
            out_dim: text.hidden_size,
        },
    ];
    if include_lm_head {
        shapes.push(MatmulShape {
            name: "lm_head",
            family: "text.lm_head",
            in_dim: text.hidden_size,
            out_dim: text.vocab_size,
        });
    }
    shapes
}

/// Deterministic host-side inputs plus timing knobs shared by every
/// dense/quantized arm of one (shape, mode) bench cell.
struct MatmulBenchCtx<'a> {
    weight_cpu: &'a Tensor,
    input_cpu: &'a Tensor,
    device: &'a Device,
    warmup: usize,
    iterations: usize,
    capture: Option<&'a CaptureCell>,
}

fn bench_dense_matmul(
    shape: &MatmulShape,
    tokens: usize,
    activation_dtype: DType,
    ctx: &MatmulBenchCtx,
) -> serde_json::Value {
    let device = ctx.device;
    let result = (|| -> Result<(Duration, Duration)> {
        let prepare_started = Instant::now();
        let weight = ctx.weight_cpu.to_device(device)?.to_dtype(activation_dtype)?;
        let input = ctx.input_cpu.to_device(device)?.to_dtype(activation_dtype)?;
        device.synchronize()?;
        let prepare_elapsed = prepare_started.elapsed();
        let elapsed = time_iterations(device, ctx.warmup, ctx.iterations, || {
            let w = weight.t()?;
            let tokens = input.dim(1)?;
            Ok(input
                .reshape((tokens, shape.in_dim))?
                .matmul(&w)?
                .reshape((1, tokens, shape.out_dim))?)
        })?;
        Ok((prepare_elapsed, elapsed))
    })();
    matmul_bench_row(
        shape,
        tokens,
        BenchRowBackend {
            backend: "dense",
            weight_dtype: Some(format!("{activation_dtype:?}")),
            activation_dtype,
            activation_cast: None,
        },
        result,
        ctx,
    )
}

fn bench_quant_matmul(
    shape: &MatmulShape,
    tokens: usize,
    quant_dtype: GgmlDType,
    activation_dtype: DType,
    ctx: &MatmulBenchCtx,
) -> serde_json::Value {
    let device = ctx.device;
    // Route through MixedLinear so the bench measures the DEPLOYED path:
    // bf16_direct kernels for Q8_0/Q4K/Q6K on Metal (F32 accumulate + one
    // output hop), the F32 input cast only where the runner actually pays it.
    let bf16_direct = activation_dtype == DType::BF16
        && device.is_metal()
        && matches!(
            quant_dtype,
            GgmlDType::Q8_0 | GgmlDType::Q4K | GgmlDType::Q6K
        );
    let result = (|| -> Result<(Duration, Duration)> {
        let prepare_started = Instant::now();
        let qweight = QTensor::quantize_onto(ctx.weight_cpu, quant_dtype, device)?;
        let linear = lmbrrr::quantized_linear::MixedLinear::from_qtensor(qweight)?;
        let input = ctx.input_cpu.to_device(device)?.to_dtype(activation_dtype)?;
        device.synchronize()?;
        let prepare_elapsed = prepare_started.elapsed();
        let elapsed = time_iterations(device, ctx.warmup, ctx.iterations, || {
            Ok(linear.forward(&input)?)
        })?;
        let capture_match = ctx.capture.is_some_and(|cell| {
            cell.shape == shape.name
                && cell.weight == format!("{quant_dtype:?}").to_ascii_uppercase()
                && cell.activation == format!("{activation_dtype:?}").to_ascii_uppercase()
                && cell.tokens == tokens
        });
        if capture_match {
            let Device::Metal(md) = device else {
                anyhow::bail!("--gpu-capture-cell requires the Metal device");
            };
            let path = std::env::current_dir()?.join(format!(
                "qmb-{}-{quant_dtype:?}-{activation_dtype:?}-m{tokens}.gputrace",
                shape.name
            ));
            if path.exists() {
                std::fs::remove_dir_all(&path)?;
            }
            md.capture(&path)?;
            // Metal capture records only command buffers CREATED inside the
            // window, and candle keeps one pre-created; retire it so the
            // captured forwards land on a fresh buffer (else the trace is
            // empty).
            device.synchronize()?;
            for _ in 0..3 {
                linear.forward(&input)?;
            }
            device.synchronize()?;
            md.stop_capture();
            eprintln!("captured {}", path.display());
        }
        Ok((prepare_elapsed, elapsed))
    })();
    matmul_bench_row(
        shape,
        tokens,
        BenchRowBackend {
            backend: "quantized",
            weight_dtype: Some(format!("{quant_dtype:?}")),
            activation_dtype,
            activation_cast: if activation_dtype == DType::F32 {
                None
            } else if bf16_direct {
                Some("bf16_direct")
            } else {
                Some("to_f32")
            },
        },
        result,
        ctx,
    )
}

/// How one bench row computed its matmul, for the report columns.
struct BenchRowBackend<'a> {
    backend: &'a str,
    weight_dtype: Option<String>,
    activation_dtype: DType,
    activation_cast: Option<&'a str>,
}

fn matmul_bench_row(
    shape: &MatmulShape,
    tokens: usize,
    backend: BenchRowBackend,
    result: Result<(Duration, Duration)>,
    ctx: &MatmulBenchCtx,
) -> serde_json::Value {
    let BenchRowBackend {
        backend,
        weight_dtype,
        activation_dtype,
        activation_cast,
    } = backend;
    let iterations = ctx.iterations;
    let tokens_per_iteration = ctx.input_cpu.dim(1).unwrap_or(1);
    match result {
        Ok((prepare_elapsed, elapsed)) => serde_json::json!({
            "shape": shape.name,
            "family": shape.family,
            "mode": matmul_mode_name(tokens),
            "tokens": tokens,
            "backend": backend,
            "weight_dtype": weight_dtype,
            "activation_dtype": format!("{activation_dtype:?}"),
            "activation_cast": activation_cast,
            "in_dim": shape.in_dim,
            "out_dim": shape.out_dim,
            "iterations": iterations,
            "prepare_seconds": secs(prepare_elapsed),
            "elapsed_seconds": secs(elapsed),
            "seconds_per_iteration": secs(elapsed) / iterations as f64,
            "tokens_per_second": tokens_per_second(tokens_per_iteration * iterations, elapsed),
            "ok": true,
        }),
        Err(err) => serde_json::json!({
            "shape": shape.name,
            "family": shape.family,
            "mode": matmul_mode_name(tokens),
            "tokens": tokens,
            "backend": backend,
            "weight_dtype": weight_dtype,
            "activation_dtype": format!("{activation_dtype:?}"),
            "activation_cast": activation_cast,
            "in_dim": shape.in_dim,
            "out_dim": shape.out_dim,
            "iterations": iterations,
            "ok": false,
            "error": err.to_string(),
        }),
    }
}

fn deterministic_values(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let a = (idx % 251) as f32 - 125.0;
            let b = ((idx / 251) % 127) as f32 - 63.0;
            (a * 0.007 + b * 0.003).sin() * scale
        })
        .collect()
}

fn run_quant_baseline_case(
    model: &mut MiniCpmForConditionalGeneration,
    device: &Device,
    tokenizer: &Tokenizer,
    row: &CalibrationRow,
    downsample_mode: &str,
    top_k: usize,
) -> Result<serde_json::Value> {
    if row.token_ids.is_empty() {
        anyhow::bail!("calibration row {} has no token ids", row.id);
    }
    model.clear_cache();
    let input = Tensor::from_slice(&row.token_ids, (1, row.token_ids.len()), device)?;
    let started = Instant::now();
    let logits = model.forward(&input, None::<&ProcessedImages>, downsample_mode, 0)?;
    device.synchronize()?;
    let forward_elapsed = started.elapsed();
    let top_logits = top_k_logits(&logits.squeeze(0)?, top_k, tokenizer)?;
    let top1 = top_logits.first();
    Ok(serde_json::json!({
        "id": row.id.as_str(),
        "category": row.category.as_str(),
        "modality": row.modality.as_str(),
        "enable_thinking": row.enable_thinking,
        "media_status": row.media_status.as_deref(),
        "prompt_tokens": row.token_ids.len(),
        "declared_prompt_tokens": row.prompt_token_count,
        "prompt_token_count_match": row.token_ids.len() == row.prompt_token_count,
        "sensitivity_focus": row.sensitivity_focus,
        "prefill_seconds": secs(forward_elapsed),
        "prefill_tokens_per_second": tokens_per_second(row.token_ids.len(), forward_elapsed),
        "top1_token_id": top1.map(|item| item.token_id),
        "top1_token": top1.map(|item| item.token.as_str()),
        "top1_logit": top1.map(|item| item.logit),
        "top_logits": top_logits_json(&top_logits),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quality_comparison_accepts_exact_token_match() {
        let thresholds = QualityThresholds {
            min_prefix_ratio: 0.9,
            min_token_jaccard: 0.9,
            min_lexical_jaccard: 0.9,
            max_length_ratio_delta: 0.0,
        };
        let comparison = compare_quality_outputs(
            &[1, 2, 3],
            "Paris is the capital.",
            &[1, 2, 3],
            "Paris is the capital.",
            &thresholds,
        );

        assert!(comparison.exact_token_match);
        assert_eq!(comparison.common_prefix_tokens, 3);
        assert_eq!(comparison.divergence_index, None);
        assert!(comparison.passed_gate);
    }

    #[test]
    fn quality_comparison_reports_divergence_and_overlap() {
        let thresholds = QualityThresholds {
            min_prefix_ratio: 0.25,
            min_token_jaccard: 0.25,
            min_lexical_jaccard: 0.25,
            max_length_ratio_delta: 1.0,
        };
        let comparison = compare_quality_outputs(
            &[10, 11, 12, 13],
            "the answer is paris",
            &[10, 99, 12],
            "the answer is london",
            &thresholds,
        );

        assert!(!comparison.exact_token_match);
        assert_eq!(comparison.common_prefix_tokens, 1);
        assert_eq!(comparison.divergence_index, Some(1));
        assert!(comparison.token_jaccard > 0.0);
        assert!(comparison.lexical_jaccard > 0.0);
        assert!(comparison.passed_gate);
    }

}
