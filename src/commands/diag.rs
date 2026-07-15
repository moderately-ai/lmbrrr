//! Diagnostic and measurement commands: logits parity vs the Transformers
//! oracle, per-component decode profiling, hidden-state tracing, the Metal
//! roofline probe, the verify-cost table producer, vision parity, and
//! fakequant export.

use crate::*;

fn aggregate_profile_events(events: &[Qwen35ProfileEvent]) -> Vec<serde_json::Value> {
    let mut groups = HashMap::<String, (usize, f64)>::new();
    for event in events {
        let entry = groups.entry(event.component.clone()).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += event.seconds;
    }
    aggregate_groups(groups)
}

fn aggregate_profile_events_by_layer_kind(events: &[Qwen35ProfileEvent]) -> Vec<serde_json::Value> {
    let mut groups = HashMap::<String, (usize, f64)>::new();
    for event in events {
        let key = event
            .layer_kind
            .clone()
            .unwrap_or_else(|| "unlayered".to_string());
        let entry = groups.entry(key).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += event.seconds;
    }
    aggregate_groups(groups)
}

fn aggregate_groups(groups: HashMap<String, (usize, f64)>) -> Vec<serde_json::Value> {
    let total = groups.values().map(|(_, seconds)| *seconds).sum::<f64>();
    let mut rows = groups
        .into_iter()
        .map(|(name, (count, seconds))| {
            serde_json::json!({
                "name": name,
                "count": count,
                "seconds": seconds,
                "avg_ms": if count > 0 { seconds * 1000.0 / count as f64 } else { 0.0 },
                "share": if total > 0.0 { seconds / total } else { 0.0 },
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        let left = left["seconds"].as_f64().unwrap_or(0.0);
        let right = right["seconds"].as_f64().unwrap_or(0.0);
        right.total_cmp(&left)
    });
    rows
}

fn trace_capture_layers(requested: &[usize], num_layers: usize) -> Result<Vec<usize>> {
    if num_layers == 0 {
        anyhow::bail!("model has zero text layers");
    }
    let mut layers = if requested.is_empty() {
        vec![0, (num_layers - 1) / 2, num_layers - 1]
    } else {
        requested.to_vec()
    };
    layers.sort_unstable();
    layers.dedup();
    if let Some(layer) = layers.iter().find(|layer| **layer >= num_layers) {
        anyhow::bail!("capture layer {layer} is outside 0..{}", num_layers - 1);
    }
    Ok(layers)
}

#[derive(Debug)]
struct TopLogitComparison {
    top1_match: bool,
    top_k_overlap: usize,
    top_k_overlap_threshold: usize,
    max_abs_shared_logit_delta: Option<f32>,
    shared_logit_deltas: Vec<serde_json::Value>,
    passed: bool,
}

fn compare_top_logits(
    candle_top: &[TopLogit],
    oracle_token_ids: &[u32],
    oracle_logits: &[f32],
) -> TopLogitComparison {
    let oracle_by_id = oracle_token_ids
        .iter()
        .zip(oracle_logits.iter())
        .map(|(token_id, logit)| (*token_id, *logit))
        .collect::<HashMap<_, _>>();
    let oracle_set = oracle_by_id
        .keys()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let top_k_overlap = candle_top
        .iter()
        .filter(|item| oracle_set.contains(&item.token_id))
        .count();
    let top_k_overlap_threshold = candle_top.len().min(oracle_token_ids.len()).min(8);
    let top1_match =
        candle_top.first().map(|item| item.token_id) == oracle_token_ids.first().copied();
    let mut max_abs_shared_logit_delta = None::<f32>;
    let mut shared_logit_deltas = Vec::new();
    for item in candle_top {
        if let Some(oracle_logit) = oracle_by_id.get(&item.token_id) {
            let delta = item.logit - oracle_logit;
            let abs_delta = delta.abs();
            max_abs_shared_logit_delta = Some(
                max_abs_shared_logit_delta
                    .map(|current| current.max(abs_delta))
                    .unwrap_or(abs_delta),
            );
            shared_logit_deltas.push(serde_json::json!({
                "token_id": item.token_id,
                "token": item.token,
                "candle_logit": item.logit,
                "transformers_logit": oracle_logit,
                "delta": delta,
                "abs_delta": abs_delta,
            }));
        }
    }
    let passed = top1_match && top_k_overlap >= top_k_overlap_threshold;
    TopLogitComparison {
        top1_match,
        top_k_overlap,
        top_k_overlap_threshold,
        max_abs_shared_logit_delta,
        shared_logit_deltas,
        passed,
    }
}

pub(crate) fn verify_table(args: VerifyTableArgs) -> Result<()> {
    if args.iterations == 0 {
        anyhow::bail!("--iterations must be greater than zero");
    }
    let mut gammas = if args.gammas.is_empty() {
        vec![1, 2, 4, 8, 16, 32]
    } else {
        args.gammas.clone()
    };
    gammas.sort_unstable();
    gammas.dedup();
    if gammas.first() != Some(&1) {
        gammas.insert(0, 1);
    }
    let max_gamma = *gammas.last().expect("gammas is non-empty");
    let profiles = if args.profiles.is_empty() {
        BenchProfile::all().to_vec()
    } else {
        args.profiles.clone()
    };

    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let runtime = lmbrrr::runtime_config::RuntimeConfig::from_env();
    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
        args.model.quantize_lm_head,
        &runtime,
    )?;
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());

    let mut rows = Vec::new();
    for profile in &profiles {
        let prompt_text = chat_prompt(profile.prompt(), 0, false);
        let prompt_tokens = tokenize_prompt(&tokenizer, prompt_text)?;

        // Realistic verify content: greedy continuation tokens, padded by
        // repeating the last token if EOS arrives before max_gamma.
        let baseline = generate_tokens(
            &mut model,
            &device,
            &greedy_generation_args(max_gamma, false),
            &prompt_tokens,
            None::<&ProcessedImages>,
            &args.model.downsample_mode,
            &eos_ids,
            &runtime.decode,
            |_, _, _, _| Ok(()),
        )?;
        let mut chunk_tokens = baseline.generated_token_ids.clone();
        let pad = chunk_tokens.last().copied().unwrap_or(prompt_tokens[0]);
        while chunk_tokens.len() < max_gamma {
            chunk_tokens.push(pad);
        }

        for &gamma in &gammas {
            let mut samples = Vec::with_capacity(args.iterations);
            for iteration in 0..args.warmup + args.iterations {
                model.clear_cache();
                let prompt_input =
                    Tensor::from_slice(&prompt_tokens, (1, prompt_tokens.len()), &device)?;
                let _ = model.forward(
                    &prompt_input,
                    None::<&ProcessedImages>,
                    &args.model.downsample_mode,
                    0,
                )?;
                device.synchronize()?;

                let chunk = &chunk_tokens[..gamma];
                let chunk_input = Tensor::from_slice(chunk, (1, gamma), &device)?;
                // LMBRRR_VT_PROFILE=1 attaches the component profiler to the
                // final iteration so the chunk cost decomposes (used to chase
                // the l=1 -> l=2 doubling). Host-side attribution: encode +
                // queue backpressure per component, not GPU time.
                let profile_this = std::env::var(lmbrrr::env_keys::VT_PROFILE).is_ok_and(|v| v == "1")
                    && iteration + 1 == args.warmup + args.iterations;
                let vt_profiler = profile_this.then(Qwen35Profiler::new);
                if let Some(p) = &vt_profiler {
                    model.set_text_profiler(Some(p.clone()));
                }
                let started = Instant::now();
                let logits = model.forward_all_logits(
                    &chunk_input,
                    None::<&ProcessedImages>,
                    &args.model.downsample_mode,
                    prompt_tokens.len(),
                )?;
                device.synchronize()?;
                let chunk_elapsed = started.elapsed();
                if let Some(p) = &vt_profiler {
                    model.set_text_profiler(None);
                    let events = p.events();
                    eprintln!(
                        "vt-profile gamma={gamma} ctx={}: {}",
                        prompt_tokens.len(),
                        serde_json::to_string(&aggregate_profile_events(&events))?
                    );
                }
                let (_, argmax_elapsed) = argmax_tokens(&logits, &device)?;
                if iteration >= args.warmup {
                    samples.push((secs(chunk_elapsed), secs(argmax_elapsed)));
                }
            }
            let mut chunk_seconds = samples.iter().map(|(chunk, _)| *chunk).collect::<Vec<_>>();
            let argmax_seconds = samples.iter().map(|(_, argmax)| *argmax).sum::<f64>()
                / samples.len().max(1) as f64;
            let median_seconds = median(&mut chunk_seconds);
            let spread = chunk_seconds
                .last()
                .copied()
                .unwrap_or(median_seconds)
                - chunk_seconds.first().copied().unwrap_or(median_seconds);
            rows.push(serde_json::json!({
                "profile": profile.name(),
                "context_tokens": prompt_tokens.len(),
                "gamma": gamma,
                "iterations": args.iterations,
                "median_verify_seconds": median_seconds,
                "spread_verify_seconds": spread,
                "mean_argmax_seconds": argmax_seconds,
                "verify_tokens_per_second": gamma as f64 / median_seconds.max(f64::EPSILON),
                "samples": chunk_seconds,
            }));
        }
    }

    // Per-token efficiency vs the gamma=1 step within each profile.
    let mut enriched = Vec::with_capacity(rows.len());
    for row in &rows {
        let profile = row["profile"].as_str().unwrap_or_default();
        let base = rows
            .iter()
            .find(|candidate| {
                candidate["profile"].as_str() == Some(profile)
                    && candidate["gamma"].as_u64() == Some(1)
            })
            .and_then(|candidate| candidate["median_verify_seconds"].as_f64());
        let mut row = row.clone();
        if let (Some(base), Some(seconds), Some(gamma)) = (
            base,
            row["median_verify_seconds"].as_f64(),
            row["gamma"].as_u64(),
        ) {
            row["chunk_cost_vs_single_step"] = serde_json::json!(seconds / base);
            row["per_token_efficiency_vs_decode"] =
                serde_json::json!(gamma as f64 * base / seconds);
        }
        enriched.push(row);
    }

    let report = serde_json::json!({
        "kind": "lmbrrr_verify_throughput_table",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "gammas": gammas,
        "concurrency": 1,
        "concurrency_note": "single-request only; batched verify lands with batched-multi-stream-decode-runner",
        "rows": enriched,
        "scheduler_contract": "T_verify(gamma) per context bucket = median_verify_seconds; T_round(gamma) = T_fixed + T_draft + T_verify(gamma). T_fixed is the per-round host residual measured in-loop (dspark report round_residual_ms.drafter_rounds.median_ms), carried by the cost-model artifact's fixed_round_ms.",
    });
    write_json_report(args.output.as_ref(), &report)
}

pub(crate) fn roofline(args: RooflineArgs) -> Result<()> {
    if args.iterations == 0 || args.dispatch_chain == 0 {
        anyhow::bail!("--iterations and --dispatch-chain must be greater than zero");
    }
    let device = select_device(args.cpu)?;
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };

    let mut copy_rows = Vec::new();
    for mb in [64usize, 256, 1024] {
        let elements = mb * 1024 * 1024 / dtype.size_in_bytes();
        // Materialize a non-trivial buffer first; a bare zeros tensor can be
        // elided by the backend and reports absurd copy bandwidth.
        let x = Tensor::zeros(elements, dtype, &device)?.affine(1.0, 0.5)?;
        device.synchronize()?;
        let elapsed = time_iterations(&device, args.warmup, args.iterations, || {
            Ok(x.affine(1.000001, 0.0)?)
        })?;
        let bytes_moved = 2.0 * (elements * dtype.size_in_bytes()) as f64 * args.iterations as f64;
        copy_rows.push(serde_json::json!({
            "tensor_mb": mb,
            "seconds": secs(elapsed),
            "achieved_gbps": bytes_moved / secs(elapsed) / 1e9,
        }));
    }
    let peak_copy_gbps = copy_rows
        .iter()
        .filter_map(|row| row["achieved_gbps"].as_f64())
        .fold(0.0f64, f64::max);

    // Matvec throughput at the model's real decode shapes (weights dominate
    // the bytes, so achieved GB/s here is the effective decode-path bandwidth).
    let matvec_shapes: [(&str, usize, usize); 9] = [
        ("mlp_up_or_gate", 3584, 1024),
        ("mlp_down", 1024, 3584),
        ("deltanet_in_proj_qkv", 6144, 1024),
        ("deltanet_out_proj", 1024, 2048),
        ("full_attn_q_gate", 4096, 1024),
        ("full_attn_kv", 512, 1024),
        ("full_attn_o", 1024, 2048),
        ("lm_head", 248094, 1024),
        ("peak_square", 8192, 8192),
    ];
    let mut matvec_rows = Vec::new();
    for (name, out_dim, in_dim) in matvec_shapes {
        let weight = Tensor::zeros((out_dim, in_dim), dtype, &device)?;
        let linear = candle_nn::Linear::new(weight, None);
        let x = Tensor::zeros((1, in_dim), dtype, &device)?;
        let elapsed = time_iterations(&device, args.warmup, args.iterations, || {
            Ok(linear.forward(&x)?)
        })?;
        let weight_bytes = (out_dim * in_dim * dtype.size_in_bytes()) as f64;
        let per_iter = secs(elapsed) / args.iterations as f64;
        matvec_rows.push(serde_json::json!({
            "shape": name,
            "out_dim": out_dim,
            "in_dim": in_dim,
            "weight_bytes": weight_bytes as u64,
            "seconds_per_iteration": per_iter,
            "achieved_gbps": weight_bytes / per_iter / 1e9,
        }));
    }

    // Dependent-chain dispatch overhead: tiny affine ops that cannot overlap,
    // mirroring the serial structure of a decode forward.
    let tiny = Tensor::zeros(1, DType::F32, &device)?;
    for _ in 0..args.warmup {
        let mut y = tiny.clone();
        for _ in 0..args.dispatch_chain {
            y = y.affine(1.000001, 0.0)?;
        }
        let _ = y.to_vec1::<f32>()?;
    }
    device.synchronize()?;
    let started = Instant::now();
    for _ in 0..args.iterations {
        let mut y = tiny.clone();
        for _ in 0..args.dispatch_chain {
            y = y.affine(1.000001, 0.0)?;
        }
        let _ = y.to_vec1::<f32>()?;
    }
    device.synchronize()?;
    let tiny_chain_elapsed = started.elapsed();
    let per_dispatch_seconds =
        secs(tiny_chain_elapsed) / (args.iterations * args.dispatch_chain) as f64;

    // Same measurement with a dependent chain of real h=1024 matvecs.
    let weight = Tensor::zeros((1024usize, 1024usize), dtype, &device)?;
    let linear = candle_nn::Linear::new(weight, None);
    let chain = 64usize;
    for _ in 0..args.warmup {
        let mut y = Tensor::zeros((1, 1024usize), dtype, &device)?;
        for _ in 0..chain {
            y = linear.forward(&y)?;
        }
        device.synchronize()?;
    }
    device.synchronize()?;
    let started = Instant::now();
    for _ in 0..args.iterations {
        let mut y = Tensor::zeros((1, 1024usize), dtype, &device)?;
        for _ in 0..chain {
            y = linear.forward(&y)?;
        }
        device.synchronize()?;
    }
    let small_matvec_chain_elapsed = started.elapsed();
    let per_small_matvec_seconds =
        secs(small_matvec_chain_elapsed) / (args.iterations * chain) as f64;

    let dispatch_bound_tok_s =
        1.0 / (args.assumed_dispatches as f64 * per_dispatch_seconds).max(f64::EPSILON);
    let bandwidth_bound_tok_s = peak_copy_gbps * 1e9 / args.assumed_weight_bytes as f64;
    let combined_tok_s = 1.0
        / (args.assumed_dispatches as f64 * per_dispatch_seconds
            + args.assumed_weight_bytes as f64 / (peak_copy_gbps * 1e9));

    let report = serde_json::json!({
        "kind": "lmbrrr_metal_roofline",
        "schema_version": 1,
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "copy_bandwidth": copy_rows,
        "peak_copy_gbps": peak_copy_gbps,
        "matvec": matvec_rows,
        "dispatch_chain_length": args.dispatch_chain,
        "per_dispatch_seconds": per_dispatch_seconds,
        "per_dispatch_microseconds": per_dispatch_seconds * 1e6,
        "per_small_matvec_seconds": per_small_matvec_seconds,
        "per_small_matvec_microseconds": per_small_matvec_seconds * 1e6,
        "projections": {
            "assumed_dispatches_per_forward": args.assumed_dispatches,
            "assumed_weight_bytes_per_forward": args.assumed_weight_bytes,
            "dispatch_bound_tokens_per_second": dispatch_bound_tok_s,
            "bandwidth_bound_tokens_per_second": bandwidth_bound_tok_s,
            "combined_projection_tokens_per_second": combined_tok_s,
        },
        "note": "Dependent-chain timings mirror serial decode structure; copy bandwidth counts read+write bytes. Projections use the assumed dispatch count until encoder-level counting lands.",
    });
    write_json_report(args.output.as_ref(), &report)
}

pub(crate) fn trace_hidden_states(args: TraceArgs) -> Result<()> {
    if args.max_new_tokens == 0 {
        anyhow::bail!("--max-new-tokens must be greater than zero");
    }
    if args.top_k_logits == 0 {
        anyhow::bail!("--top-k-logits must be greater than zero");
    }

    let bundle = resolve_artifacts(&args.model)?;
    // A GPU capture run wants a pristine step: no hidden-state recorder
    // readbacks polluting the trace.
    let capture_layers = if args.gpu_capture_step.is_some() {
        if std::env::var(lmbrrr::env_keys::METAL_CAPTURE_ENABLED).is_err() {
            anyhow::bail!(
                "--gpu-capture-step needs METAL_CAPTURE_ENABLED=1 in the environment \
                 (undocumented Metal requirement)"
            );
        }
        Vec::new()
    } else {
        trace_capture_layers(
            &args.capture_layers,
            bundle.config.text_config.num_hidden_layers,
        )?
    };
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
    let trace_recorder = Qwen35TraceRecorder::new(capture_layers.clone());
    model.set_text_trace_recorder(Some(trace_recorder.clone()));
    model.clear_cache();

    let mut generated_token_ids = Vec::with_capacity(args.max_new_tokens);
    let mut steps = Vec::with_capacity(args.max_new_tokens);
    let mut total_forward_elapsed = Duration::ZERO;
    let mut total_argmax_elapsed = Duration::ZERO;
    let mut total_logits_elapsed = Duration::ZERO;
    let mut eos_reached = false;

    trace_recorder.clear();
    let prompt_input = Tensor::from_slice(&prompt_tokens, (1, prompt_tokens.len()), &device)?;
    let forward_start = Instant::now();
    let mut logits = model.forward(
        &prompt_input,
        None::<&ProcessedImages>,
        &args.model.downsample_mode,
        0,
    )?;
    device.synchronize()?;
    let mut forward_elapsed = forward_start.elapsed();
    let mut hidden_states = trace_recorder.take();
    let mut phase = "prefill";
    let mut context_position = prompt_tokens.len() - 1;
    let mut offset = 0usize;
    let mut seq_len = prompt_tokens.len();

    for step_index in 0..args.max_new_tokens {
        total_forward_elapsed += forward_elapsed;
        let (next_token, argmax_elapsed) = argmax_token(&logits, &device)?;
        total_argmax_elapsed += argmax_elapsed;
        let logits_start = Instant::now();
        let top_logits = top_k_logits(&logits.squeeze(0)?, args.top_k_logits, &tokenizer)?;
        let logits_elapsed = logits_start.elapsed();
        total_logits_elapsed += logits_elapsed;

        let step_eos = eos_ids.contains(&next_token);
        steps.push(serde_json::json!({
            "step": step_index,
            "phase": phase,
            "context_position": context_position,
            "offset": offset,
            "seq_len": seq_len,
            "target_token_id": next_token,
            "target_token": decode_token_lossy(&tokenizer, next_token),
            "eos": step_eos,
            "model_forward_seconds": secs(forward_elapsed),
            "argmax_seconds": secs(argmax_elapsed),
            "logits_top_k_seconds": secs(logits_elapsed),
            "top_logits": top_logits_json(&top_logits),
            "hidden_state_count": hidden_states.len(),
            "hidden_states": hidden_states,
        }));

        if step_eos {
            eos_reached = true;
            break;
        }
        generated_token_ids.push(next_token);
        if generated_token_ids.len() == args.max_new_tokens {
            break;
        }

        phase = "decode";
        context_position = prompt_tokens.len() + generated_token_ids.len() - 1;
        offset = context_position;
        seq_len = 1;
        trace_recorder.clear();
        let input = Tensor::from_slice(&[next_token], (1, 1), &device)?;
        // Bounded Metal capture around exactly this decode forward (the
        // preceding steps warm the shader caches; argmax is excluded — its
        // cost is separately known).
        let capture_this = args.gpu_capture_step == Some(generated_token_ids.len() - 1);
        let capture_path = std::env::current_dir()?.join(format!(
            "decode-step-{}.gputrace",
            generated_token_ids.len() - 1
        ));
        if capture_this {
            if capture_path.exists() {
                std::fs::remove_dir_all(&capture_path)?;
            }
            match &device {
                Device::Metal(md) => md.capture(&capture_path)?,
                _ => anyhow::bail!("--gpu-capture-step requires the Metal device"),
            }
        }
        let forward_start = Instant::now();
        logits = model.forward(
            &input,
            None::<&ProcessedImages>,
            &args.model.downsample_mode,
            offset,
        )?;
        device.synchronize()?;
        if capture_this {
            if let Device::Metal(md) = &device {
                md.stop_capture();
            }
            println!("gpu capture written: {}", capture_path.display());
        }
        forward_elapsed = forward_start.elapsed();
        hidden_states = trace_recorder.take();
    }
    model.set_text_trace_recorder(None);

    let prompt_token_count = prompt_tokens.len();
    let generated_token_count = generated_token_ids.len();
    let generated_text = decode_tokens(&tokenizer, &generated_token_ids)?;
    let report = serde_json::json!({
        "kind": "lmbrrr_hidden_state_trace",
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
        "prompt": args.prompt.as_str(),
        "prompt_token_ids": &prompt_tokens,
        "prompt_tokens": prompt_token_count,
        "max_new_tokens": args.max_new_tokens,
        "generated_token_ids": &generated_token_ids,
        "generated_tokens": generated_token_count,
        "generated_text": generated_text,
        "eos_reached": eos_reached,
        "capture_layers": capture_layers,
        "top_k_logits": args.top_k_logits,
        "timing": {
            "model_forward_seconds": secs(total_forward_elapsed),
            "model_forward_tokens_per_second": tokens_per_second(steps.len(), total_forward_elapsed),
            "argmax_seconds": secs(total_argmax_elapsed),
            "logits_top_k_seconds": secs(total_logits_elapsed),
        },
        "steps": steps,
    });

    write_json_report(args.output.as_ref(), &report)
}

pub(crate) fn profile_decode(args: ProfileArgs) -> Result<()> {
    if args.max_new_tokens == 0 {
        anyhow::bail!("--max-new-tokens must be greater than zero");
    }

    let prompt = args
        .prompt
        .clone()
        .unwrap_or_else(|| args.profile.prompt().to_string());
    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let prompt_text = chat_prompt(&prompt, 0, false);
    let tokens = tokenize_prompt(&tokenizer, prompt_text)?;
    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
        args.model.quantize_lm_head,
        &lmbrrr::runtime_config::RuntimeConfig::from_env(),
    )?;
    let profiler = Qwen35Profiler::new();
    model.set_text_profiler(Some(profiler.clone()));
    model.clear_cache();

    profiler.clear();
    let prefill_input = Tensor::from_slice(&tokens, (1, tokens.len()), &device)?;
    let prefill_started = Instant::now();
    let logits = model.forward(
        &prefill_input,
        None::<&ProcessedImages>,
        &args.model.downsample_mode,
        0,
    )?;
    device.synchronize()?;
    let prefill_elapsed = prefill_started.elapsed();
    let prefill_events = profiler.events();
    let (mut next_token, prefill_argmax_elapsed) = argmax_token(&logits, &device)?;

    let mut decode_events = Vec::new();
    let mut decode_steps = Vec::with_capacity(args.max_new_tokens);
    for step in 0..args.max_new_tokens {
        let position = tokens.len() + step;
        profiler.clear();
        let input = Tensor::from_slice(&[next_token], (1, 1), &device)?;
        let forward_started = Instant::now();
        let logits = model.forward(
            &input,
            None::<&ProcessedImages>,
            &args.model.downsample_mode,
            position,
        )?;
        device.synchronize()?;
        let forward_elapsed = forward_started.elapsed();
        let events = profiler.events();
        let (sampled, argmax_elapsed) = argmax_token(&logits, &device)?;
        let component_seconds = events.iter().map(|event| event.seconds).sum::<f64>();
        decode_steps.push(serde_json::json!({
            "step": step,
            "input_token_id": next_token,
            "next_token_id": sampled,
            "position": position,
            "model_forward_seconds": secs(forward_elapsed),
            "argmax_seconds": secs(argmax_elapsed),
            "profiled_component_seconds": component_seconds,
            "profiled_event_count": events.len(),
        }));
        decode_events.extend(events);
        next_token = sampled;
    }
    model.set_text_profiler(None);

    let prefill_profile_seconds = prefill_events
        .iter()
        .map(|event| event.seconds)
        .sum::<f64>();
    let decode_model_forward_seconds = decode_steps
        .iter()
        .filter_map(|step| step["model_forward_seconds"].as_f64())
        .sum::<f64>();
    let decode_argmax_seconds = decode_steps
        .iter()
        .filter_map(|step| step["argmax_seconds"].as_f64())
        .sum::<f64>();
    let decode_profile_seconds = decode_events.iter().map(|event| event.seconds).sum::<f64>();

    let report = serde_json::json!({
        "kind": "lmbrrr_decode_profile",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "downsample_mode": args.model.downsample_mode.as_str(),
        "profile": args.profile.name(),
        "prompt": prompt,
        "prompt_tokens": tokens.len(),
        "decode_steps": args.max_new_tokens,
        "artifact_seconds": secs(bundle.elapsed),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "timing_method": "wall-clock with device.synchronize() around profiled components; intrusive but attributable",
        "prefill": {
            "seconds": secs(prefill_elapsed),
            "tokens_per_second": tokens_per_second(tokens.len(), prefill_elapsed),
            "argmax_seconds": secs(prefill_argmax_elapsed),
            "profiled_component_seconds": prefill_profile_seconds,
            "aggregate": aggregate_profile_events(&prefill_events),
        },
        "decode": {
            "model_forward_seconds": decode_model_forward_seconds,
            "argmax_seconds": decode_argmax_seconds,
            "profiled_component_seconds": decode_profile_seconds,
            "model_forward_tokens_per_second": if decode_model_forward_seconds > 0.0 {
                args.max_new_tokens as f64 / decode_model_forward_seconds
            } else {
                0.0
            },
            "argmax_share_of_forward_plus_argmax": if decode_model_forward_seconds + decode_argmax_seconds > 0.0 {
                decode_argmax_seconds / (decode_model_forward_seconds + decode_argmax_seconds)
            } else {
                0.0
            },
            "aggregate": aggregate_profile_events(&decode_events),
            "by_layer_kind": aggregate_profile_events_by_layer_kind(&decode_events),
            "steps": decode_steps,
            "events": decode_events,
        },
        "kernel_launch_note": "This report counts synchronized component scopes, not Metal command-buffer kernel launches. Use it to rank code-path families; use Xcode/Metal capture for exact launch counts.",
    });

    write_json_report(args.output.as_ref(), &report)
}

pub(crate) fn logits(args: LogitsArgs) -> Result<()> {
    if args.top_k == 0 {
        anyhow::bail!("--top-k must be greater than zero");
    }

    let fixture_text = fs::read_to_string(&args.fixture)
        .with_context(|| format!("read fixture {}", args.fixture.display()))?;
    let fixture: LogitsOracleFixture =
        serde_json::from_str(&fixture_text).context("parse logits oracle fixture")?;
    let cases = fixture
        .cases
        .iter()
        .filter(|case| case.image_count == 0 && case.next_token_logits.is_some())
        .collect::<Vec<_>>();
    if cases.is_empty() {
        anyhow::bail!("fixture contains no text-only logits cases");
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
        &lmbrrr::runtime_config::RuntimeConfig::from_env(),
    )?;

    let mut rows = Vec::with_capacity(cases.len());
    let mut all_passed = true;
    for case in cases {
        model.clear_cache();
        let input = Tensor::from_slice(&case.token_ids, (1, case.token_ids.len()), &device)?;
        let logits = model.forward(
            &input,
            None::<&ProcessedImages>,
            &args.model.downsample_mode,
            0,
        )?;
        let candle_top = top_k_logits(&logits.squeeze(0)?, args.top_k, &tokenizer)?;
        let expected = case
            .next_token_logits
            .as_ref()
            .context("missing oracle logits")?;
        let expected_token_ids = expected
            .top_token_ids
            .as_ref()
            .context("missing oracle top_token_ids")?;
        let expected_logits = expected
            .top_logits
            .as_ref()
            .context("missing oracle top_logits")?;
        let comparison = compare_top_logits(&candle_top, expected_token_ids, expected_logits);
        all_passed &= comparison.passed;

        rows.push(serde_json::json!({
            "id": case.id.as_str(),
            "user_prompt": case.user_prompt.as_str(),
            "prompt_tokens": case.token_ids.len(),
            "oracle_prompt_tokens": case.prompt_token_count,
            "candle": {
                "top_token_ids": candle_top.iter().map(|item| item.token_id).collect::<Vec<_>>(),
                "top_tokens": candle_top.iter().map(|item| item.token.as_str()).collect::<Vec<_>>(),
                "top_logits": candle_top.iter().map(|item| item.logit).collect::<Vec<_>>(),
            },
            "transformers": {
                "top_token_ids": expected_token_ids,
                "top_logits": expected_logits,
            },
            "comparison": {
                "top1_match": comparison.top1_match,
                "top_k_overlap": comparison.top_k_overlap,
                "top_k_overlap_threshold": comparison.top_k_overlap_threshold,
                "max_abs_shared_logit_delta": comparison.max_abs_shared_logit_delta,
                "shared_logit_deltas": comparison.shared_logit_deltas,
                "passed": comparison.passed,
            }
        }));
    }

    let report = serde_json::json!({
        "kind": "lmbrrr_logits_parity",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "revision": args.model.revision.as_str(),
        "fixture_model_id": fixture.model_id,
        "fixture_revision": fixture.revision,
        "fixture": args.fixture,
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "downsample_mode": args.model.downsample_mode.as_str(),
        "top_k": args.top_k,
        "artifact_seconds": secs(bundle.elapsed),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "passed": all_passed,
        "cases": rows,
    });

    write_json_report(args.output.as_ref(), &report)?;
    if args.fail_on_mismatch && !all_passed {
        anyhow::bail!("Candle logits did not match the Transformers oracle");
    }
    Ok(())
}

pub(crate) fn vision_check(args: VisionCheckArgs) -> Result<()> {
    use lmbrrr::image_processor::preprocess_rgb_images;

    #[derive(serde::Deserialize)]
    struct FeatureFixture {
        downsample_mode: String,
        feature_cases: Vec<FeatureCase>,
    }
    #[derive(serde::Deserialize)]
    struct FeatureCase {
        id: String,
        height: usize,
        width: usize,
        feature_shape: Vec<usize>,
        sample_indices: Vec<usize>,
        sample_values: Vec<f32>,
    }

    let fixture: FeatureFixture = serde_json::from_str(include_str!(
        "../../evals/fixtures/minicpm_v46_transformers_image_features.json"
    ))?;
    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let preprocessor_path = bundle
        .artifacts
        .preprocessor
        .as_ref()
        .context("vision check requires preprocessor_config.json")?;
    let preprocessor = PreprocessorConfig::from_path(preprocessor_path)?;
    let (model, _, _) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
        args.model.quantize_lm_head,
        &lmbrrr::runtime_config::RuntimeConfig::from_env(),
    )?;

    let mut worst_abs = 0f32;
    let mut worst_mean = 0f32;
    for case in &fixture.feature_cases {
        // Same generator as the oracle's synthetic_image.
        let mut img = image::RgbImage::new(case.width as u32, case.height as u32);
        for y in 0..case.height {
            for x in 0..case.width {
                img.put_pixel(
                    x as u32,
                    y as u32,
                    image::Rgb([
                        ((x * 17 + y * 3) % 256) as u8,
                        ((x * 5 + y * 11 + 37) % 256) as u8,
                        ((x * 13 + y * 7 + 91) % 256) as u8,
                    ]),
                );
            }
        }
        let processed = preprocess_rgb_images(
            &[(PathBuf::from(format!("{}.png", case.id)), img)],
            &preprocessor,
            &device,
        )?;
        let features = model.image_features(&processed, &fixture.downsample_mode, dtype)?;
        let features = Tensor::cat(&features.iter().collect::<Vec<_>>(), 0)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
        anyhow::ensure!(
            features.dims() == case.feature_shape.as_slice(),
            "{}: feature shape {:?} != oracle {:?}",
            case.id,
            features.dims(),
            case.feature_shape
        );
        let flat = features.flatten_all()?.to_vec1::<f32>()?;
        let mut max_d = 0f32;
        let mut sum_d = 0f32;
        for (index, expected) in case.sample_indices.iter().zip(case.sample_values.iter()) {
            let got = flat[*index];
            let d = (got - expected).abs();
            max_d = max_d.max(d);
            sum_d += d;
        }
        let mean_d = sum_d / case.sample_indices.len() as f32;
        println!(
            "{}: shape {:?} ok, max |Δ| {max_d:.4}, mean |Δ| {mean_d:.4}",
            case.id,
            features.dims()
        );
        worst_abs = worst_abs.max(max_d);
        worst_mean = worst_mean.max(mean_d);
    }
    println!(
        "vision-check: worst max |Δ| {worst_abs:.4} (eps {}), worst mean |Δ| {worst_mean:.4} (eps {})",
        args.max_abs_delta, args.max_mean_delta
    );
    if worst_abs > args.max_abs_delta || worst_mean > args.max_mean_delta {
        anyhow::bail!("vision-check FAILED");
    }
    println!("vision-check PASSED");
    Ok(())
}

/// See [`TreeCheckArgs`]. Per round, with the pre-round state snapshotted:
/// the main branch is the greedy continuation, the alternate is the
/// runner-up token at the anchor followed by its greedy continuation. Chain
/// references are built by restoring the snapshot and running plain chunk
/// forwards over the same tokens.

#[derive(Debug, Deserialize)]
struct LogitsOracleFixture {
    model_id: String,
    revision: String,
    cases: Vec<LogitsOracleCase>,
}

#[derive(Debug, Deserialize)]
struct LogitsOracleCase {
    id: String,
    user_prompt: String,
    image_count: usize,
    prompt_token_count: usize,
    token_ids: Vec<u32>,
    next_token_logits: Option<OracleTopLogits>,
}

#[derive(Debug, Deserialize)]
struct OracleTopLogits {
    top_token_ids: Option<Vec<u32>>,
    top_logits: Option<Vec<f32>>,
}

pub(crate) fn fakequant_export(args: FakequantExportArgs) -> Result<()> {
    use candle::quantized::{GgmlDType, QTensor};

    let bundle = resolve_artifacts(&args.model)?;
    fs::create_dir_all(&args.output_dir)?;

    let eligible = |name: &str, shape: &[usize]| -> bool {
        name.ends_with(".weight")
            && shape.len() == 2
            && name.contains(".layers.")
            && (name.contains(".mlp.")
                || name.contains(".self_attn.")
                || name.contains(".linear_attn."))
            && !name.ends_with(".in_proj_a.weight")
            && !name.ends_with(".in_proj_b.weight")
            && shape[shape.len() - 1].is_multiple_of(256)
    };

    let mut quantized = 0usize;
    let mut passthrough = 0usize;
    let mut max_shift = 0f32;
    for shard in &bundle.artifacts.weights {
        let tensors = candle::safetensors::load(shard, &Device::Cpu)?;
        let mut out = std::collections::HashMap::new();
        for (name, tensor) in tensors {
            if eligible(&name, tensor.dims()) {
                let f32_tensor = tensor.to_dtype(DType::F32)?;
                let q = QTensor::quantize(&f32_tensor, GgmlDType::Q4K)?;
                let restored = q.dequantize(&Device::Cpu)?;
                let shift = (&restored - &f32_tensor)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?;
                max_shift = max_shift.max(shift);
                out.insert(name, restored.to_dtype(tensor.dtype())?);
                quantized += 1;
            } else {
                out.insert(name, tensor);
                passthrough += 1;
            }
        }
        let file_name = shard
            .file_name()
            .context("weight shard has no file name")?;
        candle::safetensors::save(&out, args.output_dir.join(file_name))?;
        println!(
            "wrote {}: {} quantized so far",
            file_name.to_string_lossy(),
            quantized
        );
    }
    // Sidecar configs the generator needs (tokenizer, configs, index).
    if let Some(dir) = bundle.artifacts.weights.first().and_then(|p| p.parent()) {
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            if path.is_file() && !name.ends_with(".safetensors") {
                fs::copy(&path, args.output_dir.join(&name))?;
            }
        }
    }
    // lmbrrr's hub cache never fetches the chat template or tokenizer
    // config (the Rust runner doesn't need them), but transformers-side
    // generation does; backfill from the vendored model dir.
    let vendored = Path::new("docs/research/models/minicpm-v-4.6/hf-model");
    for name in ["chat_template.jinja", "tokenizer_config.json"] {
        let target = args.output_dir.join(name);
        let source = vendored.join(name);
        if !target.exists() && source.exists() {
            fs::copy(&source, &target)?;
        }
    }
    println!(
        "fakequant export: {} quantized, {} passthrough, max |Δw| {:.5} -> {}",
        quantized,
        passthrough,
        max_shift,
        args.output_dir.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_capture_layers_defaults_to_low_mid_high() {
        let layers = trace_capture_layers(&[], 24).unwrap();
        assert_eq!(layers, [0, 11, 23]);
    }

    #[test]
    fn trace_capture_layers_sorts_dedups_and_validates() {
        let layers = trace_capture_layers(&[7, 3, 7], 8).unwrap();
        assert_eq!(layers, [3, 7]);
        assert!(trace_capture_layers(&[8], 8).is_err());
    }

}
