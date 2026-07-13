//! The interactive run command and the single/batched benchmark commands.

use crate::*;

fn benchmark_writer(path: Option<&PathBuf>, append: bool) -> Result<Box<dyn Write>> {
    match path {
        Some(path) => {
            let mut options = OpenOptions::new();
            options.create(true).write(true);
            if append {
                options.append(true);
            } else {
                options.truncate(true);
            }
            let file = options
                .open(path)
                .with_context(|| format!("open benchmark output {}", path.display()))?;
            Ok(Box::new(BufWriter::new(file)))
        }
        None => Ok(Box::new(BufWriter::new(std::io::stdout()))),
    }
}

fn bench_prompts(args: &BenchArgs) -> Vec<BenchPrompt> {
    let profiles = if args.profiles.is_empty() && args.prompts.is_empty() {
        BenchProfile::all().to_vec()
    } else {
        args.profiles.clone()
    };
    let mut prompts = profiles
        .into_iter()
        .map(|profile| BenchPrompt {
            name: profile.name().to_string(),
            text: profile.prompt().to_string(),
        })
        .collect::<Vec<_>>();

    prompts.extend(
        args.prompts
            .iter()
            .enumerate()
            .map(|(idx, prompt)| BenchPrompt {
                name: format!("custom-{}", idx + 1),
                text: prompt.clone(),
            }),
    );
    prompts
}

fn prepare_run_prompt(
    args: &RunArgs,
    preprocessor: Option<&PreprocessorConfig>,
    processed_images: Option<&ProcessedImages>,
) -> Result<String> {
    let text = chat_prompt(
        &args.prompt,
        args.images.len(),
        args.generation.enable_thinking,
    );
    match (preprocessor, processed_images) {
        (Some(preprocessor), Some(images)) => expand_image_placeholders(
            text,
            images,
            preprocessor.use_image_id,
            &args.model.downsample_mode,
        ),
        _ => Ok(text),
    }
}

pub(crate) fn multi_bench(args: MultiBenchArgs) -> Result<()> {
    if args.streams.is_empty() || args.streams.contains(&0) {
        anyhow::bail!("--streams must contain positive stream counts");
    }
    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let prompt_text = chat_prompt(&args.prompt, 0, false);
    let prompt_tokens = tokenize_prompt(&tokenizer, prompt_text)?;
    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
        head_loader_quant(&args.model),
    )?;
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());
    let len = prompt_tokens.len();

    // Single-stream reference for the equivalence check (advisory: batched
    // numerics can tie-flip).
    model.clear_cache();
    let mut rows = Vec::new();
    let reference = generate_tokens(
        &mut model,
        &device,
        &greedy_generation_args(args.max_new_tokens, false),
        &prompt_tokens,
        None::<&ProcessedImages>,
        &args.model.downsample_mode,
        &eos_ids,
        |_, _, _, _| Ok(()),
    )?;

    for &n in &args.streams {
    model.clear_cache();
    let mut batched: Vec<u32> = Vec::with_capacity(n * len);
    for _ in 0..n {
        batched.extend_from_slice(&prompt_tokens);
    }
    let input = Tensor::from_slice(&batched, (n, len), &device)?;
    let prefill_start = Instant::now();
    let mut logits = model.forward(&input, None::<&ProcessedImages>, &args.model.downsample_mode, 0)?;
    device.synchronize()?;
    let prefill_elapsed = prefill_start.elapsed();

    let mut streams: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut finished = vec![false; n];
    let mut position = len;
    let decode_start = Instant::now();
    let mut steps = 0usize;
    for _ in 0..args.max_new_tokens {
        // logits [n, vocab]
        let next = logits
            .to_dtype(DType::F32)?
            .argmax(D::Minus1)?
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        let mut all_done = true;
        for (i, tok) in next.iter().enumerate() {
            if !finished[i] {
                if eos_ids.contains(tok) {
                    finished[i] = true;
                } else {
                    streams[i].push(*tok);
                    all_done = false;
                }
            }
        }
        steps += 1;
        if all_done {
            break;
        }
        // Finished streams keep decoding their last token (static batch);
        // their outputs are ignored above.
        let feed: Vec<u32> = next
            .iter()
            .enumerate()
            .map(|(i, t)| if finished[i] { eos_ids[0] } else { *t })
            .collect();
        let step_input = Tensor::from_slice(&feed, (n, 1), &device)?;
        logits = model.forward(&step_input, None::<&ProcessedImages>, &args.model.downsample_mode, position)?;
        position += 1;
    }
    let decode_elapsed = decode_start.elapsed();
    let total_tokens: usize = streams.iter().map(|s| s.len()).sum();
    let aggregate_tps = total_tokens as f64 / decode_elapsed.as_secs_f64();

    let equiv = reference
        .generated_token_ids
        .iter()
        .zip(streams[0].iter())
        .take_while(|(a, b)| a == b)
        .count();

    rows.push(serde_json::json!({
        "streams": n,
        "decode_steps": steps,
        "total_generated_tokens": total_tokens,
        "prefill_seconds": secs(prefill_elapsed),
        "decode_seconds": secs(decode_elapsed),
        "aggregate_tokens_per_second": aggregate_tps,
        "per_stream_tokens_per_second": aggregate_tps / n as f64,
        "single_stream_equivalence_prefix": equiv,
        "stream0_text_head": decode_tokens(&tokenizer, &streams[0][..streams[0].len().min(32)])?,
    }));
    eprintln!(
        "streams={n}: aggregate {aggregate_tps:.0} tok/s ({:.1}/stream)",
        aggregate_tps / n as f64
    );
    }

    let report = serde_json::json!({
        "kind": "lmbrrr_multi_bench",
        "schema_version": 1,
        "model_id": args.model.model_id.as_str(),
        "device": format!("{device:?}"),
        "dtype": format!("{dtype:?}"),
        "load_seconds": secs(load_elapsed),
        "quantized_load": quantized_load_json(&quantized_load),
        "prompt_tokens": len,
        "max_new_tokens": args.max_new_tokens,
        "single_stream_reference_tokens": reference.generated_token_ids.len(),
        "rows": rows,
    });
    write_json_report(args.output.as_ref(), &report)
}

pub(crate) fn bench(args: BenchArgs) -> Result<()> {
    if args.iterations == 0 {
        anyhow::bail!("--iterations must be greater than zero");
    }

    let prompts = bench_prompts(&args);
    if prompts.is_empty() {
        anyhow::bail!("no benchmark prompts selected");
    }

    let bundle = resolve_artifacts(&args.model)?;
    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let tokenized = prompts
        .iter()
        .map(|prompt| {
            let prompt_text = chat_prompt(&prompt.text, 0, args.generation.enable_thinking);
            Ok((prompt.clone(), tokenize_prompt(&tokenizer, prompt_text)?))
        })
        .collect::<Result<Vec<_>>>()?;

    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
        head_loader_quant(&args.model),
    )?;
    maybe_restrict_head(&mut model, &args.model)?;
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());
    let mut writer = benchmark_writer(args.output.as_ref(), args.append)?;

    for (prompt, tokens) in &tokenized {
        for _ in 0..args.warmup {
            let _ = generate_tokens(
                &mut model,
                &device,
                &args.generation,
                tokens,
                None::<&ProcessedImages>,
                &args.model.downsample_mode,
                &eos_ids,
                |_, _, _, _| Ok(()),
            )?;
        }

        for iteration in 0..args.iterations {
            let stats = generate_tokens(
                &mut model,
                &device,
                &args.generation,
                tokens,
                None::<&ProcessedImages>,
                &args.model.downsample_mode,
                &eos_ids,
                |_, _, _, _| Ok(()),
            )?;
            let raw_text = decode_tokens(&tokenizer, &stats.generated_token_ids)?;
            let text = split_reasoning_text(&raw_text, args.generation.enable_thinking);
            serde_json::to_writer(
                &mut writer,
                &serde_json::json!({
                    "kind": "lmbrrr_benchmark",
                    "model_id": args.model.model_id.as_str(),
                    "revision": args.model.revision.as_str(),
                    "device": format!("{device:?}"),
                    "dtype": format!("{dtype:?}"),
                    "downsample_mode": args.model.downsample_mode.as_str(),
                    "profile": prompt.name.as_str(),
                    "iteration": iteration,
                    "warmup_iterations": args.warmup,
                    "prompt_tokens": stats.prompt_tokens,
                    "generated_tokens": stats.generated_tokens,
                    "total_tokens": stats.total_generated_tokens(),
                    "max_generated_tokens": args.generation.max_new_tokens,
                    "max_total_tokens": stats.prompt_tokens + args.generation.max_new_tokens,
                    "eos_reached": stats.eos_reached,
                    "prefill_seconds": secs(stats.prefill_elapsed),
                    "prefill_tokens_per_second": stats.prefill_tokens_per_second(),
                    "time_to_first_token_seconds": stats.time_to_first_token().map(secs),
                    "decode_time_to_first_token_seconds": stats.first_token_after_prefill.map(secs),
                    "decode_seconds": secs(stats.decode_elapsed),
                    "decode_model_input_tokens": stats.decode_model_tokens(),
                    "decode_model_seconds": secs(stats.decode_model_elapsed),
                    "decode_model_tokens_per_second": stats.decode_model_tokens_per_second(),
                    "decode_non_model_seconds": secs(stats.decode_non_model_elapsed()),
                    "decode_non_model_share": stats.decode_non_model_share(),
                    "sampling_seconds": secs(stats.sampling_elapsed),
                    "sampling_tokens_per_second": stats.sampling_tokens_per_second(),
                    "next_input_seconds": secs(stats.next_input_elapsed),
                    "callback_seconds": secs(stats.callback_elapsed),
                    "decode_bookkeeping_seconds": secs(stats.decode_bookkeeping_elapsed()),
                    "output_tokens_per_second": stats.decode_tokens_per_second(),
                    "decode_tokens_per_second": stats.decode_tokens_per_second(),
                    "steady_state_tokens_per_second": stats.steady_state_tokens_per_second(),
                    "artifact_seconds": secs(bundle.elapsed),
                    "load_seconds": secs(load_elapsed),
                    "tensor_count": bundle.weight_report.tensor_count,
                    "has_lm_head": bundle.weight_report.has_lm_head,
                    "quantized_load": quantized_load_json(&quantized_load),
                    "text": {
                        "raw": text.raw_text,
                        "reasoning": text.reasoning_text,
                        "answer": text.answer_text,
                    },
                    "generation": {
                        "max_new_tokens": args.generation.max_new_tokens,
                        "temperature": args.generation.temperature,
                        "top_p": args.generation.top_p,
                        "top_k": args.generation.top_k,
                        "seed": args.generation.seed,
                        "repeat_penalty": args.generation.repeat_penalty,
                        "repeat_last_n": args.generation.repeat_last_n,
                        "enable_thinking": args.generation.enable_thinking,
                    },
                }),
            )?;
            writer.write_all(b"\n")?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Static-batched N-stream greedy decode. Same prompt per stream (static
/// batching), batch dimension through the whole text path. The fused
/// DeltaNet decode kernel currently gates to b == 1, so batched steps take
/// the tensor path — dispatch counts amortize across streams, which is the
/// aggregate lane's core economics; kernel batching is the follow-up.

#[derive(Clone, Debug)]
struct BenchPrompt {
    name: String,
    text: String,
}

pub(crate) fn run(args: RunArgs) -> Result<()> {
    let bundle = resolve_artifacts(&args.model)?;

    if args.dry_run {
        println!(
            "{}",
            serde_json::json!({
                "model_id": args.model.model_id.as_str(),
                "revision": args.model.revision.as_str(),
                "config": bundle.artifacts.config,
                "tokenizer": bundle.artifacts.tokenizer,
                "weights": bundle.artifacts.weights,
                "quantized_manifest": args.model.quantized_manifest,
                "tensor_count": bundle.weight_report.tensor_count,
                "has_lm_head": bundle.weight_report.has_lm_head,
                "text_layers": bundle.config.text_config.num_hidden_layers,
                "vision_layers": bundle.config.vision_config.num_hidden_layers,
                "image_inputs": args.images.len(),
            })
        );
        return Ok(());
    }

    let device = select_device(args.model.cpu)?;
    let dtype = args.model.dtype.resolve(&device);
    let preprocessor = if args.images.is_empty() {
        None
    } else {
        let path = bundle
            .artifacts
            .preprocessor
            .as_ref()
            .context("image inputs require preprocessor_config.json")?;
        Some(PreprocessorConfig::from_path(path)?)
    };
    let processed_images = match (&preprocessor, args.images.is_empty()) {
        (Some(cfg), false) => Some(preprocess_paths(&args.images, cfg, &device)?),
        _ => None,
    };

    let tokenizer = load_tokenizer(&bundle.artifacts)?;
    let prompt_text = prepare_run_prompt(&args, preprocessor.as_ref(), processed_images.as_ref())?;
    let tokens = tokenize_prompt(&tokenizer, prompt_text)?;

    let (mut model, load_elapsed, quantized_load) = load_model_with_optional_quantization(
        &bundle,
        dtype,
        &device,
        args.model.quantized_manifest.as_ref(),
        head_loader_quant(&args.model),
    )?;
    maybe_restrict_head(&mut model, &args.model)?;
    let eos_ids = bundle.config.eos_ids(bundle.generation_config.as_ref());
    let mut stream = TokenOutputStream::new(tokenizer);
    let use_tui = !args.no_progress && std::io::stdout().is_terminal();
    let initial_channel = if args.generation.enable_thinking {
        TextChannel::Reasoning
    } else {
        TextChannel::Answer
    };
    let stats = if use_tui {
        let mut renderer = TuiOutput::new(
            tokens.len(),
            args.generation.max_new_tokens,
            initial_channel,
        )?;
        let stats = generate_tokens(
            &mut model,
            &device,
            &args.generation,
            &tokens,
            processed_images.as_ref(),
            &args.model.downsample_mode,
            &eos_ids,
            |next_token, generated, elapsed, prefill_elapsed| {
                if let Some(text) = stream.next_token(next_token)? {
                    renderer.write_chunk(&text, generated, elapsed, prefill_elapsed)?;
                }
                Ok(())
            },
        )?;

        if let Some(rest) = stream.decode_rest()? {
            renderer.write_chunk(
                &rest,
                stats.generated_tokens,
                stats.decode_elapsed,
                stats.prefill_elapsed,
            )?;
        }
        let final_text = renderer.finish(&stats)?;
        print_reasoning_parts(&final_text)?;
        stats
    } else {
        let mut renderer = ReasoningRenderer::new(initial_channel);
        let stats = generate_tokens(
            &mut model,
            &device,
            &args.generation,
            &tokens,
            processed_images.as_ref(),
            &args.model.downsample_mode,
            &eos_ids,
            |next_token, _, _, _| {
                if let Some(text) = stream.next_token(next_token)? {
                    renderer.write_chunk(&text)?;
                }
                Ok(())
            },
        )?;

        if let Some(rest) = stream.decode_rest()? {
            renderer.write_chunk(&rest)?;
        }
        renderer.finish()?;
        stats
    };

    eprintln!(
        "{}",
        serde_json::json!({
            "artifact_seconds": secs(bundle.elapsed),
            "load_seconds": secs(load_elapsed),
            "prefill_seconds": secs(stats.prefill_elapsed),
            "prefill_tokens_per_second": stats.prefill_tokens_per_second(),
            "time_to_first_token_seconds": stats.time_to_first_token().map(secs),
            "decode_time_to_first_token_seconds": stats.first_token_after_prefill.map(secs),
            "decode_seconds": secs(stats.decode_elapsed),
            "decode_model_input_tokens": stats.decode_model_tokens(),
            "decode_model_seconds": secs(stats.decode_model_elapsed),
            "decode_model_tokens_per_second": stats.decode_model_tokens_per_second(),
            "decode_non_model_seconds": secs(stats.decode_non_model_elapsed()),
            "decode_non_model_share": stats.decode_non_model_share(),
            "sampling_seconds": secs(stats.sampling_elapsed),
            "sampling_tokens_per_second": stats.sampling_tokens_per_second(),
            "next_input_seconds": secs(stats.next_input_elapsed),
            "callback_seconds": secs(stats.callback_elapsed),
            "decode_bookkeeping_seconds": secs(stats.decode_bookkeeping_elapsed()),
            "prompt_tokens": stats.prompt_tokens,
            "generated_tokens": stats.generated_tokens,
            "total_tokens": stats.total_generated_tokens(),
            "max_generated_tokens": args.generation.max_new_tokens,
            "max_total_tokens": stats.prompt_tokens + args.generation.max_new_tokens,
            "eos_reached": stats.eos_reached,
            "output_tokens_per_second": stats.decode_tokens_per_second(),
            "decode_tokens_per_second": stats.decode_tokens_per_second(),
            "steady_state_tokens_per_second": stats.steady_state_tokens_per_second(),
            "device": format!("{device:?}"),
            "dtype": format!("{dtype:?}"),
            "enable_thinking": args.generation.enable_thinking,
            "quantized_load": quantized_load_json(&quantized_load),
        })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bench_args(profiles: Vec<BenchProfile>, prompts: Vec<String>) -> BenchArgs {
        BenchArgs {
            model: ModelArgs {
                model_id: "model".to_string(),
                revision: "main".to_string(),
                downsample_mode: "16x".to_string(),
                cpu: true,
                dtype: DTypeArg::Auto,
                config: None,
                tokenizer: None,
                generation_config: None,
                preprocessor: None,
                weights: Vec::new(),
                quantized_manifest: None,
                quantize_lm_head: None,
                target_head_vocab_size: None,
                target_head_vocab_ranking: std::path::PathBuf::new(),
            },
            generation: GenerationArgs {
                max_new_tokens: 128,
                temperature: 0.0,
                top_p: None,
                top_k: None,
                seed: 299792458,
                repeat_penalty: 1.0,
                repeat_last_n: 64,
                enable_thinking: false,
            },
            profiles,
            prompts,
            warmup: 1,
            iterations: 3,
            output: None,
            append: false,
        }
    }

    #[test]
    fn bench_prompts_defaults_to_all_profiles() {
        let prompts = bench_prompts(&bench_args(Vec::new(), Vec::new()));
        let names = prompts
            .into_iter()
            .map(|prompt| prompt.name)
            .collect::<Vec<_>>();
        assert_eq!(names, ["short", "medium", "long"]);
    }

    #[test]
    fn bench_prompts_custom_prompt_does_not_add_defaults() {
        let prompts = bench_prompts(&bench_args(Vec::new(), vec!["hello".to_string()]));
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].name, "custom-1");
        assert_eq!(prompts[0].text, "hello");
    }

}
