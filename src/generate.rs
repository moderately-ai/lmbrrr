//! Greedy/sampled token generation over the MiniCPM runner: the device-chain
//! greedy fast path, generation statistics with the per-token budget
//! decomposition, and the shared sampling/argmax helpers.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use candle::{Device, Tensor, D};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    utils::apply_repeat_penalty,
};
use clap::Args;

use crate::image_processor::ProcessedImages;
use crate::minicpm::MiniCpmForConditionalGeneration;

#[derive(Args, Clone, Debug)]
pub struct GenerationArgs {
    #[arg(long, default_value_t = 128)]
    pub max_new_tokens: usize,

    #[arg(long, default_value_t = 0.0)]
    pub temperature: f64,

    #[arg(long)]
    pub top_p: Option<f64>,

    #[arg(long)]
    pub top_k: Option<usize>,

    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,

    #[arg(long, default_value_t = 1.0)]
    pub repeat_penalty: f32,

    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,

    #[arg(long)]
    pub enable_thinking: bool,
}

pub fn greedy_generation_args(max_new_tokens: usize, enable_thinking: bool) -> GenerationArgs {
    GenerationArgs {
        max_new_tokens,
        temperature: 0.0,
        top_p: None,
        top_k: None,
        seed: 299792458,
        repeat_penalty: 1.0,
        repeat_last_n: 64,
        enable_thinking,
    }
}

#[derive(Clone, Debug)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub generated_token_ids: Vec<u32>,
    pub eos_reached: bool,
    pub prefill_elapsed: Duration,
    pub decode_elapsed: Duration,
    pub decode_model_elapsed: Duration,
    pub sampling_elapsed: Duration,
    pub next_input_elapsed: Duration,
    pub callback_elapsed: Duration,
    pub first_token_after_prefill: Option<Duration>,
}

impl GenerationStats {
    pub fn total_generated_tokens(&self) -> usize {
        self.prompt_tokens + self.generated_tokens
    }

    pub fn time_to_first_token(&self) -> Option<Duration> {
        self.first_token_after_prefill
            .map(|decode| self.prefill_elapsed + decode)
    }

    pub fn prefill_tokens_per_second(&self) -> f64 {
        tokens_per_second(self.prompt_tokens, self.prefill_elapsed)
    }

    pub fn decode_tokens_per_second(&self) -> f64 {
        tokens_per_second(self.generated_tokens, self.decode_elapsed)
    }

    pub fn decode_model_tokens(&self) -> usize {
        self.generated_tokens.saturating_sub(1)
    }

    pub fn decode_model_tokens_per_second(&self) -> Option<f64> {
        let tokens = self.decode_model_tokens();
        (tokens > 0).then(|| tokens_per_second(tokens, self.decode_model_elapsed))
    }

    pub fn sampling_tokens_per_second(&self) -> f64 {
        tokens_per_second(self.generated_tokens, self.sampling_elapsed)
    }

    pub fn decode_non_model_elapsed(&self) -> Duration {
        self.decode_elapsed
            .saturating_sub(self.decode_model_elapsed)
    }

    pub fn decode_non_model_share(&self) -> f64 {
        if self.decode_elapsed.is_zero() {
            0.0
        } else {
            self.decode_non_model_elapsed().as_secs_f64() / self.decode_elapsed.as_secs_f64()
        }
    }

    pub fn decode_bookkeeping_elapsed(&self) -> Duration {
        self.decode_elapsed.saturating_sub(
            self.decode_model_elapsed
                + self.sampling_elapsed
                + self.next_input_elapsed
                + self.callback_elapsed,
        )
    }

    pub fn steady_state_tokens_per_second(&self) -> Option<f64> {
        let first = self.first_token_after_prefill?;
        let steady_elapsed = self.decode_elapsed.checked_sub(first)?;
        let steady_tokens = self.generated_tokens.checked_sub(1)?;
        if steady_tokens == 0 {
            None
        } else {
            Some(tokens_per_second(steady_tokens, steady_elapsed))
        }
    }
}

pub fn is_greedy_generation(generation: &GenerationArgs) -> bool {
    generation.temperature <= 0.0
}

pub fn sample_next_token(
    logits_processor: &mut Option<LogitsProcessor>,
    generation: &GenerationArgs,
    logits: &Tensor,
) -> Result<u32> {
    if is_greedy_generation(generation) {
        Ok(logits.argmax(D::Minus1)?.to_scalar::<u32>()?)
    } else {
        Ok(logits_processor
            .as_mut()
            .context("non-greedy generation requires a logits processor")?
            .sample(logits)?)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate_tokens(
    model: &mut MiniCpmForConditionalGeneration,
    device: &Device,
    generation: &GenerationArgs,
    input_tokens: &[u32],
    images: Option<&ProcessedImages>,
    downsample_mode: &str,
    eos_ids: &[u32],
    mut on_token: impl FnMut(u32, usize, Duration, Duration) -> Result<()>,
) -> Result<GenerationStats> {
    model.clear_cache();
    let mut tokens = input_tokens.to_vec();
    let mut logits_processor = (!is_greedy_generation(generation)).then(|| {
        LogitsProcessor::from_sampling(
            generation.seed,
            sampling(generation.temperature, generation.top_k, generation.top_p),
        )
    });

    let input = Tensor::from_slice(&tokens, (1, tokens.len()), device)?;
    let prefill_start = Instant::now();
    let mut logits = model.forward(&input, images, downsample_mode, 0)?;
    device.synchronize()?;
    let prefill_elapsed = prefill_start.elapsed();
    let mut position = tokens.len();
    let decode_start = Instant::now();
    let mut first_token_after_prefill = None::<Duration>;
    let mut generated = 0usize;
    let mut generated_token_ids = Vec::with_capacity(generation.max_new_tokens);
    let mut eos_reached = false;
    let mut decode_model_elapsed = Duration::ZERO;
    let mut sampling_elapsed = Duration::ZERO;
    let mut next_input_elapsed = Duration::ZERO;
    let mut callback_elapsed = Duration::ZERO;

    // Greedy fast path: the sampled token stays on device (u32 argmax feeds
    // the next forward directly), and ids are read back in batches of K —
    // removing both the per-token 4-byte upload (a fresh Metal buffer +
    // residency commit each step) and the per-token host sync, letting the
    // host run ahead of the GPU. Generation may run up to K-1 forwards past
    // EOS; the surplus is discarded and the next caller clears the cache.
    let device_chain =
        is_greedy_generation(generation) && (generation.repeat_penalty - 1.0).abs() < f32::EPSILON;
    if device_chain {
        const READBACK_EVERY: usize = 8;
        let mut pending: Vec<Tensor> = Vec::with_capacity(READBACK_EVERY);
        'outer: loop {
            let sampling_start = Instant::now();
            // argmax over [1, vocab] keeps the id rank-1 (cat/reshape need it).
            let next_id = logits.argmax(D::Minus1)?;
            pending.push(next_id.clone());
            sampling_elapsed += sampling_start.elapsed();

            let produced = generated_token_ids.len() + pending.len();
            let flush = pending.len() >= READBACK_EVERY || produced >= generation.max_new_tokens;
            if flush {
                let sampling_start = Instant::now();
                let refs = pending.iter().collect::<Vec<_>>();
                let ids = Tensor::cat(&refs, 0)?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<u32>()?;
                pending.clear();
                sampling_elapsed += sampling_start.elapsed();
                for id in ids {
                    if eos_ids.contains(&id) {
                        eos_reached = true;
                        break 'outer;
                    }
                    if first_token_after_prefill.is_none() {
                        first_token_after_prefill = Some(decode_start.elapsed());
                    }
                    generated_token_ids.push(id);
                    generated += 1;
                    let callback_start = Instant::now();
                    on_token(id, generated, decode_start.elapsed(), prefill_elapsed)?;
                    callback_elapsed += callback_start.elapsed();
                    if generated == generation.max_new_tokens {
                        break 'outer;
                    }
                }
            }
            if produced >= generation.max_new_tokens {
                break;
            }

            let decode_model_start = Instant::now();
            let input = next_id.reshape((1, 1))?;
            logits = model.forward(&input, None::<&ProcessedImages>, downsample_mode, position)?;
            decode_model_elapsed += decode_model_start.elapsed();
            position += 1;
        }
        return Ok(GenerationStats {
            prompt_tokens: input_tokens.len(),
            generated_tokens: generated,
            generated_token_ids,
            eos_reached,
            prefill_elapsed,
            decode_elapsed: decode_start.elapsed(),
            decode_model_elapsed,
            sampling_elapsed,
            next_input_elapsed,
            callback_elapsed,
            first_token_after_prefill,
        });
    }

    for _ in 0..generation.max_new_tokens {
        let sampling_start = Instant::now();
        let logits_1d = logits.squeeze(0)?;
        let logits_1d = if (generation.repeat_penalty - 1.0).abs() < f32::EPSILON {
            logits_1d
        } else {
            let start_at = tokens.len().saturating_sub(generation.repeat_last_n);
            apply_repeat_penalty(&logits_1d, generation.repeat_penalty, &tokens[start_at..])?
        };
        let next_token = sample_next_token(&mut logits_processor, generation, &logits_1d)?;
        sampling_elapsed += sampling_start.elapsed();

        if eos_ids.contains(&next_token) {
            eos_reached = true;
            break;
        }

        if first_token_after_prefill.is_none() {
            first_token_after_prefill = Some(decode_start.elapsed());
        }
        tokens.push(next_token);
        generated_token_ids.push(next_token);
        generated += 1;
        let callback_start = Instant::now();
        on_token(
            next_token,
            generated,
            decode_start.elapsed(),
            prefill_elapsed,
        )?;
        callback_elapsed += callback_start.elapsed();

        if generated == generation.max_new_tokens {
            break;
        }

        let input_start = Instant::now();
        let input = Tensor::from_slice(&[next_token], (1, 1), device)?;
        next_input_elapsed += input_start.elapsed();
        let decode_model_start = Instant::now();
        logits = model.forward(&input, None::<&ProcessedImages>, downsample_mode, position)?;
        // No synchronize here: the argmax readback in sample_next_token is the
        // only wait per token (an extra wait costs ~1-2ms of OS latency and
        // purges the Metal buffer pool). decode_model_elapsed is encode/queue
        // time; the GPU wait lands in sampling_elapsed.
        decode_model_elapsed += decode_model_start.elapsed();
        position += 1;
    }

    Ok(GenerationStats {
        prompt_tokens: input_tokens.len(),
        generated_tokens: generated,
        generated_token_ids,
        eos_reached,
        prefill_elapsed,
        decode_elapsed: decode_start.elapsed(),
        decode_model_elapsed,
        sampling_elapsed,
        next_input_elapsed,
        callback_elapsed,
        first_token_after_prefill,
    })
}

// No leading synchronize in either helper: to_scalar/to_vec1 already wait
// for the GPU, and a second wait per token costs a commit/fence cycle plus a
// buffer-pool purge. The returned Duration therefore covers any outstanding
// forward work too — callers use it for coarse reporting only.
pub fn argmax_token(logits: &Tensor, _device: &Device) -> Result<(u32, Duration)> {
    let started = Instant::now();
    let token = logits.squeeze(0)?.argmax(D::Minus1)?.to_scalar::<u32>()?;
    Ok((token, started.elapsed()))
}

pub fn argmax_tokens(logits: &Tensor, _device: &Device) -> Result<(Vec<u32>, Duration)> {
    let started = Instant::now();
    let tokens = logits
        .squeeze(0)?
        .argmax(D::Minus1)?
        .to_device(&Device::Cpu)?
        .to_vec1::<u32>()?;
    Ok((tokens, started.elapsed()))
}

pub fn sampling(temperature: f64, top_k: Option<usize>, top_p: Option<f64>) -> Sampling {
    if temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        match (top_k, top_p) {
            (None, None) => Sampling::All { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    }
}

pub fn secs(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

pub fn tokens_per_second(tokens: usize, duration: Duration) -> f64 {
    if tokens == 0 || duration.is_zero() {
        0.0
    } else {
        tokens as f64 / duration.as_secs_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_generation_is_temperature_driven() {
        let mut args = greedy_generation_args(128, false);
        args.temperature = 0.0;
        args.top_k = Some(4);
        args.top_p = Some(0.8);
        assert!(is_greedy_generation(&args));

        args.temperature = 0.7;
        assert!(!is_greedy_generation(&args));
    }

    #[test]
    fn generation_stats_separates_model_and_non_model_decode_time() {
        let stats = GenerationStats {
            prompt_tokens: 10,
            generated_tokens: 4,
            generated_token_ids: vec![1, 2, 3, 4],
            eos_reached: false,
            prefill_elapsed: Duration::from_millis(20),
            decode_elapsed: Duration::from_millis(100),
            decode_model_elapsed: Duration::from_millis(70),
            sampling_elapsed: Duration::from_millis(12),
            next_input_elapsed: Duration::from_millis(3),
            callback_elapsed: Duration::from_millis(5),
            first_token_after_prefill: Some(Duration::from_millis(10)),
        };

        assert_eq!(stats.decode_model_tokens(), 3);
        assert_eq!(stats.decode_model_tokens_per_second(), Some(3000.0 / 70.0));
        assert_eq!(stats.decode_non_model_elapsed(), Duration::from_millis(30));
        assert_eq!(
            stats.decode_bookkeeping_elapsed(),
            Duration::from_millis(10)
        );
    }
}
