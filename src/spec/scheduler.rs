//! Hardware-aware prefix admission (DSpark paper Appendix A) and its two
//! inputs: the measured round-cost model and the STS-calibrated survival
//! probabilities.

use std::path::Path;

use anyhow::{Context, Result};

/// Platt-style calibration of the confidence head (we own this; absent from
/// DeepSpec): p_accept = sigmoid(scale * logit + shift), fitted offline on
/// (logit, accepted) records and stored as sts.json in the drafter dir.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct StsPositionCalibration {
    pub scale: f32,
    pub shift: f32,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct StsCalibration {
    pub scale: f32,
    pub shift: f32,
    /// Per-position Platt parameters (STS left-to-right fit); the scheduler's
    /// cumulative survival is the product of these calibrated marginals —
    /// validated reliability within ~3 points across all bins.
    #[serde(default)]
    pub positions: Vec<StsPositionCalibration>,
}

impl StsCalibration {
    pub fn identity() -> Self {
        Self {
            scale: 1.0,
            shift: 0.0,
            positions: Vec::new(),
        }
    }

    pub fn load(drafter_dir: &Path) -> Result<Self> {
        let path = drafter_dir.join("sts.json");
        if !path.exists() {
            return Ok(Self::identity());
        }
        let file = std::fs::File::open(&path)
            .with_context(|| format!("open sts calibration {}", path.display()))?;
        Ok(serde_json::from_reader(file)
            .with_context(|| format!("parse sts calibration {}", path.display()))?)
    }

    pub fn probability(&self, logit: f32) -> f32 {
        1.0 / (1.0 + (-(self.scale * logit + self.shift)).exp())
    }

    pub fn position_probability(&self, position: usize, logit: f32) -> f32 {
        match self.positions.get(position.min(self.positions.len().saturating_sub(1))) {
            Some(p) if !self.positions.is_empty() => {
                1.0 / (1.0 + (-(p.scale * logit + p.shift)).exp())
            }
            _ => self.probability(logit),
        }
    }
}

/// Measured round costs (ms) for the scheduler's throughput objective.
/// verify_ms\[l\] is the chunk cost at length l = width + 1; defaults are the
/// post-fusion in-loop table (target/vt-gdc2.json + measured draft).
/// fixed_ms is the per-round host cost (readback waits, rollback bookkeeping,
/// encode overhead) that kernel-time tables miss — omitting it biases the
/// admission argmax narrow, because a constant amortizes over wider widths.
pub struct RoundCostModel {
    pub fixed_ms: f64,
    pub draft_ms: f64,
    pub verify_ms: Vec<f64>,
    /// Realized cost of a plain greedy step INSIDE the spec loop (the skip
    /// probe / no-draft round). Measured in-loop it runs ~0.7-1.1 ms over
    /// verify_ms\[1\]: with no draft to overlap behind, the per-round host
    /// cost shows up bare. This is the hysteresis's greedy counterfactual;
    /// understating it over-values skip mode by ~19% (2026-07-12, n=606
    /// no-draft rounds across the calibration split).
    pub greedy_step_ms: f64,
}

impl RoundCostModel {
    pub fn load(path: &Path) -> Result<Self> {
        #[derive(serde::Deserialize)]
        struct Artifact {
            default_draft_ms: f64,
            verify_ms_by_chunk_len: Vec<f64>,
            #[serde(default)]
            fixed_round_ms: f64,
            #[serde(default)]
            greedy_step_ms: Option<f64>,
        }
        let artifact: Artifact = serde_json::from_reader(
            std::fs::File::open(path)
                .with_context(|| format!("open cost model {}", path.display()))?,
        )
        .with_context(|| format!("parse cost model {}", path.display()))?;
        if artifact.verify_ms_by_chunk_len.len() < 3 {
            anyhow::bail!("cost model needs verify_ms for chunk lengths >= 2");
        }
        Ok(Self {
            fixed_ms: artifact.fixed_round_ms,
            draft_ms: artifact.default_draft_ms,
            // Legacy artifacts (no greedy_step_ms) keep the historical
            // kernel-only reference so their measured behaviour is unchanged.
            greedy_step_ms: artifact
                .greedy_step_ms
                .unwrap_or(artifact.verify_ms_by_chunk_len[1]),
            verify_ms: artifact.verify_ms_by_chunk_len,
        })
    }

    pub fn measured_default() -> Self {
        // Index by chunk length l (0 unused); interpolated from the
        // 2026-07-10 post-fusion verify table at short/medium context.
        let verify_ms = vec![
            0.0, 6.5, 13.9, 14.2, 14.5, 14.9, 15.2, 15.5, 15.7, 16.0, 16.4, 16.8, 17.2,
        ];
        Self {
            // In-loop drafted-round residuals vs the kernel table sit within
            // +/-0.7 ms of zero (2026-07-12); the width-dependent truth lives
            // in the table, not a constant, so the default stays 0.
            fixed_ms: 0.0,
            draft_ms: 5.0,
            greedy_step_ms: verify_ms[1],
            verify_ms,
        }
    }

    /// Kernel-time round cost (no fixed term): the contract the rate-based
    /// skip-hysteresis was validated on — its greedy counterfactual
    /// (verify_ms\[1\]) is also kernel-only, so the comparison stays
    /// like-with-like there while the admission objective below carries the
    /// full cost.
    pub fn kernel_ms(&self, width: usize) -> f64 {
        self.draft_ms + self.verify_kernel_ms(width + 1)
    }

    /// Table verify cost at chunk length l (clamped to the table tail).
    pub fn verify_kernel_ms(&self, l: usize) -> f64 {
        self.verify_ms[l.min(self.verify_ms.len() - 1)]
    }

    pub fn t_round_ms(&self, width: usize) -> f64 {
        self.fixed_ms + self.kernel_ms(width)
    }
}

/// DSpark hardware-aware prefix admission (paper Appendix A): scan positions
/// left to right, admitting while expected throughput improves, and STOP at
/// the first non-improving position without reading the next confidence —
/// c_{k+1} is a function of the realized token x_k, so looking ahead would
/// introduce retrospective selection bias. `survival_probs` must therefore
/// be lazy; this function reads exactly `admitted + 1` items (or fewer at
/// gamma).
pub fn schedule_prefix_width(
    mut survival_probs: impl Iterator<Item = f64>,
    t_round_ms: impl Fn(usize) -> f64,
    gamma: usize,
) -> usize {
    let mut best_width = 0usize;
    let mut best = 1.0 / t_round_ms(0);
    let mut survival = 1.0f64;
    let mut expected = 1.0f64; // bonus token
    for k in 1..=gamma {
        let Some(p_k) = survival_probs.next() else {
            break;
        };
        survival *= p_k;
        expected += survival;
        let f = expected / t_round_ms(k);
        if f > best {
            best = f;
            best_width = k;
        } else {
            break;
        }
    }
    best_width
}

/// Mean committed tokens per round (accepted drafts + bonus) straight from
/// the acceptance histogram — exact, unlike committed.len()/rounds.
pub fn mean_committed_per_round(accepted_histogram: &[usize], rounds: usize) -> f64 {
    if rounds == 0 {
        return 0.0;
    }
    let committed: usize = accepted_histogram
        .iter()
        .enumerate()
        .map(|(accepted, count)| (accepted + 1) * count)
        .sum();
    committed as f64 / rounds as f64
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;
