use super::*;

/// Paper Appendix A non-anticipation scenario: a1 = 0.8 with SPS
/// throughputs {1.0, 0.5, 0.45}. f(0) = 1*1.0 = 1.0; f(1) = 1.8*0.5 =
/// 0.9 — non-improving, so the scheduler must return width 0 WITHOUT
/// ever evaluating c_2 (it is a function of the realized token x_1).
#[test]
fn scheduler_appendix_a_non_anticipation() {
    use std::cell::Cell;
    let reads = Cell::new(0usize);
    let probs = [0.8f64, f64::NAN /* reading this is the bug */]
        .into_iter()
        .inspect(|_| reads.set(reads.get() + 1));
    // Encode SPS as throughput via t_round = E_max_possible / SPS-style
    // rates: t(0)=1/1.0, t(1)=1/0.5, t(2)=1/0.45.
    let sps = [1.0f64, 0.5, 0.45];
    let width = schedule_prefix_width(probs, |w| 1.0 / sps[w], 2);
    assert_eq!(width, 0);
    assert_eq!(reads.get(), 1, "c_2 must never be read");
}

/// A fixed per-round cost amortizes over wider widths: admitting the
/// second row needs t2/t1 < E2/E1 = 2.35/1.9; at t = [10.0, 10.2, 12.8]
/// the ratio 12.8/10.2 fails that bare test but (12.8+2)/(10.2+2)
/// passes it. Omitting c therefore biases admission narrow — the
/// confirmed contract gap behind the truthful-STS regression.
#[test]
fn scheduler_fixed_cost_amortizes_toward_wider_widths() {
    let probs = [0.9f64, 0.5];
    let verify = [10.0f64, 10.2, 12.8];
    let without_c = schedule_prefix_width(
        probs.into_iter(),
        |w| verify[w],
        2,
    );
    let with_c = schedule_prefix_width(
        probs.into_iter(),
        |w| 2.0 + verify[w],
        2,
    );
    assert_eq!(without_c, 1, "c = 0 must reject the marginal second row");
    assert_eq!(with_c, 2, "c = 2 must admit the same second row");
}

/// Artifacts without fixed_round_ms (all pre-2026-07-12 cost models)
/// must parse with fixed_ms = 0.0; artifacts carrying the field must
/// load it into t_round_ms.
#[test]
fn cost_model_artifact_fixed_round_ms_back_compat() {
    let dir = std::env::temp_dir().join(format!(
        "lmbrrr-cost-model-test-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let legacy = dir.join("legacy.json");
    std::fs::write(
        &legacy,
        r#"{"default_draft_ms": 5.0, "verify_ms_by_chunk_len": [0.0, 6.5, 13.9, 14.2]}"#,
    )
    .unwrap();
    let model = RoundCostModel::load(&legacy).unwrap();
    assert_eq!(model.fixed_ms, 0.0);
    assert_eq!(model.t_round_ms(1), 5.0 + 13.9);
    assert_eq!(model.kernel_ms(1), model.t_round_ms(1));
    // Legacy hysteresis reference: the kernel-only l=1 entry.
    assert_eq!(model.greedy_step_ms, 6.5);

    let with_fixed = dir.join("fixed.json");
    std::fs::write(
        &with_fixed,
        r#"{"default_draft_ms": 5.0, "verify_ms_by_chunk_len": [0.0, 6.5, 13.9, 14.2], "fixed_round_ms": 1.25, "greedy_step_ms": 7.4}"#,
    )
    .unwrap();
    let model = RoundCostModel::load(&with_fixed).unwrap();
    assert_eq!(model.fixed_ms, 1.25);
    assert_eq!(model.t_round_ms(1), 1.25 + 5.0 + 13.9);
    assert_eq!(model.kernel_ms(1), 5.0 + 13.9);
    assert_eq!(model.greedy_step_ms, 7.4);
    std::fs::remove_dir_all(&dir).ok();
}

/// Positions beyond the fitted range clamp to the LAST per-position fit
/// (deep positions behave like the deepest fitted one, not the global
/// fallback); an empty positions vec falls back to the global fit.
#[test]
fn sts_position_probability_clamps_to_last_fit() {
    let sts = StsCalibration {
        scale: 1.0,
        shift: 0.0,
        positions: vec![
            StsPositionCalibration { scale: 2.0, shift: 0.0 },
            StsPositionCalibration { scale: 0.5, shift: 1.0 },
        ],
    };
    assert_eq!(sts.position_probability(1, 2.0), sts.position_probability(7, 2.0));
    assert!(sts.position_probability(0, 2.0) != sts.position_probability(7, 2.0));

    let global_only = StsCalibration {
        scale: 1.0,
        shift: 0.0,
        positions: Vec::new(),
    };
    assert_eq!(global_only.position_probability(3, 1.5), global_only.probability(1.5));
}

/// Committed per round = accepted + bonus, exactly from the histogram.
#[test]
fn mean_committed_per_round_from_histogram() {
    // 3 rounds at 0 accepted, 2 at 2, 1 at 5 -> (3*1 + 2*3 + 1*6) / 6.
    let histogram = [3usize, 0, 2, 0, 0, 1];
    assert_eq!(mean_committed_per_round(&histogram, 6), 15.0 / 6.0);
    assert_eq!(mean_committed_per_round(&histogram, 0), 0.0);
}

/// Improving throughput admits positions and stops at the first decline.
#[test]
fn scheduler_admits_while_improving() {
    // Flat costs: admitting high-probability positions always improves.
    let width = schedule_prefix_width(
        [0.9f64, 0.8, 0.1, 0.9].into_iter(),
        |w| 10.0 + 0.5 * w as f64,
        4,
    );
    // f grows through k=2 (0.9, then +0.72), k=3 adds only 0.072
    // expected tokens for +0.5ms — declining; must stop at 2 and never
    // read the fourth probability.
    assert_eq!(width, 2);
}
