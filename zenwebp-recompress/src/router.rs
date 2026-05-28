//! Decision router.
//!
//! Given the source analysis, the calibration table, and the target, pick
//! the strategy that:
//!
//! 1. Projects zensim-A at-or-above target, AND
//! 2. Projects the smallest output size (`output_len / input_len`), AND
//! 3. Has projected output size strictly < input size.
//!
//! If no strategy meets (3), fall back to `LosslessRemux`. If the source
//! already projects at-or-above target with low room to shrink, return
//! `NoOp`.

use crate::api::{
    Budget, LosslessReason, NoOpReason, RecompressOptions, RecompressResult, StrategyKind,
};
use crate::budget::Stopwatch;
use crate::calibration::{AllEstimates, CalibrationLookup, CellEstimate};
use crate::error::Error;
use crate::source::{SourceAnalysis, SourceKind};
use crate::strategies;
use crate::target::source_q_to_zensim_a_estimate;

/// Band around target zensim-A where the source is considered "already
/// good enough" so we ship a no-op rather than risking an undershoot.
pub(crate) const ZENSIM_A_NOOP_BAND: f32 = 1.0;

/// Maximum output/input size ratio below which JXL handoff is *not*
/// suggested. Above this, all standard strategies are projected to grow
/// the file or fail to meet target, and JXL is likely the better fit.
const JXL_HANDOFF_GROWTH_THRESHOLD: f32 = 0.95;

/// Output of [`decide_strategy`].
#[derive(Debug, Clone)]
#[allow(dead_code)] // estimates exposed via expert API for callers inspecting routing.
pub struct RouterDecision {
    /// What to do.
    pub action: Action,
    /// Hint for the caller.
    pub better_handled_by_jxl: bool,
    /// Per-strategy projections that fed the decision.
    pub estimates: AllEstimates,
}

#[derive(Debug, Clone, Copy)]
pub enum Action {
    NoOp(NoOpReason),
    Recompress {
        strategy: StrategyKind,
        estimate: CellEstimate,
    },
    LosslessOnly(LosslessReason),
}

/// Decide the strategy purely from analysis + calibration. Pure function;
/// does not touch pixels.
pub fn decide_strategy(analysis: &SourceAnalysis, opts: &RecompressOptions) -> RouterDecision {
    let table = CalibrationLookup::load();
    let estimates = table.project_all(analysis, opts.target_zensim_a);

    // 1. Animated → always lossless remux.
    if analysis.lossy_recompression_unsafe() {
        return RouterDecision {
            action: Action::LosslessOnly(LosslessReason::SourceIsAnimated),
            better_handled_by_jxl: false,
            estimates,
        };
    }

    // 1b. Lossless source → re-encoding as VP8L is always quality-safe
    //     (lossless → lossless). The only question is whether our encoder
    //     beats the source's. We can't know without trying, and the
    //     projection is conservatively 1.0, so dispatch LosslessReencode
    //     speculatively: the dispatcher keeps the result only if it
    //     actually shrinks, else falls back to LosslessRemux. No target
    //     gating — quality is preserved by construction.
    if matches!(analysis.kind, SourceKind::LosslessVp8L) {
        let est = estimates.lossless_reencode.unwrap_or(CellEstimate {
            projected_zensim_a: 100.0,
            projected_size_ratio: 1.0,
            ci_low_zensim_a: 100.0,
            ci_high_zensim_a: 100.0,
        });
        return RouterDecision {
            action: Action::Recompress {
                strategy: StrategyKind::LosslessReencode,
                estimate: est,
            },
            better_handled_by_jxl: false,
            estimates,
        };
    }

    // 2. Source already at-or-above target with little slack? NoOp.
    let source_zensim_a = source_q_to_zensim_a_estimate(analysis.encoder_family, analysis.source_q);
    if source_zensim_a < opts.target_zensim_a + ZENSIM_A_NOOP_BAND
        && source_zensim_a >= opts.target_zensim_a - ZENSIM_A_NOOP_BAND
    {
        return RouterDecision {
            action: Action::NoOp(NoOpReason::SourceAlreadyMeetsTarget),
            better_handled_by_jxl: false,
            estimates,
        };
    }

    // 3. Find the smallest at-or-above-target candidate that shrinks the
    //    file. First try strict matching; if either no candidates OR no
    //    candidate shrinks the file, retry with the tolerance band so we
    //    can accept a slight target undershoot in exchange for actual
    //    size savings.
    let strict = filter_candidates(&estimates, opts.target_zensim_a);
    let strict_can_shrink = strict.iter().any(|(_, e)| e.projected_size_ratio < 1.0);
    let candidates = if strict_can_shrink {
        strict
    } else {
        let relaxed = filter_candidates(
            &estimates,
            opts.target_zensim_a - opts.tolerance_below_target.max(0.0),
        );
        if relaxed.iter().any(|(_, e)| e.projected_size_ratio < 1.0) {
            relaxed
        } else {
            // Neither strict nor relaxed candidates produce a shrink;
            // pass strict (possibly empty) through so we report the
            // honest NoStrategyMeetsTarget downstream.
            strict
        }
    };

    if candidates.is_empty() {
        // No strategy meets the target. If source is at-or-above target
        // already we ship LosslessRemux; otherwise we ship lossless (we
        // cannot shrink without dropping below target).
        let reason = if source_zensim_a >= opts.target_zensim_a {
            LosslessReason::NoStrategyShrinksFile
        } else {
            LosslessReason::NoStrategyMeetsTarget
        };
        let jxl_hint = analysis.source_q <= 35.0
            && matches!(analysis.kind, SourceKind::LossyVp8)
            && estimates
                .reencode
                .map(|e| e.projected_size_ratio >= JXL_HANDOFF_GROWTH_THRESHOLD)
                .unwrap_or(true);
        return RouterDecision {
            action: Action::LosslessOnly(reason),
            better_handled_by_jxl: jxl_hint,
            estimates,
        };
    }

    let (strategy, estimate) = candidates
        .iter()
        .min_by(|a, b| {
            a.1.projected_size_ratio
                .partial_cmp(&b.1.projected_size_ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
        .expect("non-empty above");

    if estimate.projected_size_ratio >= 1.0 {
        // Smallest valid strategy still grows the file; fall back to remux.
        return RouterDecision {
            action: Action::LosslessOnly(LosslessReason::NoStrategyShrinksFile),
            better_handled_by_jxl: false,
            estimates,
        };
    }

    RouterDecision {
        action: Action::Recompress { strategy, estimate },
        better_handled_by_jxl: false,
        estimates,
    }
}

fn filter_candidates(
    estimates: &AllEstimates,
    target_zensim_a: f32,
) -> Vec<(StrategyKind, CellEstimate)> {
    estimates
        .iter()
        .filter(|(_, e)| e.projected_zensim_a >= target_zensim_a)
        // LosslessRemux is the fallback path — only consider it when no
        // recompression strategy beats it, handled separately below.
        .filter(|(k, _)| !matches!(k, StrategyKind::LosslessRemux))
        // CoeffEdit is not yet implemented; never let the router pick it.
        // When the implementation lands, drop this filter.
        .filter(|(k, _)| !matches!(k, StrategyKind::CoeffEdit))
        // DeblockReencode is measured-dominated by Reencode in every tested
        // source config (default-filtered: −2.75 zensim/+3.9% size;
        // weak-filtered: −3.35; worst-blocking qi≥90: −5.09, wins 0/60
        // cells). Post-decode spatial smoothing moves the image away from
        // the sharp original reference. See
        // benchmarks/deblock_experiment_2026-05-28.md. The filter remains a
        // tested building block via expert::deblock_rgba; the router never
        // selects the strategy until a config is found where it wins.
        .filter(|(k, _)| !matches!(k, StrategyKind::DeblockReencode))
        .collect()
}

/// Run the chosen strategy. Honors [`Budget`] — under `OneShot` we run
/// the pure strategy once; under `MaxIterations` we measure-and-refine
/// via secant step on libwebp_q.
pub fn dispatch(
    webp_bytes: &[u8],
    analysis: &SourceAnalysis,
    decision: RouterDecision,
    opts: &RecompressOptions,
) -> Result<RecompressResult, Error> {
    let stopwatch = Stopwatch::from_budget(opts.budget);

    match decision.action {
        Action::NoOp(reason) => Ok(RecompressResult::NoOp { reason }),
        Action::LosslessOnly(reason) => {
            let bytes = strategies::lossless_remux::run_lossless_remux(webp_bytes, analysis)?;
            Ok(RecompressResult::LosslessOnly {
                bytes,
                reason,
                better_handled_by_jxl: decision.better_handled_by_jxl,
            })
        }
        Action::Recompress { strategy, estimate } => {
            let (bytes, measured) = match opts.budget {
                Budget::OneShot => (run_strategy(strategy, webp_bytes, analysis, opts)?, None),
                Budget::MaxIterations(n) if n <= 1 => {
                    let b = run_strategy(strategy, webp_bytes, analysis, opts)?;
                    let m = crate::measure::score_recompression(webp_bytes, &b).ok();
                    (b, m)
                }
                Budget::MaxIterations(n) => {
                    iterate_secant(strategy, webp_bytes, analysis, opts, n.min(8), &stopwatch)?
                }
                Budget::MaxTime(_) => {
                    iterate_secant(strategy, webp_bytes, analysis, opts, 8, &stopwatch)?
                }
            };
            let ratio = if !webp_bytes.is_empty() {
                bytes.len() as f32 / webp_bytes.len() as f32
            } else {
                1.0
            };

            // Speculative strategies (LosslessReencode on a lossless source,
            // dispatched without target gating) must only ship if they
            // actually shrink the file. Otherwise fall back to a clean
            // re-mux — the source bytes are already optimal.
            if ratio >= 1.0 && matches!(strategy, StrategyKind::LosslessReencode) {
                let remux = strategies::lossless_remux::run_lossless_remux(webp_bytes, analysis)?;
                return Ok(RecompressResult::LosslessOnly {
                    bytes: remux,
                    reason: LosslessReason::NoStrategyShrinksFile,
                    better_handled_by_jxl: decision.better_handled_by_jxl,
                });
            }

            Ok(RecompressResult::Recompressed {
                bytes,
                strategy,
                projected_zensim_a: estimate.projected_zensim_a,
                measured_zensim_a: measured,
                source_to_output_ratio: ratio,
                better_handled_by_jxl: decision.better_handled_by_jxl,
            })
        }
    }
}

/// Iterative secant search for a libwebp_q that achieves
/// `target_zensim_a` generation-loss within the tolerance band. Returns
/// the best `(bytes, measured_zensim_a)` seen.
///
/// The "measured" we have is `zensim_a_vs_source` — the generation-loss
/// signal. Cumulative-vs-reference is bounded by `min(gen_loss,
/// source_cum)`; since `source_cum` is fixed for a given source, hitting
/// `gen_loss >= target_zensim_a` is the best we can do at runtime.
fn iterate_secant(
    strategy: StrategyKind,
    webp_bytes: &[u8],
    analysis: &SourceAnalysis,
    opts: &RecompressOptions,
    max_passes: u32,
    stopwatch: &Stopwatch,
) -> Result<(Vec<u8>, Option<f32>), Error> {
    use crate::strategies;

    // Strategies that don't have a q dial: one-shot + measure.
    if !matches!(
        strategy,
        StrategyKind::Reencode | StrategyKind::DeblockReencode
    ) {
        let b = run_strategy(strategy, webp_bytes, analysis, opts)?;
        let m = crate::measure::score_recompression(webp_bytes, &b).ok();
        return Ok((b, m));
    }

    let target_t = opts.target_zensim_a;
    let tol = opts.tolerance_below_target.max(0.5);
    let ship_overshoot: f32 = 2.0;

    // Initial q from anchor table.
    let mut q_curr = crate::target::target_zensim_a_to_libwebp_q(target_t);
    let mut q_prev: Option<i32> = None;
    let mut m_prev: Option<f32> = None;
    let mut best: Option<(Vec<u8>, f32)> = None;

    for _ in 0..max_passes {
        if stopwatch.expired() {
            break;
        }
        let bytes = match strategy {
            StrategyKind::Reencode => {
                strategies::reencode::run_reencode_at_q(webp_bytes, analysis, q_curr)?
            }
            StrategyKind::DeblockReencode => {
                strategies::deblock_reencode::run_deblock_reencode_at_q(
                    webp_bytes, analysis, q_curr,
                )?
            }
            _ => unreachable!(),
        };
        let m = match crate::measure::score_recompression(webp_bytes, &bytes) {
            Ok(v) if v.is_finite() => v,
            _ => {
                // No measurement available; ship what we have.
                return Ok((bytes, None));
            }
        };

        // Track best — closer-to-target preferred, prefer smaller bytes on tie.
        let better = match &best {
            None => true,
            Some((b, m_best)) => {
                let d_curr = (m - target_t).abs();
                let d_best = (*m_best - target_t).abs();
                d_curr < d_best || ((d_curr - d_best).abs() < 0.25 && bytes.len() < b.len())
            }
        };
        if better {
            best = Some((bytes.clone(), m));
        }

        // Ship if within band.
        if m >= target_t - tol && m <= target_t + ship_overshoot {
            return Ok((bytes, Some(m)));
        }

        // Secant step on q.
        let q_next = match (q_prev, m_prev) {
            (Some(qp), Some(mp)) if (m - mp).abs() > 0.5 => {
                let dq_dm = (q_curr as f32 - qp as f32) / (m - mp);
                let raw = q_curr as f32 + (target_t - m) * dq_dm;
                raw.clamp(1.0, 100.0).round() as i32
            }
            _ => {
                // First step: jump by fixed amount in the right direction.
                let step: i32 = if m < target_t { 8 } else { -6 };
                (q_curr as i32 + step).clamp(1, 100)
            }
        };
        if q_next as u8 == q_curr {
            // Step didn't move; ship best.
            break;
        }
        q_prev = Some(q_curr as i32);
        m_prev = Some(m);
        q_curr = q_next as u8;
    }

    let (bytes, m) = best.expect("at least one iteration ran");
    Ok((bytes, Some(m)))
}

fn run_strategy(
    strategy: StrategyKind,
    webp_bytes: &[u8],
    analysis: &SourceAnalysis,
    opts: &RecompressOptions,
) -> Result<Vec<u8>, Error> {
    use crate::strategies;
    match strategy {
        StrategyKind::LosslessRemux => {
            strategies::lossless_remux::run_lossless_remux(webp_bytes, analysis)
        }
        StrategyKind::CoeffEdit => {
            strategies::coeff_edit::run_coeff_edit(webp_bytes, analysis, opts)
        }
        StrategyKind::Reencode => strategies::reencode::run_reencode(webp_bytes, analysis, opts),
        StrategyKind::DeblockReencode => {
            strategies::deblock_reencode::run_deblock_reencode(webp_bytes, analysis, opts)
        }
        StrategyKind::LosslessReencode => {
            strategies::lossless_reencode::run_lossless_reencode(webp_bytes, analysis)
        }
    }
}
