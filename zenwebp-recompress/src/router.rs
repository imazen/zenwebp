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
use crate::calibration::{AllEstimates, CalibrationLookup, CellEstimate, data};
use crate::error::Error;
use crate::source::{SourceAnalysis, SourceKind};
use crate::strategies;

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
            chosen_libwebp_q: None,
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
    // `source_cum` is the source's own cumulative zensim-A vs the original,
    // from the decode-based effective quality (NOT the header quantizer).
    let source_zensim_a = if matches!(analysis.kind, SourceKind::LosslessVp8L) {
        100.0
    } else {
        data::source_cum(analysis.estimated_quality)
    };
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
        // JXL handoff hint: a low-quality lossy source where WebP re-encode
        // can't shrink at target. Keyed on the decode-based effective
        // quality (header quantizer is unreliable).
        let jxl_hint = matches!(analysis.kind, SourceKind::LossyVp8)
            && analysis.estimated_quality <= 45.0
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
            let chosen_q = estimate.chosen_libwebp_q;
            let (bytes, measured) = match opts.budget {
                Budget::OneShot => (
                    run_strategy(strategy, webp_bytes, analysis, opts, chosen_q)?,
                    None,
                ),
                Budget::MaxIterations(n) if n <= 1 => {
                    let b = run_strategy(strategy, webp_bytes, analysis, opts, chosen_q)?;
                    let m = crate::measure::score_recompression(webp_bytes, &b).ok();
                    (b, m)
                }
                Budget::MaxIterations(n) => minimize_size(
                    strategy,
                    webp_bytes,
                    analysis,
                    opts,
                    chosen_q,
                    n.min(8),
                    &stopwatch,
                )?,
                Budget::MaxTime(_) => minimize_size(
                    strategy, webp_bytes, analysis, opts, chosen_q, 8, &stopwatch,
                )?,
            };
            let ratio = if !webp_bytes.is_empty() {
                bytes.len() as f32 / webp_bytes.len() as f32
            } else {
                1.0
            };

            // GROUND-TRUTH SIZE GUARD. The router picks a strategy from
            // *projected* size ratios, but projections are p50 estimates —
            // a particular image can grow even when the cell median
            // shrinks. The goal is explicit: "Sometimes reencoding with any
            // strategy will make a file larger at the target zensim value,
            // at which point only lossless optimization should be done." So
            // whatever the projection said, if the ACTUAL output is not
            // smaller than the source, ship a clean re-mux instead. Applies
            // to every recompression strategy, not just the speculative
            // lossless path. We already hold `bytes`, so the only extra cost
            // is a metadata-only re-mux.
            if ratio >= 1.0 {
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

/// Real-size-minimizing local search for the measured budgets.
///
/// At runtime we CANNOT measure the cumulative-vs-original target (the
/// reference is gone) — so we trust the calibration's `chosen_q` for which
/// quality hits the cumulative target, and use the budget to minimize the
/// ACTUAL output size, which we *can* measure. We try `chosen_q` and a few
/// lower qualities (lower q ⇒ smaller, but the model cumulative drops), and
/// keep the smallest real encode whose MODEL cumulative still meets the
/// target. This corrects per-image size variance the p50 table misses
/// without pretending to measure cumulative.
///
/// Returns `(bytes, measured_gen_loss_vs_source)`. The measured value is
/// the generation-loss vs the source (the only measurable signal), exposed
/// for transparency — it is NOT the cumulative target.
fn minimize_size(
    strategy: StrategyKind,
    webp_bytes: &[u8],
    analysis: &SourceAnalysis,
    opts: &RecompressOptions,
    chosen_q: Option<u8>,
    max_passes: u32,
    stopwatch: &Stopwatch,
) -> Result<(Vec<u8>, Option<f32>), Error> {
    // Strategies without a quality dial (remux, vp8l, coeff): single run.
    let dialable = matches!(
        strategy,
        StrategyKind::Reencode | StrategyKind::DeblockReencode
    );
    let Some(start_q) = chosen_q.filter(|_| dialable) else {
        let b = run_strategy(strategy, webp_bytes, analysis, opts, chosen_q)?;
        let m = crate::measure::score_recompression(webp_bytes, &b).ok();
        return Ok((b, m));
    };

    let target = opts.target_zensim_a;
    let eff_q = analysis.estimated_quality;
    // Candidate qualities: chosen_q, then step down by 3 (lower q ⇒ smaller
    // file). Each candidate is gated by `model_cum_meets` inside the loop,
    // so we never keep one whose modeled cumulative drops below target.
    let mut candidates = vec![start_q];
    let mut q = start_q as i32 - 3;
    while candidates.len() < max_passes as usize && q >= 1 {
        candidates.push(q as u8);
        q -= 3;
    }

    let mut best: Option<(Vec<u8>, u8)> = None;
    for &cq in &candidates {
        if stopwatch.expired() {
            break;
        }
        let bytes = match strategy {
            StrategyKind::Reencode => {
                strategies::reencode::run_reencode_at_q(webp_bytes, analysis, cq)?
            }
            StrategyKind::DeblockReencode => {
                strategies::deblock_reencode::run_deblock_reencode_at_q(webp_bytes, analysis, cq)?
            }
            _ => unreachable!(),
        };
        // Keep this candidate only if it actually shrinks AND its model
        // cumulative (at the encode q, for this source) still meets target.
        let shrinks = bytes.len() < webp_bytes.len();
        let model_ok = model_cum_meets(eff_q, cq, target);
        let take = shrinks
            && model_ok
            && match &best {
                None => true,
                Some((b, _)) => bytes.len() < b.len(),
            };
        if take {
            best = Some((bytes, cq));
        }
    }

    match best {
        Some((bytes, _)) => {
            let m = crate::measure::score_recompression(webp_bytes, &bytes).ok();
            Ok((bytes, m))
        }
        // Nothing shrank while meeting target — run chosen_q once so the
        // size guard can fall it back to a remux.
        None => {
            let b = strategies::reencode::run_reencode_at_q(webp_bytes, analysis, start_q)
                .or_else(|_| run_strategy(strategy, webp_bytes, analysis, opts, chosen_q))?;
            let m = crate::measure::score_recompression(webp_bytes, &b).ok();
            Ok((b, m))
        }
    }
}

/// Does re-encoding a source of effective quality `eff_q` at libwebp `q`
/// project a cumulative zensim-A at-or-above `target`? Uses the calibration
/// table's cumulative column nearest `q`.
fn model_cum_meets(eff_q: f32, q: u8, target: f32) -> bool {
    // Find the table column for this q (nearest grid point) and read the
    // interpolated cumulative for eff_q.
    let grid = data::TARGET_Q_GRID;
    let mut nearest = 0usize;
    let mut bestd = i32::MAX;
    for (i, &g) in grid.iter().enumerate() {
        let d = (g as i32 - q as i32).abs();
        if d < bestd {
            bestd = d;
            nearest = i;
        }
    }
    // Interpolate cumulative over eff_q at that column via best_reencode's
    // machinery: reconstruct by scanning. Simpler: use source_cum ceiling
    // and the column cumulative.
    let cum = data::reencode_cum_at(eff_q, nearest);
    cum + 1e-3 >= target
}

fn run_strategy(
    strategy: StrategyKind,
    webp_bytes: &[u8],
    analysis: &SourceAnalysis,
    opts: &RecompressOptions,
    chosen_q: Option<u8>,
) -> Result<Vec<u8>, Error> {
    use crate::strategies;
    match strategy {
        StrategyKind::LosslessRemux => {
            strategies::lossless_remux::run_lossless_remux(webp_bytes, analysis)
        }
        StrategyKind::CoeffEdit => {
            strategies::coeff_edit::run_coeff_edit(webp_bytes, analysis, opts)
        }
        StrategyKind::Reencode => match chosen_q {
            Some(q) => strategies::reencode::run_reencode_at_q(webp_bytes, analysis, q),
            None => strategies::reencode::run_reencode(webp_bytes, analysis, opts),
        },
        StrategyKind::DeblockReencode => match chosen_q {
            Some(q) => {
                strategies::deblock_reencode::run_deblock_reencode_at_q(webp_bytes, analysis, q)
            }
            None => strategies::deblock_reencode::run_deblock_reencode(webp_bytes, analysis, opts),
        },
        StrategyKind::LosslessReencode => {
            strategies::lossless_reencode::run_lossless_reencode(webp_bytes, analysis)
        }
    }
}
