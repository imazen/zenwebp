//! Frozen public API.
//!
//! This module pins the minimal contract for [`recompress`]:
//! one entry point, three enums, and the option/result structs.
//!
//! See [`crate::DESIGN`](crate) for the full design rationale. The summary:
//! - [`RecompressOptions`] specifies the perceptual target and budget.
//! - [`RecompressResult`] tells the caller what happened, including the
//!   strategy chosen and a projected score. Use [`Budget::MaxIterations`] or
//!   [`Budget::MaxTime`] to ask for a *measured* score; the default
//!   [`Budget::OneShot`] returns a model-projected score only.

use crate::error::Error;
use crate::router;
use crate::source;

/// Recompression budget.
///
/// The default is [`Budget::OneShot`]: no measurements, no IQA loop. This is
/// what production servers should use — it costs one decode + encode and
/// returns deterministically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Budget {
    /// Pick a strategy from the calibration table, run it once, ship the
    /// result. No measurement loop.
    #[default]
    OneShot,

    /// Run the chosen strategy, measure achieved zensim-A against source,
    /// and refine via secant step up to `N` total iterations. `N = 1` is
    /// equivalent to [`OneShot`](Self::OneShot) + a measurement.
    MaxIterations(u32),

    /// Same as [`MaxIterations`](Self::MaxIterations) but wall-clock bounded.
    MaxTime(std::time::Duration),
}

/// Options passed to [`recompress`].
///
/// Add fields by bumping the minor (0.x) version — callers using
/// `..Default::default()` continue to compile. The struct intentionally
/// does NOT carry `#[non_exhaustive]` so direct construction works for
/// callers who want to set every field explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecompressOptions {
    /// Target zensim Profile A score in `[0.0, 100.0]`.
    ///
    /// This is interpreted as the *cumulative* score vs the (unknown)
    /// reference, NOT the generation-loss vs the source. The calibration
    /// table maps `(source_q, target_zensim_a)` to a per-strategy quality
    /// dial that hits the cumulative target in expectation.
    pub target_zensim_a: f32,

    /// Compute budget; see [`Budget`].
    pub budget: Budget,

    /// Tolerance below `target_zensim_a` that the router will accept when
    /// no strategy projects strictly at-or-above target. Used in
    /// "best-effort" mode where exact target matching is not required.
    ///
    /// Default: `1.5`. Set to `0.0` for strict-only behavior. Set to
    /// e.g. `5.0` for aggressive shrinking that tolerates more cumulative
    /// loss vs the reference.
    pub tolerance_below_target: f32,
}

impl Default for RecompressOptions {
    fn default() -> Self {
        Self {
            target_zensim_a: 80.0,
            budget: Budget::OneShot,
            tolerance_below_target: 1.5,
        }
    }
}

/// Strategy actually chosen by the router.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StrategyKind {
    /// Coefficient-domain VP8 tighten (no pixel round-trip). Lossy VP8 only.
    CoeffEdit,
    /// Decode + deblock + re-encode at calibrated Q.
    DeblockReencode,
    /// Decode + re-encode at calibrated Q (no deblock).
    Reencode,
    /// Decode + re-encode as VP8L (lossless).
    LosslessReencode,
    /// Container re-mux only: strip metadata, normalize chunks, no
    /// recompression.
    LosslessRemux,
}

/// Why the router chose `LosslessOnly` over a recompression path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LosslessReason {
    /// Every candidate strategy projected output ≥ input size at target.
    NoStrategyShrinksFile,
    /// Every candidate strategy projected zensim-A below target.
    NoStrategyMeetsTarget,
    /// Source was already lossless (VP8L); recompression as lossy would
    /// degrade quality below target without a guaranteed size win.
    SourceWasLossless,
    /// Source is animated; 0.1.x only does `LosslessRemux` for animations.
    SourceIsAnimated,
    /// Source quality is so low that any standard encoder would inflate it
    /// to reach the target.
    SourceQualityTooLow,
}

/// Why the router emitted `NoOp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NoOpReason {
    /// Source's projected zensim-A is already within the "do not improve"
    /// band of the target.
    SourceAlreadyMeetsTarget,
}

/// Result of [`recompress`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RecompressResult {
    /// Recompression happened. `bytes` is the new WebP.
    Recompressed {
        /// Recompressed WebP bytes.
        bytes: Vec<u8>,
        /// Which strategy the router picked.
        strategy: StrategyKind,
        /// Model-projected zensim-A vs reference.
        projected_zensim_a: f32,
        /// Measured zensim-A vs source. `Some(_)` only when budget allows
        /// measurement; `None` for [`Budget::OneShot`].
        measured_zensim_a: Option<f32>,
        /// `output.len() / input.len()`.
        source_to_output_ratio: f32,
        /// `true` if the source's `(encoder, q, content_class)` cell hints
        /// that a JXL transcode would shrink further at the same target.
        /// The caller decides whether to switch formats.
        better_handled_by_jxl: bool,
    },
    /// Recompression would have lost (file grew or undershot target).
    /// `bytes` is a metadata-normalized re-mux of the source — same VP8/VP8L
    /// payload, smaller container.
    LosslessOnly {
        /// Re-muxed WebP bytes.
        bytes: Vec<u8>,
        /// Why we fell back to lossless-only.
        reason: LosslessReason,
        /// Same `better_handled_by_jxl` hint as the `Recompressed` arm.
        better_handled_by_jxl: bool,
    },
    /// Source already meets target; no-op.
    NoOp {
        /// Why no-op.
        reason: NoOpReason,
    },
}

impl RecompressResult {
    /// Return the output bytes regardless of which arm matched, for callers
    /// that just want the optimized WebP.
    pub fn into_bytes(self, source: &[u8]) -> Vec<u8> {
        match self {
            Self::Recompressed { bytes, .. } | Self::LosslessOnly { bytes, .. } => bytes,
            Self::NoOp { .. } => source.to_vec(),
        }
    }

    /// Which strategy was chosen, if any. `None` for `NoOp` (no strategy
    /// ran), `Some(LosslessRemux)` for `LosslessOnly`.
    pub fn strategy(&self) -> Option<StrategyKind> {
        match self {
            Self::Recompressed { strategy, .. } => Some(*strategy),
            Self::LosslessOnly { .. } => Some(StrategyKind::LosslessRemux),
            Self::NoOp { .. } => None,
        }
    }

    /// Hint that the source is a better candidate for JXL transcoding than
    /// for further WebP recompression. Caller decides.
    pub fn better_handled_by_jxl(&self) -> bool {
        match self {
            Self::Recompressed {
                better_handled_by_jxl,
                ..
            }
            | Self::LosslessOnly {
                better_handled_by_jxl,
                ..
            } => *better_handled_by_jxl,
            Self::NoOp { .. } => false,
        }
    }
}

/// Preview of [`recompress`]'s decision without actually running the
/// strategy. Callers wanting to pre-flight a batch (e.g. "would
/// recompressing this image even help? or should I look at JXL?") use
/// [`plan`] to query the router cheaply.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Plan {
    /// The router would recompress with this strategy. The projection is
    /// model-based, not measured — call [`recompress`] with a non-OneShot
    /// budget to get a measurement.
    Recompress {
        /// Which strategy the router picked.
        strategy: StrategyKind,
        /// Model-projected zensim-A vs reference.
        projected_zensim_a: f32,
        /// Model-projected `output_len / input_len`.
        projected_size_ratio: f32,
        /// Hint that JXL transcoding may serve this image better.
        better_handled_by_jxl: bool,
    },
    /// The router would ship a `LosslessOnly` result. No recompression
    /// would happen.
    LosslessOnly {
        /// Why the router declined to recompress.
        reason: LosslessReason,
        /// Hint that JXL transcoding may serve this image better.
        better_handled_by_jxl: bool,
    },
    /// The router would no-op (source already meets target).
    NoOp { reason: NoOpReason },
}

/// Preview the router decision **without** running the chosen strategy.
///
/// Returns the [`Plan`] the router would dispatch — the strategy, projected
/// score and size ratio, and the `better_handled_by_jxl` hint — so callers
/// can do batch / JXL-vs-WebP triage before committing to a full recompress.
///
/// This decodes the source and runs the decode-based quality estimator (one
/// decode + a few probe encodes), because header-only quality detection is
/// unreliable for segmented WebP (see `docs/QUALITY_DETECTION.md`) and a
/// wrong preview would defeat the purpose of triage. It does NOT run the
/// chosen strategy itself, so it is still markedly cheaper than
/// [`recompress`] — especially under a measured [`Budget`].
pub fn plan(webp_bytes: &[u8], opts: &RecompressOptions) -> Result<Plan, Error> {
    if !(0.0..=100.0).contains(&opts.target_zensim_a) || opts.target_zensim_a.is_nan() {
        return Err(Error::TargetOutOfRange(opts.target_zensim_a));
    }
    let mut analysis = crate::source::analyze_source(webp_bytes)?;
    // Refine with the reliable decode-based estimate so the preview matches
    // what `recompress` would actually decide.
    crate::source::refine_from_decode(&mut analysis, webp_bytes);
    let decision = crate::router::decide_strategy(&analysis, opts);
    Ok(match decision.action {
        crate::router::Action::NoOp(reason) => Plan::NoOp { reason },
        crate::router::Action::LosslessOnly(reason) => Plan::LosslessOnly {
            reason,
            better_handled_by_jxl: decision.better_handled_by_jxl,
        },
        crate::router::Action::Recompress { strategy, estimate } => Plan::Recompress {
            strategy,
            projected_zensim_a: estimate.projected_zensim_a,
            projected_size_ratio: estimate.projected_size_ratio,
            better_handled_by_jxl: decision.better_handled_by_jxl,
        },
    })
}

/// Recompress an already-encoded WebP toward a perceptual target.
///
/// See the crate-level documentation and [`DESIGN.md`] for the full
/// contract. The short version:
///
/// 1. Probe the input via [`zenwebp::detect::probe`].
/// 2. Estimate the source's zensim-A from `(encoder_family, source_q,
///    content_class)`.
/// 3. If already at target, return [`RecompressResult::NoOp`].
/// 4. Otherwise consult the calibration table for the optimal strategy and
///    dispatch it; fall back to a lossless re-mux if no strategy shrinks
///    the file at-or-above target.
///
/// `target_zensim_a` is clamped to `[0.0, 100.0]`; values outside this
/// range return [`Error::TargetOutOfRange`].
///
/// [`DESIGN.md`]: https://github.com/imazen/zenwebp-recompress/blob/main/DESIGN.md
pub fn recompress(webp_bytes: &[u8], opts: &RecompressOptions) -> Result<RecompressResult, Error> {
    if !(0.0..=100.0).contains(&opts.target_zensim_a) || opts.target_zensim_a.is_nan() {
        return Err(Error::TargetOutOfRange(opts.target_zensim_a));
    }

    let mut analysis = source::analyze_source(webp_bytes)?;
    // Decode once to refine BOTH content_class and the reliable
    // `estimated_quality` (header quality detection is unreliable for
    // segmented WebP — see docs/QUALITY_DETECTION.md). recompress decodes
    // anyway for any pixel-domain strategy, so this is the right place to
    // pay the cost; plan() stays header-only and conservative.
    source::refine_from_decode(&mut analysis, webp_bytes);
    let decision = router::decide_strategy(&analysis, opts);
    router::dispatch(webp_bytes, &analysis, decision, opts)
}
