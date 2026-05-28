//! Calibration table loader.
//!
//! Empirical constants fit from
//! `benchmarks/paired_sweep_2026-05-28.csv` (10 lossless WebP refs × 16
//! synthetic source qualities × 10 target zensim-A levels × 4 strategies =
//! 6,400 cells). The fit is keyed on **VP8 quantizer index** (not the
//! derived `source_q` estimate, which is unreliable for non-libwebp
//! encoders) and emits the median cumulative zensim-A vs reference and
//! median `output_len / input_len` per cell.
//!
//! Schema (3-axis grid):
//!
//! - `qi_bin` ∈ `{0..20, 21..40, 41..60, 61..80, 81..127}` (5 bins)
//! - `strategy` ∈ `{Reencode, DeblockReencode, LosslessReencode, LosslessRemux}`
//! - `target_zensim_a` ∈ `{50, 55, …, 95}` (10 bins; outside this range we
//!   clamp)
//!
//! See `docs/CALIBRATION_NOTES.md` for the running log of fit decisions.

pub mod data;

use crate::api::StrategyKind;
use crate::source::{ContentClass, SourceAnalysis, SourceKind};

/// Identifier for the calibration table version. Increments whenever the
/// underlying corpus or fit changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Versioning hook for future remote / on-disk tables.
pub struct TableId(pub u32);

/// Public entry point for table queries.
#[derive(Debug)]
pub struct CalibrationLookup;

impl CalibrationLookup {
    /// Load the embedded calibration table. Currently returns a hand-fit
    /// model; once `zwr-calibrate` is run, this loads from `data.parquet`
    /// via `include_bytes!`.
    pub fn load() -> Self {
        Self
    }

    /// Project `(zensim_a_vs_reference, output_size_ratio)` for each
    /// candidate strategy at this `(source, target)` cell.
    pub fn project_all(&self, analysis: &SourceAnalysis, target_zensim_a: f32) -> AllEstimates {
        let strategies = [
            StrategyKind::CoeffEdit,
            StrategyKind::DeblockReencode,
            StrategyKind::Reencode,
            StrategyKind::LosslessReencode,
            StrategyKind::LosslessRemux,
        ];
        let mut out = AllEstimates::default();
        for s in strategies {
            out.push(s, self.project(analysis, target_zensim_a, s));
        }
        out
    }

    /// Project a single strategy. See [`CellEstimate`].
    ///
    /// Keyed on `analysis.estimated_quality` — the decode-based effective
    /// quality (header detection is unreliable; see
    /// `docs/QUALITY_DETECTION.md`).
    pub fn project(
        &self,
        analysis: &SourceAnalysis,
        target_zensim_a: f32,
        strategy: StrategyKind,
    ) -> CellEstimate {
        let is_lossless = matches!(analysis.kind, SourceKind::LosslessVp8L);
        let eff_q = analysis.estimated_quality;

        // Source's own cumulative zensim-A vs the original. Lossless source
        // → the decoded pixels ARE the reference, so ~100.
        let source_cum = if is_lossless {
            100.0
        } else {
            data::source_cum(eff_q)
        };

        match strategy {
            StrategyKind::LosslessRemux => CellEstimate {
                projected_zensim_a: source_cum,
                projected_size_ratio: 1.0,
                ci_low_zensim_a: source_cum - 1.0,
                ci_high_zensim_a: source_cum,
                chosen_libwebp_q: None,
            },
            StrategyKind::LosslessReencode => {
                // VP8L preserves the source pixels exactly; cumulative ==
                // source_cum. A lossless source re-encoded as VP8L is
                // near-identity (~1.0, measured-corrected by the size
                // guard); lossy photo balloons; screen/line-art shrinks.
                let ratio = if is_lossless {
                    1.0
                } else {
                    match analysis.content_class {
                        ContentClass::Screen | ContentClass::LineArt => 0.55,
                        ContentClass::Photo | ContentClass::Mixed => data::VP8L_PHOTO_RATIO,
                    }
                };
                CellEstimate {
                    projected_zensim_a: source_cum,
                    projected_size_ratio: ratio,
                    ci_low_zensim_a: source_cum - 0.5,
                    ci_high_zensim_a: source_cum,
                    chosen_libwebp_q: None,
                }
            }
            StrategyKind::Reencode => match data::best_reencode(eff_q, target_zensim_a) {
                Some(c) => CellEstimate {
                    projected_zensim_a: c.cum,
                    projected_size_ratio: c.size_ratio,
                    ci_low_zensim_a: c.cum - 4.0,
                    ci_high_zensim_a: c.cum + 2.0,
                    chosen_libwebp_q: Some(c.target_q),
                },
                // No shrinking re-encode meets the target. Report the best
                // achievable cumulative at ratio ≥ 1 so the router rejects
                // it (and the size guard would too).
                None => CellEstimate {
                    projected_zensim_a: data::max_reencode_cum(eff_q).min(source_cum),
                    projected_size_ratio: 1.5,
                    ci_low_zensim_a: 0.0,
                    ci_high_zensim_a: source_cum,
                    chosen_libwebp_q: None,
                },
            },
            StrategyKind::DeblockReencode => {
                // Measured net-negative (docs/deblock_experiment); the
                // router de-selects it. Project worse than Reencode so it
                // never wins even if the filter is re-enabled.
                let base = data::best_reencode(eff_q, target_zensim_a);
                match base {
                    Some(c) => CellEstimate {
                        projected_zensim_a: c.cum - 2.75,
                        projected_size_ratio: c.size_ratio * 1.039,
                        ci_low_zensim_a: c.cum - 6.0,
                        ci_high_zensim_a: c.cum,
                        chosen_libwebp_q: Some(c.target_q),
                    },
                    None => CellEstimate {
                        projected_zensim_a: 0.0,
                        projected_size_ratio: 1.6,
                        ci_low_zensim_a: 0.0,
                        ci_high_zensim_a: 0.0,
                        chosen_libwebp_q: None,
                    },
                }
            }
            StrategyKind::CoeffEdit => {
                // Not yet implemented; project so the router never picks it.
                CellEstimate {
                    projected_zensim_a: if is_lossless { 0.0 } else { source_cum - 1.0 },
                    projected_size_ratio: 1.5,
                    ci_low_zensim_a: 0.0,
                    ci_high_zensim_a: source_cum,
                    chosen_libwebp_q: None,
                }
            }
        }
    }
}

/// Single-cell estimate.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // CI bounds exposed for expert callers / future risk-bound mode.
pub struct CellEstimate {
    /// Projected zensim-A vs reference at this strategy + target.
    pub projected_zensim_a: f32,
    /// Projected `output_size / input_size`.
    pub projected_size_ratio: f32,
    /// 2.5th percentile bound on `projected_zensim_a`.
    pub ci_low_zensim_a: f32,
    /// 97.5th percentile bound on `projected_zensim_a`.
    pub ci_high_zensim_a: f32,
    /// For re-encode strategies: the libwebp quality the projection chose
    /// to hit the target at minimum size. The dispatcher encodes at this
    /// `q` rather than re-deriving one from the target. `None` for
    /// strategies without a quality dial (remux, vp8l).
    pub chosen_libwebp_q: Option<u8>,
}

/// Bag of per-strategy estimates the router picks from.
#[derive(Debug, Default, Clone)]
pub struct AllEstimates {
    pub coeff_edit: Option<CellEstimate>,
    pub deblock: Option<CellEstimate>,
    pub reencode: Option<CellEstimate>,
    pub lossless_reencode: Option<CellEstimate>,
    pub lossless_remux: Option<CellEstimate>,
}

impl AllEstimates {
    fn push(&mut self, kind: StrategyKind, est: CellEstimate) {
        match kind {
            StrategyKind::CoeffEdit => self.coeff_edit = Some(est),
            StrategyKind::DeblockReencode => self.deblock = Some(est),
            StrategyKind::Reencode => self.reencode = Some(est),
            StrategyKind::LosslessReencode => self.lossless_reencode = Some(est),
            StrategyKind::LosslessRemux => self.lossless_remux = Some(est),
        }
    }

    /// Iterate `(strategy, estimate)` pairs. Iteration order is the
    /// router's tie-break preference (cheaper strategies first):
    /// `Reencode → DeblockReencode → LosslessReencode → CoeffEdit →
    /// LosslessRemux`. The router uses `min_by` on size_ratio, so when
    /// two strategies project the same ratio, the one earlier in this
    /// iteration wins.
    pub fn iter(&self) -> impl Iterator<Item = (StrategyKind, CellEstimate)> + '_ {
        [
            (StrategyKind::Reencode, self.reencode),
            (StrategyKind::DeblockReencode, self.deblock),
            (StrategyKind::LosslessReencode, self.lossless_reencode),
            (StrategyKind::CoeffEdit, self.coeff_edit),
            (StrategyKind::LosslessRemux, self.lossless_remux),
        ]
        .into_iter()
        .filter_map(|(k, e)| e.map(|e| (k, e)))
    }
}

/// Choice returned by the router.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // Returned via `expert` API once external callers wire it.
pub struct StrategyChoice {
    pub strategy: StrategyKind,
    pub estimate: CellEstimate,
}

// `EncoderClass` re-export removed in 0.1 cleanup. Callers use
// `crate::source::EncoderFamily` directly via the `expert` module.
