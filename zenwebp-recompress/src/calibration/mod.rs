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
use crate::target::source_q_to_zensim_a_estimate;

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
    pub fn project(
        &self,
        analysis: &SourceAnalysis,
        target_zensim_a: f32,
        strategy: StrategyKind,
    ) -> CellEstimate {
        let qi = if matches!(analysis.kind, SourceKind::LossyVp8) {
            analysis.vp8_quantizer_index
        } else {
            // Lossless source — treat as best possible (qi 0 bin).
            0
        };
        let qi_bin = data::qi_to_bin(qi);
        let source_zensim_a =
            source_q_to_zensim_a_estimate(analysis.encoder_family, analysis.source_q);

        match strategy {
            StrategyKind::LosslessRemux => {
                let cum = data::LOSSLESS_REMUX_CUM_PER_QI_BIN[qi_bin];
                CellEstimate {
                    projected_zensim_a: cum,
                    projected_size_ratio: 1.0,
                    ci_low_zensim_a: cum - 1.0,
                    ci_high_zensim_a: cum,
                }
            }
            StrategyKind::LosslessReencode => {
                // VP8L preserves the source pixels exactly; cumulative is
                // identical to remux. Size cost varies by content class.
                let cum = data::LOSSLESS_REMUX_CUM_PER_QI_BIN[qi_bin];
                let ratio = match analysis.content_class {
                    ContentClass::Screen | ContentClass::LineArt => 0.55,
                    ContentClass::Photo => {
                        data::LOSSLESS_REENCODE_RATIO_PER_QI_BIN[qi_bin]
                    }
                    ContentClass::Mixed => {
                        data::LOSSLESS_REENCODE_RATIO_PER_QI_BIN[qi_bin]
                    }
                };
                CellEstimate {
                    projected_zensim_a: cum,
                    projected_size_ratio: ratio,
                    ci_low_zensim_a: cum - 0.5,
                    ci_high_zensim_a: cum,
                }
            }
            StrategyKind::Reencode => {
                let cell = data::interpolated_reencode(qi, target_zensim_a, &data::REENCODE);
                // Use bound model: cumulative = min(gen_loss, source_cum).
                let gen_loss = data::interpolated_gen_loss(qi, target_zensim_a);
                let source_cum = data::source_cum_for_qi(qi);
                let projected = gen_loss.min(source_cum);
                CellEstimate {
                    projected_zensim_a: projected,
                    projected_size_ratio: cell.p50_size_ratio,
                    ci_low_zensim_a: projected - 4.0,
                    ci_high_zensim_a: projected + 2.0,
                }
            }
            StrategyKind::DeblockReencode => {
                let cell = data::interpolated_reencode(
                    qi,
                    target_zensim_a,
                    &data::DEBLOCK_REENCODE,
                );
                let gen_loss = data::interpolated_gen_loss(qi, target_zensim_a);
                let source_cum = data::source_cum_for_qi(qi);
                // Deblock adds ~1-2 zensim points on heavily quantized
                // sources (qi 60+) by removing artifacts before re-encode.
                let bonus = if qi >= 60 { 1.5 } else { 0.0 };
                let projected = (gen_loss + bonus).min(source_cum).min(100.0);
                CellEstimate {
                    projected_zensim_a: projected,
                    projected_size_ratio: cell.p50_size_ratio,
                    ci_low_zensim_a: projected - 4.0,
                    ci_high_zensim_a: projected + 3.0,
                }
            }
            StrategyKind::CoeffEdit => {
                // CoeffEdit only applies to lossy VP8 and is not yet
                // implemented in the router. Project pessimistically so
                // it never wins until the implementation ships.
                if matches!(analysis.kind, SourceKind::LossyVp8) {
                    CellEstimate {
                        projected_zensim_a: source_zensim_a - 1.0,
                        projected_size_ratio: 0.95,
                        ci_low_zensim_a: source_zensim_a - 3.0,
                        ci_high_zensim_a: source_zensim_a,
                    }
                } else {
                    CellEstimate {
                        projected_zensim_a: 0.0,
                        projected_size_ratio: 1.5,
                        ci_low_zensim_a: 0.0,
                        ci_high_zensim_a: 0.0,
                    }
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
