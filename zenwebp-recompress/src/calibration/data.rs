//! Empirical calibration constants.
//!
//! **Keyed on effective source quality, NOT the header quantizer.** Header
//! quality detection is unreliable for segmented WebP (see
//! `docs/QUALITY_DETECTION.md`); the router supplies a decode-based
//! estimate (`src/estimate.rs`) as the lookup key.
//!
//! Fit from a clean lossless-only sweep
//! (`benchmarks/clean_sweep_2026-05-28`): 3 photo references decoded from
//! true-lossless WebP, re-encoded at each source quality in the grid, then
//! re-compressed at each target quality. Each cell is the mean over the
//! photo refs of `(output_len / source_len, cumulative_zensim_a vs the
//! lossless original)`. **Preliminary** — 3 photo refs is below the
//! 50-per-class bar; re-fit recipe in `docs/QUALITY_DETECTION.md`.
//!
//! Cumulative is bounded by the source's own quality: re-encoding cannot
//! make an image closer to the original than the source already is.

/// Source-quality anchors (effective libwebp quality) for the table rows.
pub const SOURCE_Q_ANCHORS: &[f32] = &[30.0, 50.0, 75.0, 90.0];

/// Target libwebp quality grid for the table columns.
pub const TARGET_Q_GRID: &[u8] = &[45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95];

/// `REENCODE_RATIO[src_anchor][target_col]` — `output_len / source_len`.
pub const REENCODE_RATIO: [[f32; 11]; 4] = [
    // source q30
    [
        0.981, 0.981, 1.000, 1.017, 1.036, 1.055, 1.071, 1.089, 1.174, 1.297, 1.634,
    ],
    // source q50
    [
        0.910, 0.910, 0.943, 0.969, 0.983, 0.998, 1.016, 1.033, 1.097, 1.200, 1.470,
    ],
    // source q75
    [
        0.806, 0.806, 0.848, 0.886, 0.911, 0.933, 0.957, 0.975, 1.036, 1.110, 1.334,
    ],
    // source q90
    [
        0.622, 0.622, 0.644, 0.665, 0.684, 0.703, 0.719, 0.738, 0.838, 0.943, 1.104,
    ],
];

/// `REENCODE_CUM[src_anchor][target_col]` — cumulative zensim-A vs the
/// original reference after re-encoding the source at the column's target
/// quality.
pub const REENCODE_CUM: [[f32; 11]; 4] = [
    // source q30
    [
        53.7, 53.7, 54.4, 54.3, 55.4, 56.3, 58.5, 59.3, 60.3, 61.3, 62.6,
    ],
    // source q50
    [
        56.1, 56.1, 61.1, 63.5, 64.6, 65.2, 65.2, 65.5, 66.3, 67.0, 67.7,
    ],
    // source q75
    [
        57.2, 57.2, 60.6, 64.7, 65.9, 66.5, 67.4, 70.8, 69.9, 72.1, 74.2,
    ],
    // source q90
    [
        60.2, 60.2, 65.0, 66.3, 67.3, 69.2, 70.2, 73.2, 76.0, 78.9, 80.3,
    ],
];

/// VP8L (lossless re-encode) size ratio for photo content (balloons).
/// Screen / line-art shrinks instead (handled in the projection).
pub const VP8L_PHOTO_RATIO: f32 = 5.0;

/// A re-encode projection: encode the source at `target_q`, projected to
/// land at `cum` cumulative zensim-A with `size_ratio` of the source size.
#[derive(Debug, Clone, Copy)]
pub struct ReencodeChoice {
    pub target_q: u8,
    pub cum: f32,
    pub size_ratio: f32,
}

/// The source's own cumulative zensim-A vs the original, as a function of
/// effective quality. Measured (photo): q50→73, q75→79, q90→86. Monotone
/// linear fit, clamped. Used as the achievable ceiling and the NoOp check.
pub fn source_cum(eff_q: f32) -> f32 {
    (0.32 * eff_q + 57.0).clamp(30.0, 92.0)
}

/// Pick the cheapest re-encode target quality whose projected cumulative
/// meets `target_zensim_a`, for a source of effective quality `eff_q`.
///
/// Interpolates the table rows over `eff_q`, scans target columns, and
/// returns the minimum-`size_ratio` column with `cum >= target_zensim_a`
/// and `size_ratio < 1.0` (must actually shrink). Returns `None` if no
/// column both meets the target and shrinks — the router then falls back
/// to a lossless path.
pub fn best_reencode(eff_q: f32, target_zensim_a: f32) -> Option<ReencodeChoice> {
    let (lo, hi, frac) = bracket_anchor(eff_q);
    let mut best: Option<ReencodeChoice> = None;
    for (col, &tq) in TARGET_Q_GRID.iter().enumerate() {
        let ratio = lerp(REENCODE_RATIO[lo][col], REENCODE_RATIO[hi][col], frac);
        let cum = lerp(REENCODE_CUM[lo][col], REENCODE_CUM[hi][col], frac);
        if cum + 1e-3 >= target_zensim_a && ratio < 1.0 {
            let take = match &best {
                None => true,
                Some(b) => ratio < b.size_ratio,
            };
            if take {
                best = Some(ReencodeChoice {
                    target_q: tq,
                    cum,
                    size_ratio: ratio,
                });
            }
        }
    }
    best
}

/// Interpolated cumulative for source `eff_q` at table column `col`.
pub fn reencode_cum_at(eff_q: f32, col: usize) -> f32 {
    let col = col.min(TARGET_Q_GRID.len() - 1);
    let (lo, hi, frac) = bracket_anchor(eff_q);
    lerp(REENCODE_CUM[lo][col], REENCODE_CUM[hi][col], frac)
}

/// Best achievable cumulative from re-encoding (ignoring the shrink
/// requirement) — used to report the projection even when nothing shrinks.
pub fn max_reencode_cum(eff_q: f32) -> f32 {
    let (lo, hi, frac) = bracket_anchor(eff_q);
    let mut m = 0.0f32;
    for (lo_cum, hi_cum) in REENCODE_CUM[lo].iter().zip(REENCODE_CUM[hi].iter()) {
        m = m.max(lerp(*lo_cum, *hi_cum, frac));
    }
    m
}

fn bracket_anchor(eff_q: f32) -> (usize, usize, f32) {
    let last = SOURCE_Q_ANCHORS.len() - 1;
    if eff_q <= SOURCE_Q_ANCHORS[0] {
        return (0, 0, 0.0);
    }
    if eff_q >= SOURCE_Q_ANCHORS[last] {
        return (last, last, 0.0);
    }
    for i in 0..last {
        let a = SOURCE_Q_ANCHORS[i];
        let b = SOURCE_Q_ANCHORS[i + 1];
        if eff_q >= a && eff_q <= b {
            return (i, i + 1, (eff_q - a) / (b - a).max(f32::EPSILON));
        }
    }
    (last, last, 0.0)
}

#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_cum_is_monotone_and_sane() {
        assert!(source_cum(50.0) < source_cum(75.0));
        assert!(source_cum(75.0) < source_cum(90.0));
        assert!((source_cum(50.0) - 73.0).abs() < 4.0);
        assert!((source_cum(90.0) - 86.0).abs() < 4.0);
    }

    #[test]
    fn high_quality_source_recompresses_for_mid_target() {
        // q90 source targeting cumulative 70: measured target_q ~75 →
        // cum 70.2, ratio 0.72.
        let c = best_reencode(90.0, 70.0).expect("q90 → target 70 should recompress");
        assert!(c.size_ratio < 1.0, "must shrink, got {}", c.size_ratio);
        assert!(c.cum >= 70.0 - 1e-3);
    }

    #[test]
    fn unreachable_target_returns_none() {
        assert!(best_reencode(90.0, 95.0).is_none());
        assert!(best_reencode(50.0, 90.0).is_none());
    }

    #[test]
    fn returned_choice_always_shrinks() {
        for eq in [30.0, 50.0, 75.0, 90.0] {
            for t in [50.0, 55.0, 60.0, 65.0, 70.0, 75.0] {
                if let Some(c) = best_reencode(eq, t) {
                    assert!(c.size_ratio < 1.0, "eq{eq} t{t} ratio {}", c.size_ratio);
                    assert!(c.cum + 1e-3 >= t);
                }
            }
        }
    }
}
