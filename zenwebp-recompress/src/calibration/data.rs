//! Empirical calibration constants, fit from
//! `benchmarks/paired_sweep_2026-05-28.csv` (10 reference WebPs decoded
//! losslessly, 16 source quality levels, 10 zensim-A targets, 4 strategies,
//! 6400 cells total).
//!
//! Each row stores p50 (median) cumulative zensim-A vs reference and p50
//! `output_len / input_len`. Indexed by `(quantizer_index_bin,
//! requested_target_zensim_a)`. The router looks up the per-strategy cell
//! and picks the smallest-`size_ratio` strategy whose projected cumulative
//! meets target.
//!
//! Re-fit recipe (see `docs/CALIBRATION_NOTES.md`):
//!
//! ```text
//! cargo run -p zwr-calibrate --release -- \
//!   --refs <dir of lossless WebP/PNG refs> \
//!   --q-grid 20:95:5 \
//!   --targets 50:95:5 \
//!   --strategies remux,reencode,deblock,vp8l \
//!   --output benchmarks/paired_sweep_<date>.csv
//! ```

/// Single calibration cell.
#[derive(Debug, Clone, Copy)]
pub struct Cell {
    pub p50_cum_zensim_a: f32,
    pub p50_size_ratio: f32,
}

impl Cell {
    pub const fn new(cum: f32, ratio: f32) -> Self {
        Self {
            p50_cum_zensim_a: cum,
            p50_size_ratio: ratio,
        }
    }
}

/// Upper bound of each quantizer-index bin. Sources with `qi <= QI_BINS[i]`
/// fall in bin `i`. Last bin is "very low quality" (qi >= 81).
pub const QI_BINS: &[u8] = &[20, 40, 60, 80, 127];

/// Target zensim-A grid points (must align with sweep).
pub const TARGET_BINS: &[f32] = &[50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0];

/// `REENCODE[qi_bin][target_bin]` — full pixel-domain re-encode.
///
/// `p50_cum_zensim_a` is the median *cumulative* zensim-A vs reference
/// (bounded by source ceiling). `p50_size_ratio` is `output_len /
/// input_len`. See `REENCODE_GEN_LOSS` for the per-encoder predictability
/// signal used by the bound model.
pub const REENCODE: [[Cell; 10]; 5] = [
    // qi 0-20 (very high quality source ~q90+)
    [
        Cell::new(47.0, 0.948), Cell::new(49.5, 0.990), Cell::new(50.8, 1.018),
        Cell::new(51.6, 1.042), Cell::new(52.8, 1.075), Cell::new(53.7, 1.112),
        Cell::new(55.1, 1.149), Cell::new(57.7, 1.228), Cell::new(59.5, 1.375),
        Cell::new(62.2, 1.794),
    ],
    // qi 21-40 (high quality source ~q70-85)
    [
        Cell::new(47.2, 0.882), Cell::new(50.4, 0.929), Cell::new(51.9, 0.962),
        Cell::new(53.2, 0.993), Cell::new(54.5, 1.025), Cell::new(55.9, 1.060),
        Cell::new(57.2, 1.098), Cell::new(60.3, 1.235), Cell::new(62.2, 1.431),
        Cell::new(63.4, 1.977),
    ],
    // qi 41-60 (medium-high source ~q50-70) — SWEET SPOT for recompression
    [
        Cell::new(49.2, 0.638), Cell::new(54.1, 0.686), Cell::new(57.1, 0.725),
        Cell::new(59.9, 0.764), Cell::new(61.9, 0.796), Cell::new(64.2, 0.829),
        Cell::new(65.9, 0.859), Cell::new(69.6, 0.968), Cell::new(72.4, 1.094),
        Cell::new(74.9, 1.399),
    ],
    // qi 61-80 (medium source ~q35-50)
    [
        Cell::new(44.9, 0.827), Cell::new(48.9, 0.892), Cell::new(52.0, 0.936),
        Cell::new(54.3, 0.971), Cell::new(55.5, 1.000), Cell::new(56.8, 1.030),
        Cell::new(57.9, 1.057), Cell::new(59.7, 1.180), Cell::new(62.2, 1.325),
        Cell::new(64.3, 1.774),
    ],
    // qi 81-127 (low source ~q<35) — recompression rarely wins
    [
        Cell::new(44.0, 0.858), Cell::new(48.2, 0.917), Cell::new(51.6, 0.960),
        Cell::new(53.3, 0.991), Cell::new(54.3, 1.020), Cell::new(55.1, 1.056),
        Cell::new(55.6, 1.092), Cell::new(58.7, 1.199), Cell::new(61.3, 1.377),
        Cell::new(63.1, 1.868),
    ],
];

/// `REENCODE_GEN_LOSS[qi_bin][target_bin]` — predicted generation-loss
/// zensim-A (recompressed vs source). This is the encoder-controllable
/// upper bound on cumulative — actual cumulative = `min(gen_loss,
/// source_cum)`.
///
/// The bound model is more accurate per-image than the raw `REENCODE`
/// cumulative averages: gen_loss is encoder-deterministic given target_q,
/// while cumulative averages mix source-content noise.
pub const REENCODE_GEN_LOSS: [[f32; 10]; 5] = [
    // qi 0-20
    [71.7, 75.5, 78.0, 78.0, 78.9, 80.2, 81.6, 85.5, 87.8, 90.5],
    // qi 21-40
    [68.8, 72.8, 74.7, 76.4, 77.9, 79.7, 80.9, 84.4, 87.1, 90.1],
    // qi 41-60
    [60.5, 65.6, 68.8, 72.3, 74.5, 77.3, 79.6, 83.7, 87.1, 90.3],
    // qi 61-80
    [63.6, 68.7, 72.3, 75.4, 77.6, 79.9, 81.4, 83.8, 87.0, 90.1],
    // qi 81-127
    [63.2, 69.1, 73.0, 75.5, 77.1, 78.3, 79.5, 83.5, 86.7, 90.2],
];

/// `DEBLOCK_REENCODE` projection. Until the deblock filter ships in a
/// rerun of the sweep, this mirrors REENCODE — the filter is gradient-
/// gated so its effect on output bpp is small; we update these rows the
/// next time the sweep runs against a deblock-bearing build.
pub const DEBLOCK_REENCODE: [[Cell; 10]; 5] = REENCODE;

/// `LOSSLESS_REMUX[qi_bin]` — pass through (size ratio 1.0) with the
/// source's cumulative zensim as the achievement. Same for all targets;
/// stored once per qi bin.
pub const LOSSLESS_REMUX_CUM_PER_QI_BIN: [f32; 5] = [64.4, 65.6, 78.4, 67.2, 65.7];

/// `LOSSLESS_REENCODE[qi_bin]` — full VP8L re-encode of the decoded
/// RGBA. Always preserves the source's cumulative zensim (no further
/// loss) but at a large size cost for photo content.
pub const LOSSLESS_REENCODE_RATIO_PER_QI_BIN: [f32; 5] = [5.089, 7.066, 4.551, 4.0, 4.0];

/// Quantizer-bin index for a given quantizer index (`qi`).
pub fn qi_to_bin(qi: u8) -> usize {
    for (i, &upper) in QI_BINS.iter().enumerate() {
        if qi <= upper {
            return i;
        }
    }
    QI_BINS.len() - 1
}

/// Target-bin index (linear in 5-point steps from 50 to 95).
/// Clamps to `[0, 9]`. Use bilinear interpolation in callers if needed.
#[allow(dead_code)] // Available for callers that prefer bin lookup over interpolation.
pub fn target_bin(target_zensim_a: f32) -> usize {
    let t = target_zensim_a.clamp(50.0, 95.0);
    let idx = ((t - 50.0) / 5.0).round() as usize;
    idx.min(TARGET_BINS.len() - 1)
}

/// QI-bin "centers" (median qi within each bin), used for interpolation.
/// Derived from sweep data: roughly 10, 30, 50, 70, 100 for the 5 bins.
const QI_BIN_CENTERS: [f32; 5] = [10.0, 30.0, 50.0, 70.0, 100.0];

/// Bilinearly interpolate the cell at exact `(qi, target_zensim_a)` from
/// the table. Falls back to bin lookup at the edges.
pub fn interpolated_reencode(qi: u8, target_zensim_a: f32, table: &[[Cell; 10]; 5]) -> Cell {
    // Find the two qi bins bracketing `qi`.
    let q = qi as f32;
    let (lo_i, hi_i, q_frac) = bracket_qi(q);
    let (lo_t, hi_t, t_frac) = bracket_target(target_zensim_a);

    let c00 = table[lo_i][lo_t];
    let c01 = table[lo_i][hi_t];
    let c10 = table[hi_i][lo_t];
    let c11 = table[hi_i][hi_t];

    let cum_lo = lerp(c00.p50_cum_zensim_a, c01.p50_cum_zensim_a, t_frac);
    let cum_hi = lerp(c10.p50_cum_zensim_a, c11.p50_cum_zensim_a, t_frac);
    let ratio_lo = lerp(c00.p50_size_ratio, c01.p50_size_ratio, t_frac);
    let ratio_hi = lerp(c10.p50_size_ratio, c11.p50_size_ratio, t_frac);

    Cell::new(lerp(cum_lo, cum_hi, q_frac), lerp(ratio_lo, ratio_hi, q_frac))
}

fn bracket_qi(q: f32) -> (usize, usize, f32) {
    if q <= QI_BIN_CENTERS[0] {
        return (0, 0, 0.0);
    }
    let last = QI_BIN_CENTERS.len() - 1;
    if q >= QI_BIN_CENTERS[last] {
        return (last, last, 0.0);
    }
    for i in 0..last {
        let lo = QI_BIN_CENTERS[i];
        let hi = QI_BIN_CENTERS[i + 1];
        if q >= lo && q <= hi {
            return (i, i + 1, (q - lo) / (hi - lo).max(f32::EPSILON));
        }
    }
    (last, last, 0.0)
}

fn bracket_target(t: f32) -> (usize, usize, f32) {
    let last = TARGET_BINS.len() - 1;
    if t <= TARGET_BINS[0] {
        return (0, 0, 0.0);
    }
    if t >= TARGET_BINS[last] {
        return (last, last, 0.0);
    }
    for i in 0..last {
        let lo = TARGET_BINS[i];
        let hi = TARGET_BINS[i + 1];
        if t >= lo && t <= hi {
            return (i, i + 1, (t - lo) / (hi - lo).max(f32::EPSILON));
        }
    }
    (last, last, 0.0)
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Bilinearly-interpolated generation-loss prediction at exact
/// `(qi, target_zensim_a)`. Lower bound on `cumulative_zensim_a` since
/// cumulative is also bounded by `source_cum`.
pub fn interpolated_gen_loss(qi: u8, target_zensim_a: f32) -> f32 {
    let q = qi as f32;
    let (lo_i, hi_i, q_frac) = bracket_qi(q);
    let (lo_t, hi_t, t_frac) = bracket_target(target_zensim_a);

    let g_lo = lerp(
        REENCODE_GEN_LOSS[lo_i][lo_t],
        REENCODE_GEN_LOSS[lo_i][hi_t],
        t_frac,
    );
    let g_hi = lerp(
        REENCODE_GEN_LOSS[hi_i][lo_t],
        REENCODE_GEN_LOSS[hi_i][hi_t],
        t_frac,
    );
    lerp(g_lo, g_hi, q_frac)
}

/// Source's own cumulative zensim-A bound for a given qi (linear
/// interpolation between `LOSSLESS_REMUX_CUM_PER_QI_BIN` centers).
pub fn source_cum_for_qi(qi: u8) -> f32 {
    let (lo_i, hi_i, q_frac) = bracket_qi(qi as f32);
    lerp(
        LOSSLESS_REMUX_CUM_PER_QI_BIN[lo_i],
        LOSSLESS_REMUX_CUM_PER_QI_BIN[hi_i],
        q_frac,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qi_binning_is_monotone() {
        assert_eq!(qi_to_bin(0), 0);
        assert_eq!(qi_to_bin(20), 0);
        assert_eq!(qi_to_bin(21), 1);
        assert_eq!(qi_to_bin(60), 2);
        assert_eq!(qi_to_bin(80), 3);
        assert_eq!(qi_to_bin(127), 4);
        assert_eq!(qi_to_bin(255), 4);
    }

    #[test]
    fn sweet_spot_is_qi_41_60() {
        let bin = qi_to_bin(50);
        let cell = REENCODE[bin][0];
        // At qi 50, target 50, expect significant shrink (ratio < 0.7) and
        // cumulative very close to the target (within ±1 zensim point).
        assert!(
            cell.p50_size_ratio < 0.7,
            "qi 50 target 50 should shrink, got {}",
            cell.p50_size_ratio
        );
        assert!(
            (cell.p50_cum_zensim_a - 50.0).abs() < 1.5,
            "qi 50 target 50 should hit target, got {}",
            cell.p50_cum_zensim_a
        );
    }

    #[test]
    fn lossless_remux_passes_through() {
        // Source cumulative depends on content; no target dependency.
        for cum in LOSSLESS_REMUX_CUM_PER_QI_BIN {
            assert!(cum > 0.0 && cum <= 100.0);
        }
    }
}
