//! ConfigSpec for the zenwebp picker spike.
//!
//! 16-cell categorical grid over `(sns_strength, filter_strength,
//! segments)` with `filter_sharpness=0` and `method=4` fixed. The
//! tuple returned by [`cell_to_tuning`] is the same shape
//! `content_type_to_tuning` produces, so the wire-in is a one-line
//! swap at the existing call sites.

#![allow(dead_code)] // some helpers used only by the sweep harness / future wire-in.

/// Number of cells in the v0.1 picker grid.
pub const N_CELLS: usize = 16;

/// SNS strength values (zenwebp `sns_strength`, 0..=100).
pub const SNS_VALUES: [u8; 4] = [0, 25, 50, 80];
/// Loop-filter strength values (zenwebp `filter_strength`, 0..=100).
pub const FILTER_VALUES: [u8; 2] = [30, 60];
/// Segment counts (zenwebp `num_segments`, 1..=4).
pub const SEGMENT_VALUES: [u8; 2] = [1, 4];
/// Fixed filter sharpness for the v0.1 spike.
pub const FIXED_FILTER_SHARPNESS: u8 = 0;
/// Fixed encoder method for the v0.1 spike.
pub const FIXED_METHOD: u8 = 4;

/// A single cell in the picker grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfigSpec {
    pub sns_strength: u8,
    pub filter_strength: u8,
    pub filter_sharpness: u8,
    pub num_segments: u8,
}

impl ConfigSpec {
    /// Tuple in the same order [`crate::encoder::analysis::content_type_to_tuning`]
    /// returns: `(sns, filter, sharpness, segments)`.
    pub const fn as_tuning(self) -> (u8, u8, u8, u8) {
        (
            self.sns_strength,
            self.filter_strength,
            self.filter_sharpness,
            self.num_segments,
        )
    }

    /// Stable, lossless name for manifests and logs.
    pub fn name(self) -> alloc::string::String {
        alloc::format!(
            "sns{}_f{}_seg{}",
            self.sns_strength, self.filter_strength, self.num_segments
        )
    }
}

/// Compile-time-stable enumeration of the 16 cells.
///
/// Order: `sns` outermost, then `filter`, then `segments`. The picker
/// model's output index maps 1:1 to this array — changing the order
/// invalidates baked models. Bumps to [`SCHEMA_VERSION_TAG`] gate
/// re-bakes; do not reorder without bumping.
pub const CONFIGS: [ConfigSpec; N_CELLS] = {
    let mut out = [ConfigSpec {
        sns_strength: 0,
        filter_strength: 0,
        filter_sharpness: FIXED_FILTER_SHARPNESS,
        num_segments: 1,
    }; N_CELLS];
    let mut i = 0;
    let mut s = 0;
    while s < SNS_VALUES.len() {
        let mut f = 0;
        while f < FILTER_VALUES.len() {
            let mut g = 0;
            while g < SEGMENT_VALUES.len() {
                out[i] = ConfigSpec {
                    sns_strength: SNS_VALUES[s],
                    filter_strength: FILTER_VALUES[f],
                    filter_sharpness: FIXED_FILTER_SHARPNESS,
                    num_segments: SEGMENT_VALUES[g],
                };
                i += 1;
                g += 1;
            }
            f += 1;
        }
        s += 1;
    }
    out
};

/// Convenience: cell index → encoder tuning tuple.
pub const fn cell_to_tuning(idx: usize) -> (u8, u8, u8, u8) {
    CONFIGS[idx].as_tuning()
}

/// zenanalyze feature schema consumed by the picker, in the order
/// the bake script and Rust loader both expect.
///
/// 13 raw zenanalyze signals + `log_pixels` + `target_zensim_norm`
/// + cross terms. Cross-term layout matches zenjpeg's distill script
/// so the same `bake_picker.py` works.
pub const FEAT_COLS: &[&str] = &[
    "feat_screen_content",
    "feat_text_likelihood",
    "feat_natural_likelihood",
    "feat_flat_color_block_ratio",
    "feat_distinct_color_bins",
    "feat_variance",
    "feat_edge_density",
    "feat_uniformity",
    "feat_high_freq_energy_ratio",
    "feat_palette_fits_in_256",
    "feat_indexed_palette_width",
    "feat_line_art_score",
    "feat_skin_tone_fraction",
    "feat_edge_slope_stdev",
];

/// Schema version tag — bump to invalidate previously baked models.
pub const SCHEMA_VERSION_TAG: &str = "zenwebp.picker.v0.1.spike";

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;

    #[test]
    fn cells_are_unique() {
        for i in 0..N_CELLS {
            for j in (i + 1)..N_CELLS {
                assert_ne!(CONFIGS[i], CONFIGS[j], "cell {i} == cell {j}");
            }
        }
    }

    #[test]
    fn cell_count_matches_axes() {
        assert_eq!(
            N_CELLS,
            SNS_VALUES.len() * FILTER_VALUES.len() * SEGMENT_VALUES.len()
        );
    }

    #[test]
    fn tuning_tuple_shape() {
        let (sns, f, sh, seg) = cell_to_tuning(0);
        assert_eq!(sns, 0);
        assert_eq!(f, 30);
        assert_eq!(sh, FIXED_FILTER_SHARPNESS);
        assert_eq!(seg, 1);
    }
}
