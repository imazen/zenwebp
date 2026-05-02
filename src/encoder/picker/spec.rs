//! Cell taxonomy + feature schema for the v0.1 zenwebp picker.
//!
//! Mirrors `zenanalyze/zentrain/examples/zenwebp_picker_config.py`'s
//! v0.2 schema (the one PR #36 unblocked + #52 + the 2026-04-30 ablation
//! pruned). Categorical axes form 6 cells; the picker's output vector
//! has `1 + 3 = 4` blocks of 6 entries each (bytes_log + 3 scalar heads).

#![allow(dead_code)] // some items used only by the bake harness / future wiring.

use core::ops::Range;

/// Number of cells in the v0.1 picker grid.
pub const N_CELLS: usize = 6;

/// One cell. Cells are formed from the `(method, segments)` Cartesian
/// product; each tuple is one categorical bucket the model picks among
/// via argmin over the bytes_log sub-range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellSpec {
    /// VP8 method (RD optimization tier). Production grid: {4, 5, 6}.
    pub method: u8,
    /// Adaptive-quantization segment count. Production grid: {1, 4}.
    pub segments: u8,
}

/// The 6-cell taxonomy. **Order is load-bearing** — the picker's
/// output vector is laid out cell-major:
///   `bytes_log[0..6], sns[0..6], filter_strength[0..6], filter_sharpness[0..6]`.
/// Reordering invalidates any baked model. Bumps to
/// [`SCHEMA_VERSION_TAG`] gate re-bakes; do NOT reorder without
/// bumping the tag and re-baking.
///
/// The order matches what zenpicker's training-side `categorical_key`
/// produces after sorting tuples lexicographically:
/// `(method, segments)` over `[(4,1),(4,4),(5,1),(5,4),(6,1),(6,4)]`.
pub const CELLS: [CellSpec; N_CELLS] = [
    CellSpec {
        method: 4,
        segments: 1,
    },
    CellSpec {
        method: 4,
        segments: 4,
    },
    CellSpec {
        method: 5,
        segments: 1,
    },
    CellSpec {
        method: 5,
        segments: 4,
    },
    CellSpec {
        method: 6,
        segments: 1,
    },
    CellSpec {
        method: 6,
        segments: 4,
    },
];

/// Hybrid-heads output layout. Each block is a sub-range of the
/// model's output vector covering all `N_CELLS` outputs.
pub const RANGE_BYTES_LOG: Range<usize> = 0..N_CELLS;
pub const RANGE_SNS: Range<usize> = N_CELLS..(2 * N_CELLS);
pub const RANGE_FILTER_STRENGTH: Range<usize> = (2 * N_CELLS)..(3 * N_CELLS);
pub const RANGE_FILTER_SHARPNESS: Range<usize> = (3 * N_CELLS)..(4 * N_CELLS);

/// Total output dimension: 6 cells × (1 bytes_log + 3 scalars) = 24.
pub const N_OUTPUTS: usize = 4 * N_CELLS;

/// User-facing constraints: which cells the caller is willing to
/// accept. Default = all allowed. The codec API translates the
/// caller's `LossyConfig` knobs (e.g., `max_method=5`) into a
/// per-cell `bool` mask before invoking the picker.
#[derive(Clone, Copy, Debug, Default)]
pub struct PickerConstraints {
    /// If `Some`, only cells whose `method` is in this list are
    /// allowed. `None` = all of `{4, 5, 6}` allowed.
    pub allowed_methods: Option<&'static [u8]>,
    /// If `Some`, only cells whose `segments` value is in this list
    /// are allowed. `None` = both `{1, 4}` allowed.
    pub allowed_segments: Option<&'static [u8]>,
}

impl PickerConstraints {
    /// Build the per-cell `bool` mask the picker's `argmin_masked_in_range`
    /// consumes.
    pub fn allowed_mask(&self) -> [bool; N_CELLS] {
        core::array::from_fn(|i| {
            let cell = &CELLS[i];
            let m_ok = self
                .allowed_methods
                .is_none_or(|s| s.contains(&cell.method));
            let s_ok = self
                .allowed_segments
                .is_none_or(|s| s.contains(&cell.segments));
            m_ok && s_ok
        })
    }
}

/// zenanalyze feature schema consumed by the v0.1 picker — the
/// 32-feature safe-prune from the 2026-04-30 ablation. Order MUST
/// match the Python codec config's `KEEP_FEATURES` exactly; the
/// `schema_hash` baked into the .bin guards against drift but the
/// runtime vector building still has to feed features in this order.
///
/// Mirror of `zenwebp_picker_config.py` KEEP_FEATURES, post-prune.
pub const FEAT_COLS: &[&str] = &[
    // Top tier (Δ ≥ +0.20pp from the 2026-04-30 ablation)
    "feat_laplacian_variance_p50",
    "feat_laplacian_variance_p75",
    "feat_laplacian_variance",
    "feat_quant_survival_y",
    "feat_cb_sharpness",
    "feat_pixel_count",
    "feat_uniformity",
    "feat_distinct_color_bins",
    "feat_cr_sharpness",
    "feat_edge_density",
    "feat_noise_floor_y_p50",
    "feat_luma_histogram_entropy",
    // Mid tier (Δ +0.10..+0.20pp)
    "feat_natural_likelihood",
    "feat_quant_survival_y_p50",
    "feat_noise_floor_uv_p50",
    "feat_aq_map_mean",
    "feat_cr_horiz_sharpness",
    "feat_min_dim",
    "feat_edge_slope_stdev",
    "feat_laplacian_variance_p90",
    "feat_patch_fraction",
    "feat_max_dim",
    "feat_aspect_min_over_max",
    "feat_aq_map_p75",
    // Low tier (Δ +0.05..+0.10pp)
    "feat_cb_horiz_sharpness",
    "feat_noise_floor_y_p25",
    "feat_noise_floor_uv",
    "feat_chroma_complexity",
    "feat_quant_survival_y_p75",
    "feat_aq_map_std",
    "feat_gradient_fraction",
    "feat_noise_floor_y_p75",
    "feat_screen_content_likelihood",
    "feat_high_freq_energy_ratio",
    "feat_colourfulness",
    "feat_quant_survival_uv",
];

/// Number of raw features this build expects. The engineered input
/// vector adds `4 size_oh + 5 polynomials + N_FEAT_COLS cross-terms
/// + 1 icc_bytes` → `2 * N_FEAT_COLS + 10` MLP inputs.
pub const N_FEAT_COLS: usize = FEAT_COLS.len();

/// Schema version tag — bump when CELLS reorder or FEAT_COLS change.
/// The Python `bake_picker.py` hashes `(FEAT_COLS, extra_axes,
/// SCHEMA_VERSION_TAG)` into a u64 the runtime checks against the
/// `Model::from_bytes_with_schema` call.
pub const SCHEMA_VERSION_TAG: &str = "zenwebp.picker.v0.1";

/// Schema hash captured at bake time — emitted by `bake_picker.py`
/// stderr. Verified at codec startup via `Model::from_bytes_with_schema`.
/// Bumping this requires re-running the trainer + baker (the JSON's
/// fingerprint changes any time we reorder FEAT_COLS or extra_axes).
pub const SCHEMA_HASH: u64 = 0xb2aca28a2d7a34ec;

/// Convenience: cell index → encoder tuning tuple under the v0.2
/// shape. The runtime calls this after picking the cell + reading the
/// scalar heads to assemble the final encoder config.
pub const fn cell_to_method_segments(idx: usize) -> (u8, u8) {
    let cell = CELLS[idx];
    (cell.method, cell.segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn cells_are_unique() {
        for i in 0..N_CELLS {
            for j in (i + 1)..N_CELLS {
                assert_ne!(CELLS[i], CELLS[j], "cell {i} == cell {j}");
            }
        }
    }

    #[test]
    fn cells_match_lex_sort_of_method_segments() {
        let methods: [u8; 3] = [4, 5, 6];
        let segments: [u8; 2] = [1, 4];
        let mut expected = Vec::with_capacity(N_CELLS);
        for &m in &methods {
            for &s in &segments {
                expected.push(CellSpec {
                    method: m,
                    segments: s,
                });
            }
        }
        for (i, cell) in CELLS.iter().enumerate() {
            assert_eq!(*cell, expected[i], "cell {i} mismatch");
        }
    }

    #[test]
    fn output_layout_partitions_correctly() {
        assert_eq!(RANGE_BYTES_LOG.end - RANGE_BYTES_LOG.start, N_CELLS);
        assert_eq!(RANGE_SNS.end - RANGE_SNS.start, N_CELLS);
        assert_eq!(
            RANGE_FILTER_STRENGTH.end - RANGE_FILTER_STRENGTH.start,
            N_CELLS
        );
        assert_eq!(
            RANGE_FILTER_SHARPNESS.end - RANGE_FILTER_SHARPNESS.start,
            N_CELLS
        );
        assert_eq!(RANGE_FILTER_SHARPNESS.end, N_OUTPUTS);
    }

    #[test]
    fn constraints_default_allows_all() {
        let mask = PickerConstraints::default().allowed_mask();
        assert_eq!(mask, [true; N_CELLS]);
    }

    #[test]
    fn constraints_method_only_4() {
        let c = PickerConstraints {
            allowed_methods: Some(&[4]),
            ..Default::default()
        };
        let mask = c.allowed_mask();
        // (4,1) + (4,4) allowed; rest masked
        assert_eq!(mask, [true, true, false, false, false, false]);
    }

    #[test]
    fn constraints_method_4_or_5_segments_1() {
        let c = PickerConstraints {
            allowed_methods: Some(&[4, 5]),
            allowed_segments: Some(&[1]),
        };
        let mask = c.allowed_mask();
        assert_eq!(mask, [true, false, true, false, false, false]);
    }
}
