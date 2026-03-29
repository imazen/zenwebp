//! Precomputed per-frame tables for the v2 VP8 decoder.
//!
//! After header parsing, all segment-dependent quantization, filter, and
//! probability parameters are flattened into `FrameTables`. This avoids
//! per-macroblock re-derivation during decode.

use alloc::boxed::Box;

use crate::common::types::MAX_SEGMENTS;

/// DC + AC dequantization pair for one coefficient plane within one segment.
#[derive(Clone, Copy, Default)]
pub(super) struct DequantPair {
    pub dc: i16,
    pub ac: i16,
}

/// Precomputed loop-filter parameters for a (segment, is_b_mode) combination.
/// Identical layout to the v1 `PrecomputedFilterParams`.
#[derive(Clone, Copy, Default)]
pub(super) struct PrecomputedFilterParams {
    pub filter_level: u8,
    pub interior_limit: u8,
    pub hev_threshold: u8,
    pub mbedge_limit: u8,
    pub sub_bedge_limit: u8,
}

/// All frame-level precomputed tables, populated once per frame during
/// header parsing.
pub(super) struct FrameTables {
    /// Dequantization values per segment: `[segment][plane]`
    /// where plane 0 = Y, 1 = Y2, 2 = UV.
    pub dequant: [[DequantPair; 3]; MAX_SEGMENTS],

    /// Filter parameters per segment and B-mode flag: `[segment][is_b_mode]`.
    pub filter: [[PrecomputedFilterParams; 2]; MAX_SEGMENTS],

    /// Position-indexed probability table for coefficient decoding.
    /// `[plane][position][context][token_prob_index]`
    /// Eliminates the COEFF_BANDS lookup in the hot path.
    /// Position 16 is a sentinel (copies band 7) for n+1 lookahead.
    pub probs_by_pos: Box<[[[[u8; 11]; 3]; 17]; 4]>,

    /// True = simple filter, false = normal filter.
    pub filter_type: bool,

    /// Frame-level base filter level (0-63).
    pub filter_level: u8,

    /// Sharpness level (0-7).
    pub sharpness_level: u8,

    /// Probability of skip-coefficient flag; `None` if feature not enabled.
    pub prob_skip_false: Option<u8>,

    /// Whether segment-based features are enabled for this frame.
    pub segments_enabled: bool,

    /// Whether the segment map should be updated from the bitstream.
    pub segments_update_map: bool,

    /// Segment tree probabilities for segment ID decoding.
    pub segment_tree_probs: [u8; 3],

    /// Per-segment delta_values flag (absolute vs delta quantizer mode).
    pub segment_delta_values: [bool; MAX_SEGMENTS],

    /// Per-segment quantizer level (raw from bitstream, before delta application).
    pub segment_quantizer_level: [i8; MAX_SEGMENTS],

    /// Per-segment loop filter level (raw from bitstream).
    pub segment_loopfilter_level: [i8; MAX_SEGMENTS],

    /// Loop filter adjustment reference deltas.
    pub ref_delta: [i32; 4],

    /// Loop filter adjustment mode deltas.
    pub mode_delta: [i32; 4],

    /// Whether loop filter adjustments are enabled.
    pub loop_filter_adjustments_enabled: bool,

    /// Number of token partitions (1, 2, 4, or 8).
    pub num_partitions: u8,

    /// Extra cache rows for filtering context.
    /// 8 for normal filter, 2 for simple, 0 for no filter.
    pub extra_y_rows: usize,

    /// Frame version from the bitstream tag.
    pub version: u8,

    /// Whether this frame is intended for display.
    pub for_display: bool,

    /// Pixel type (color space indicator from section 9.2).
    pub pixel_type: u8,

    /// Frame width in pixels.
    pub width: u16,

    /// Frame height in pixels.
    pub height: u16,

    /// Width in macroblocks.
    pub mbwidth: u16,

    /// Height in macroblocks.
    pub mbheight: u16,

    /// UV AC quantizer indices per segment, used for dithering amplitude computation.
    pub uv_quant_indices: [i32; MAX_SEGMENTS],
}

impl FrameTables {
    /// Create a new `FrameTables` with default/zero values.
    /// The `probs_by_pos` table is heap-allocated.
    pub fn new() -> Self {
        Self {
            dequant: [[DequantPair::default(); 3]; MAX_SEGMENTS],
            filter: [[PrecomputedFilterParams::default(); 2]; MAX_SEGMENTS],
            probs_by_pos: Box::new([[[[0u8; 11]; 3]; 17]; 4]),
            filter_type: false,
            filter_level: 0,
            sharpness_level: 0,
            prob_skip_false: None,
            segments_enabled: false,
            segments_update_map: false,
            segment_tree_probs: [255; 3],
            segment_delta_values: [false; MAX_SEGMENTS],
            segment_quantizer_level: [0; MAX_SEGMENTS],
            segment_loopfilter_level: [0; MAX_SEGMENTS],
            ref_delta: [0; 4],
            mode_delta: [0; 4],
            loop_filter_adjustments_enabled: false,
            num_partitions: 1,
            extra_y_rows: 0,
            version: 0,
            for_display: false,
            pixel_type: 0,
            width: 0,
            height: 0,
            mbwidth: 0,
            mbheight: 0,
            uv_quant_indices: [0; MAX_SEGMENTS],
        }
    }
}
