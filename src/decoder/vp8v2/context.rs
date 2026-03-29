//! DecoderContext: reusable, streaming-oriented VP8 decode state.
//!
//! The context owns all mutable buffers that persist across frames.
//! `ensure_capacity` resizes only when dimensions change, avoiding
//! heap churn and memset overhead for same-size re-decodes.

use alloc::vec::Vec;

use super::MbRowEntry;
use super::tables::FrameTables;
use crate::common::prediction::{CHROMA_BLOCK_SIZE, LUMA_BLOCK_SIZE, MB_COEFF_SIZE};
use crate::common::types::MAX_SEGMENTS;
use crate::decoder::bit_reader::{VP8HeaderBitReader, VP8Partitions};
use crate::decoder::dither::VP8Random;
use crate::decoder::loop_filter::MbFilterParams;

/// Maximum stride supported for bounds-check-free loop filtering.
/// WebP max dimension is 16383, rounded up to MB boundary = 16384.
const MAX_FILTER_STRIDE: usize = 16384;

/// Y-channel padding for bounds-check-free loop filtering.
const FILTER_PADDING_Y: usize = 15 * MAX_FILTER_STRIDE + 16;

/// UV-channel padding for bounds-check-free loop filtering.
const FILTER_PADDING_UV: usize = 7 * MAX_FILTER_STRIDE + 16;

/// Info from a previously decoded macroblock needed for future predictions.
/// For top MBs this stores the bottom edge; for the left MB the right edge.
#[derive(Default, Clone, Copy)]
pub(super) struct PreviousMacroBlock {
    pub bpred: [crate::common::types::IntraMode; 4],
    /// Coefficient complexity context: [y2, y, y, y, y, u, u, v, v]
    pub complexity: [u8; 9],
}

/// Reusable VP8 decoder context with streaming row cache.
///
/// All large buffers are allocated on first use and reused across
/// frames via `ensure_capacity`. Fixed-size arrays for per-MB working
/// storage live on the stack (inside this struct).
pub struct DecoderContext {
    // ---- Row cache (streaming: no full-frame Y/U/V) ----
    pub(super) cache_y: Vec<u8>,
    pub(super) cache_u: Vec<u8>,
    pub(super) cache_v: Vec<u8>,
    pub(super) cache_y_stride: usize,
    pub(super) cache_uv_stride: usize,

    // ---- Prediction workspaces (fixed size, reused per MB) ----
    pub(super) luma_ws: [u8; LUMA_BLOCK_SIZE],
    pub(super) chroma_u_ws: [u8; CHROMA_BLOCK_SIZE],
    pub(super) chroma_v_ws: [u8; CHROMA_BLOCK_SIZE],

    // ---- Coefficient storage (cleared per MB) ----
    pub(super) coeff_blocks: [i32; MB_COEFF_SIZE],

    // ---- Per-row buffers (reused, resized by ensure_capacity) ----
    pub(super) mb_filter_params: Vec<MbFilterParams>,
    pub(super) mb_dither_buf: Vec<i32>,
    pub(super) mb_row_data: Vec<MbRowEntry>,

    // ---- Border state for intra prediction ----
    pub(super) top_border_y: Vec<u8>,
    pub(super) left_border_y: [u8; 17],
    pub(super) top_border_u: Vec<u8>,
    pub(super) left_border_u: [u8; 9],
    pub(super) top_border_v: Vec<u8>,
    pub(super) left_border_v: [u8; 9],

    // ---- Top MB context (one per column) ----
    pub(super) top: Vec<PreviousMacroBlock>,
    pub(super) left: PreviousMacroBlock,

    // ---- Precomputed tables (rebuilt per frame) ----
    pub(super) tables: FrameTables,

    // ---- Header bit reader (partition 0, used for per-MB mode parsing) ----
    pub(super) header_reader: VP8HeaderBitReader,

    // ---- Partitions ----
    pub(super) partitions: VP8Partitions,

    // ---- Output Y/U/V frame buffers (allocated per decode_to_frame only) ----
    pub(super) ybuf: Vec<u8>,
    pub(super) ubuf: Vec<u8>,
    pub(super) vbuf: Vec<u8>,

    // ---- Streaming RGB: boundary UV rows for fancy upsampling ----
    // Saves the last visible UV row from the previous MB row's cache,
    // needed by the first even Y row of the next MB row for the "far"
    // chroma reference. One chroma row per channel (~mbwidth*8 bytes each).
    pub(super) prev_last_u_row: Vec<u8>,
    pub(super) prev_last_v_row: Vec<u8>,

    // ---- Chroma dithering state ----
    /// User-requested dithering strength (0-100, 0=off). Default: 50.
    pub(super) dither_strength: u8,
    /// Whether dithering is active for the current frame (computed after header parsing).
    pub(super) dither_enabled: bool,
    /// PRNG for dithering noise generation.
    pub(super) dither_rg: VP8Random,
    /// Per-segment dither amplitudes (computed from UV AC quantizer indices and strength).
    pub(super) dither_amp: [i32; MAX_SEGMENTS],

    // ---- Reuse tracking ----
    pub(super) last_mbwidth: u16,
    pub(super) last_mbheight: u16,
}

impl DecoderContext {
    /// Create a new `DecoderContext` with empty buffers.
    /// Call `ensure_capacity` after parsing the frame header to size the
    /// buffers for the current frame dimensions.
    pub fn new() -> Self {
        Self {
            cache_y: Vec::new(),
            cache_u: Vec::new(),
            cache_v: Vec::new(),
            cache_y_stride: 0,
            cache_uv_stride: 0,

            luma_ws: [0u8; LUMA_BLOCK_SIZE],
            chroma_u_ws: [0u8; CHROMA_BLOCK_SIZE],
            chroma_v_ws: [0u8; CHROMA_BLOCK_SIZE],

            coeff_blocks: [0i32; MB_COEFF_SIZE],

            mb_filter_params: Vec::new(),
            mb_dither_buf: Vec::new(),
            mb_row_data: Vec::new(),

            top_border_y: Vec::new(),
            left_border_y: [129u8; 17],
            top_border_u: Vec::new(),
            left_border_u: [129u8; 9],
            top_border_v: Vec::new(),
            left_border_v: [129u8; 9],

            top: Vec::new(),
            left: PreviousMacroBlock::default(),

            tables: FrameTables::new(),

            header_reader: VP8HeaderBitReader::new(),

            partitions: VP8Partitions::new(),

            ybuf: Vec::new(),
            ubuf: Vec::new(),
            vbuf: Vec::new(),

            prev_last_u_row: Vec::new(),
            prev_last_v_row: Vec::new(),

            dither_strength: 0,
            dither_enabled: false,
            dither_rg: VP8Random::new(),
            dither_amp: [0; MAX_SEGMENTS],

            last_mbwidth: 0,
            last_mbheight: 0,
        }
    }

    /// Pixel width after header parse.
    #[allow(dead_code)]
    pub fn width(&self) -> u16 {
        self.tables.width
    }

    /// Pixel height after header parse.
    #[allow(dead_code)]
    pub fn height(&self) -> u16 {
        self.tables.height
    }

    /// Macroblock width (columns) after header parse.
    #[allow(dead_code)]
    pub fn mbwidth(&self) -> u16 {
        self.tables.mbwidth
    }

    /// Macroblock height (rows) after header parse.
    #[allow(dead_code)]
    pub fn mbheight(&self) -> u16 {
        self.tables.mbheight
    }

    /// Extra Y rows in the cache for filter context (8/2/0).
    #[allow(dead_code)]
    pub fn extra_y_rows(&self) -> usize {
        self.tables.extra_y_rows
    }

    /// Set chroma dithering strength (0=off, 100=max). Default: 50.
    ///
    /// Adds random noise to U/V chroma planes after loop filtering to hide
    /// banding artifacts from coarse chroma quantization at low quality settings.
    #[allow(dead_code)]
    #[must_use]
    pub fn with_dithering_strength(mut self, strength: u8) -> Self {
        self.dither_strength = strength;
        self
    }

    /// Set chroma dithering strength on an existing context (non-consuming).
    pub fn set_dithering_strength(&mut self, strength: u8) {
        self.dither_strength = strength;
    }

    /// Resize buffers for the given macroblock dimensions and filter context.
    ///
    /// Only reallocates when capacity is insufficient. When the dimensions
    /// match the previous frame, this is essentially free (just resets lengths
    /// on Vecs that already have the right capacity).
    ///
    /// `extra_y_rows`: number of extra rows in the Y cache for filter context
    /// (8 for normal filter, 2 for simple, 0 for no filter).
    ///
    /// Returns `Err` if buffer sizes overflow (malicious dimensions).
    pub(super) fn ensure_capacity(
        &mut self,
        mbwidth: u16,
        mbheight: u16,
        extra_y_rows: usize,
    ) -> Result<(), crate::decoder::api::DecodeError> {
        use crate::decoder::api::DecodeError;
        let mbw = usize::from(mbwidth);
        let _mbh = usize::from(mbheight);

        // Cache strides — checked to reject overflow from malicious dimensions
        self.cache_y_stride = mbw.checked_mul(16).ok_or(DecodeError::ImageTooLarge)?;
        self.cache_uv_stride = mbw.checked_mul(8).ok_or(DecodeError::ImageTooLarge)?;

        let extra_uv_rows = extra_y_rows / 2;
        let cache_y_rows = extra_y_rows + 16;
        let cache_uv_rows = extra_uv_rows + 8;

        let cache_y_size = cache_y_rows
            .checked_mul(self.cache_y_stride)
            .and_then(|n| n.checked_add(FILTER_PADDING_Y))
            .ok_or(DecodeError::ImageTooLarge)?;
        let cache_uv_size = cache_uv_rows
            .checked_mul(self.cache_uv_stride)
            .and_then(|n| n.checked_add(FILTER_PADDING_UV))
            .ok_or(DecodeError::ImageTooLarge)?;

        // Cache buffers: resize only grows (no fill needed — all data is
        // overwritten during prediction before it is read). Only the extra
        // rows region needs initialization for the first MB row's filter.
        self.cache_y.resize(cache_y_size, 0);
        self.cache_u.resize(cache_uv_size, 0);
        self.cache_v.resize(cache_uv_size, 0);

        // Initialize extra rows region (filter context for first MB row).
        // Only fills the small extra region, not the entire cache.
        let extra_y_bytes = extra_y_rows * self.cache_y_stride;
        self.cache_y[..extra_y_bytes].fill(128);
        let extra_uv_bytes = extra_uv_rows * self.cache_uv_stride;
        self.cache_u[..extra_uv_bytes].fill(128);
        self.cache_v[..extra_uv_bytes].fill(128);

        // Per-row buffers
        self.mb_filter_params.resize(
            mbw,
            MbFilterParams {
                filter_level: 0,
                interior_limit: 0,
                hev_threshold: 0,
                mbedge_limit: 0,
                sub_bedge_limit: 0,
                do_subblock_filtering: false,
            },
        );
        self.mb_dither_buf.resize(mbw, 0);
        self.mb_row_data.resize(mbw, MbRowEntry::default());

        // Border buffers: MUST be reinitialized for every frame.
        // These carry per-row prediction context that is invalid across frames.
        let top_y_size = mbw * 16 + 4 + 16;
        self.top_border_y.resize(top_y_size, 127);
        self.top_border_y.fill(127);
        self.top_border_u.resize(mbw * 8, 127);
        self.top_border_u.fill(127);
        self.top_border_v.resize(mbw * 8, 127);
        self.top_border_v.fill(127);

        // Top MB context: MUST be reinitialized for every frame.
        // Contains complexity context from the previous frame's bottom row.
        self.top.resize(mbw, PreviousMacroBlock::default());
        self.top.fill(PreviousMacroBlock::default());

        // Reset left border to initial state (always, not just on dim change)
        self.left_border_y = [129u8; 17];
        self.left_border_u = [129u8; 9];
        self.left_border_v = [129u8; 9];
        self.left = PreviousMacroBlock::default();

        // Clear coefficient blocks and workspaces
        self.coeff_blocks = [0i32; MB_COEFF_SIZE];

        self.last_mbwidth = mbwidth;
        self.last_mbheight = mbheight;
        Ok(())
    }
}

impl Default for DecoderContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_context_has_empty_buffers() {
        let ctx = DecoderContext::new();
        assert!(ctx.cache_y.is_empty());
        assert!(ctx.cache_u.is_empty());
        assert!(ctx.cache_v.is_empty());
        assert_eq!(ctx.last_mbwidth, 0);
        assert_eq!(ctx.last_mbheight, 0);
    }

    #[test]
    fn test_ensure_capacity_allocates() {
        let mut ctx = DecoderContext::new();
        ctx.ensure_capacity(10, 10, 8).unwrap();

        assert_eq!(ctx.cache_y_stride, 160);
        assert_eq!(ctx.cache_uv_stride, 80);
        assert!(!ctx.cache_y.is_empty());
        assert!(!ctx.cache_u.is_empty());
        assert!(!ctx.cache_v.is_empty());
        assert_eq!(ctx.mb_filter_params.len(), 10);
        assert_eq!(ctx.mb_row_data.len(), 10);
        assert_eq!(ctx.top.len(), 10);
        assert_eq!(ctx.last_mbwidth, 10);
        assert_eq!(ctx.last_mbheight, 10);
    }

    #[test]
    fn test_ensure_capacity_reuses_allocation() {
        let mut ctx = DecoderContext::new();
        ctx.ensure_capacity(10, 10, 8).unwrap();
        let y_ptr = ctx.cache_y.as_ptr();
        let y_cap = ctx.cache_y.capacity();

        // Same dimensions: should reuse allocation
        ctx.ensure_capacity(10, 10, 8).unwrap();
        assert_eq!(ctx.cache_y.as_ptr(), y_ptr);
        assert_eq!(ctx.cache_y.capacity(), y_cap);
    }

    #[test]
    fn test_ensure_capacity_grows() {
        let mut ctx = DecoderContext::new();
        ctx.ensure_capacity(10, 10, 8).unwrap();
        let small_len = ctx.cache_y.len();

        ctx.ensure_capacity(20, 20, 8).unwrap();
        assert!(ctx.cache_y.len() > small_len);
        assert_eq!(ctx.cache_y_stride, 320);
    }

    #[test]
    fn test_ensure_capacity_no_filter() {
        let mut ctx = DecoderContext::new();
        ctx.ensure_capacity(10, 10, 0).unwrap();

        // With extra_y_rows=0: cache is just 16 rows + padding
        let expected_y = 16 * 160 + FILTER_PADDING_Y;
        assert_eq!(ctx.cache_y.len(), expected_y);
    }
}
