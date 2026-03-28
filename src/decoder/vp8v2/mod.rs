//! VP8v2 lossy decoder — ground-up redesign targeting libwebp C parity.
//!
//! Key differences from v1 (`super::vp8`):
//! - `DecoderContext` with buffer reuse (no memset per decode)
//! - Streaming row pipeline (no full-frame Y/U/V buffers)
//! - Precomputed filter/dequant tables in `FrameTables`
//! - Flat u8 probability tables (no TreeNode indirection)
//! - Fixed-size arrays for per-MB working storage

mod coefficients;
mod context;
mod header;
mod tables;

pub(crate) use context::DecoderContext;

use crate::common::types::{ChromaMode, IntraMode, LumaMode};

/// Per-macroblock data from coefficient parsing, consumed by the
/// prediction + reconstruction + filter pipeline.
#[derive(Clone, Copy, Default)]
pub(super) struct MbRowEntry {
    /// Luma prediction mode for this macroblock.
    pub luma_mode: LumaMode,
    /// Chroma prediction mode for this macroblock.
    pub chroma_mode: ChromaMode,
    /// I4 sub-block prediction modes (only valid when `luma_mode == LumaMode::B`).
    pub bpred: [IntraMode; 16],
    /// Segment index (0-3).
    pub segmentid: u8,
    /// Whether all coefficients were skipped (zero block).
    pub coeffs_skipped: bool,
    /// Per-block non-zero bitmap. Bit i set = block i has non-zero coefficients.
    /// Blocks 0-15 = Y, 16-19 = U, 20-23 = V.
    pub non_zero_blocks: u32,
    /// True if any block in this MB has non-zero DCT coefficients.
    pub non_zero_dct: bool,
    /// True if any UV sub-block has non-zero AC coefficients.
    /// Used to suppress dithering on blocks with actual chroma detail.
    pub has_nonzero_uv_ac: bool,
}
