//! Perceptual distortion model for encoder-side optimizations.
//!
//! Provides frequency-weighted distortion computation using CSF (Contrast
//! Sensitivity Function) tables and SATD-based energy preservation (psy-rd).
//!
//! All features are gated by `method` level — methods 0-2 produce bit-identical
//! output to the baseline encoder.

use super::tables::{VP8_WEIGHT_TRELLIS, VP8_WEIGHT_Y};

/// Perceptual encoding configuration for a segment.
///
/// Controls how distortion is computed during mode selection and trellis
/// quantization. When all strengths are 0, behavior is identical to baseline.
#[derive(Clone, Debug)]
pub(crate) struct PsyConfig {
    /// Psy-RD strength (fixed-point, 0 = disabled).
    /// Penalizes energy loss (smoothing) in SATD domain.
    pub(crate) psy_rd_strength: u32,
    /// Psy-trellis strength (fixed-point, 0 = disabled).
    /// Biases trellis to retain perceptually important coefficients.
    pub(crate) psy_trellis_strength: u32,
    /// CSF weights for luma spectral distortion (TDisto).
    pub(crate) luma_csf: [u16; 16],
    /// CSF weights for chroma spectral distortion (TDisto).
    pub(crate) chroma_csf: [u16; 16],
    /// Distortion weights for trellis quantization.
    pub(crate) trellis_weights: [u16; 16],
}

impl Default for PsyConfig {
    fn default() -> Self {
        Self {
            psy_rd_strength: 0,
            psy_trellis_strength: 0,
            luma_csf: VP8_WEIGHT_Y,
            chroma_csf: VP8_WEIGHT_Y, // same as libwebp default
            trellis_weights: VP8_WEIGHT_TRELLIS,
        }
    }
}

impl PsyConfig {
    /// Create a PsyConfig for the given encoding parameters.
    ///
    /// Currently all features are disabled (strengths = 0) and tables match
    /// the libwebp defaults. This is Phase 1: a pure refactor with no
    /// behavioral change at any method level.
    pub(crate) fn new(_method: u8, _quant_index: u8, _sns_strength: u8) -> Self {
        // Phase 1: all disabled, exact same tables as before
        Self::default()
    }
}
