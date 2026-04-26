//! VP8 lossy encoder implementation.
//!
//! This module provides DCT-based lossy compression for WebP images,
//! compatible with the VP8 intra-frame format.
//!
//! ## Module organization
//!
//! - `header`: VP8 bitstream header encoding
//! - `mode_selection`: Intra mode selection (I4, I16, UV) with RD optimization
//! - `prediction`: Block prediction generation
//! - residuals: Token buffer and coefficient encoding
//!
//! ## Encoding pipeline
//!
//! 1. **Color conversion**: RGB → YUV420 (with optional sharp YUV)
//! 2. **Analysis pass**: Compute per-MB complexity (alpha) for segmentation
//! 3. **Segmentation**: K-means clustering assigns MBs to 1-4 quantization segments
//! 4. **Mode selection**: For each MB, choose best prediction mode via RD cost
//! 5. **Transform & quantize**: DCT + quantization + optional trellis optimization
//! 6. **Entropy coding**: Arithmetic coding of residual coefficients
//! 7. **Loop filter**: Deblocking filter parameters computed for decoder
//!
//! ## Quality settings
//!
//! The encoder supports quality 0-100 (like JPEG), with method levels 0-6
//! controlling the speed/quality trade-off:
//! - Methods 0-2: Fast, basic mode selection
//! - Methods 3-4: Better mode decisions, perceptual optimizations
//! - Methods 5-6: Exhaustive search, trellis quantization

use alloc::vec;
use alloc::vec::Vec;
use archmage::prelude::*;
use core::mem;

#[allow(unused_imports)]
use whereat::at;

use super::api::EncodeError;
use super::api::PixelLayout;
use super::arithmetic::ArithmeticEncoder;
use super::cost::{
    LevelCosts, ProbaStats, analyze_image, assign_segments_kmeans, classify_image_type,
    compute_segment_quant, content_type_to_tuning,
};
use super::vec_writer::VecWriter;
use crate::common::prediction::*;
use crate::common::types::Frame;
use crate::common::types::*;
use crate::encoder::tables::VP8_AC_TABLE2;
// convert_image_sharp_yuv_with_config is called via full path below
use crate::decoder::yuv::convert_image_y;

mod header;
pub(crate) mod mode_selection;
mod prediction;
mod residuals;

//------------------------------------------------------------------------------
// Quality to quantization index mapping
//
// Use centralized functions from fast_math module

use super::fast_math::quality_to_quant_index;

//------------------------------------------------------------------------------
// Quality search state for target_size convergence
//
// Ported from libwebp src/enc/frame_enc.c PassStats

/// Convergence threshold for quality search (ported from libwebp DQ_LIMIT).
/// Quality search is considered converged when |dq| < DQ_LIMIT.
const DQ_LIMIT: f32 = 0.4;

/// State for quality search convergence (target size or PSNR).
/// Uses secant method to interpolate toward target value.
/// Ported from libwebp's PassStats struct.
struct PassStats {
    is_first: bool,
    dq: f32,
    q: f32,
    last_q: f32,
    qmin: f32,
    qmax: f32,
    value: f64,      // current encoded size
    last_value: f64, // previous encoded size
    target: f64,     // target size
}

impl PassStats {
    /// Initialize pass stats for target size search.
    fn new_for_size(target_size: u32, quality: u8, qmin: u8, qmax: u8) -> Self {
        let qmin_f = f32::from(qmin);
        let qmax_f = f32::from(qmax);
        let q = f32::from(quality).clamp(qmin_f, qmax_f);
        Self {
            is_first: true,
            dq: 10.0,
            q,
            last_q: q,
            qmin: qmin_f,
            qmax: qmax_f,
            value: 0.0,
            last_value: 0.0,
            target: f64::from(target_size),
        }
    }

    /// Initialize pass stats for target PSNR search.
    /// PSNR increases with quality, so the secant direction is reversed
    /// compared to size search (higher quality = higher PSNR).
    fn new_for_psnr(target_psnr: f32, quality: u8, qmin: u8, qmax: u8) -> Self {
        let qmin_f = f32::from(qmin);
        let qmax_f = f32::from(qmax);
        let q = f32::from(quality).clamp(qmin_f, qmax_f);
        Self {
            is_first: true,
            dq: 10.0,
            q,
            last_q: q,
            qmin: qmin_f,
            qmax: qmax_f,
            value: 0.0,
            last_value: 0.0,
            target: f64::from(target_psnr),
        }
    }

    /// Compute next quality value using secant method.
    /// Returns the new quality to try.
    fn compute_next_q(&mut self) -> f32 {
        let dq = if self.is_first {
            // First iteration: move in direction of target
            self.is_first = false;
            if self.value > self.target {
                -self.dq
            } else {
                self.dq
            }
        } else if (self.value - self.last_value).abs() > f64::EPSILON {
            // Secant method: linear interpolation to find next q
            let slope = (self.target - self.value) / (self.last_value - self.value);
            (slope * f64::from(self.last_q - self.q)) as f32
        } else {
            0.0 // converged
        };

        // Limit dq to avoid large swings
        self.dq = dq.clamp(-30.0, 30.0);
        self.last_q = self.q;
        self.last_value = self.value;
        self.q = (self.q + self.dq).clamp(self.qmin, self.qmax);
        self.q
    }

    /// Check if convergence is reached.
    fn is_converged(&self) -> bool {
        self.dq.abs() <= DQ_LIMIT
    }
}

//------------------------------------------------------------------------------
// SSE (Sum of Squared Errors) distortion functions
//
// These measure the distortion between source and predicted blocks.
// Lower SSE = better prediction = less data to encode.

/// Compute SSE for a 16x16 luma block within bordered prediction buffer
/// Compares source YUV data against predicted block with border
#[inline]
pub(crate) fn sse_16x16_luma(
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    incant!(
        sse_16x16_luma_dispatch(src_y, src_width, mbx, mby, pred),
        [v3, neon, wasm128, scalar]
    )
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn sse_16x16_luma_dispatch_v3(
    _token: X64V3Token,
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    crate::common::simd_sse::sse_16x16_luma_sse2(_token, src_y, src_width, mbx, mby, pred)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn sse_16x16_luma_dispatch_neon(
    token: NeonToken,
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    crate::common::simd_neon::sse_16x16_luma_neon(token, src_y, src_width, mbx, mby, pred)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn sse_16x16_luma_dispatch_wasm128(
    token: Wasm128Token,
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    crate::common::simd_wasm::sse_16x16_luma_wasm(token, src_y, src_width, mbx, mby, pred)
}

#[inline(always)]
fn sse_16x16_luma_dispatch_scalar(
    _token: ScalarToken,
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    let mut sse = 0u32;
    let src_base = mby * 16 * src_width + mbx * 16;
    for y in 0..16 {
        let src_row = src_base + y * src_width;
        let pred_row = (y + 1) * LUMA_STRIDE + 1;
        for x in 0..16 {
            let diff = i32::from(src_y[src_row + x]) - i32::from(pred[pred_row + x]);
            sse += (diff * diff) as u32;
        }
    }
    sse
}

/// Compute SSE for an 8x8 chroma block within bordered prediction buffer
#[inline]
pub(crate) fn sse_8x8_chroma(
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    incant!(
        sse_8x8_chroma_dispatch(src_uv, src_width, mbx, mby, pred),
        [v3, neon, wasm128, scalar]
    )
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn sse_8x8_chroma_dispatch_v3(
    _token: X64V3Token,
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    crate::common::simd_sse::sse_8x8_chroma_sse2(_token, src_uv, src_width, mbx, mby, pred)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn sse_8x8_chroma_dispatch_neon(
    token: NeonToken,
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    crate::common::simd_neon::sse_8x8_chroma_neon(token, src_uv, src_width, mbx, mby, pred)
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn sse_8x8_chroma_dispatch_wasm128(
    token: Wasm128Token,
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    crate::common::simd_wasm::sse_8x8_chroma_wasm(token, src_uv, src_width, mbx, mby, pred)
}

#[inline(always)]
fn sse_8x8_chroma_dispatch_scalar(
    _token: ScalarToken,
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    let mut sse = 0u32;
    let src_base = mby * 8 * src_width + mbx * 8;
    for y in 0..8 {
        let src_row = src_base + y * src_width;
        let pred_row = (y + 1) * CHROMA_STRIDE + 1;
        for x in 0..8 {
            let diff = i32::from(src_uv[src_row + x]) - i32::from(pred[pred_row + x]);
            sse += (diff * diff) as u32;
        }
    }
    sse
}

// currently in decoder it actually stores this information on the macroblock but that's confusing
// because it doesn't update the macroblock, just the complexity values as we decode
// this is used as the complexity per 13.3 in the decoder
#[derive(Clone, Copy, Default)]
struct Complexity {
    y2: u8,
    y: [u8; 4],
    u: [u8; 2],
    v: [u8; 2],
}

impl Complexity {
    fn clear(&mut self, include_y2: bool) {
        self.y = [0; 4];
        self.u = [0; 2];
        self.v = [0; 2];
        if include_y2 {
            self.y2 = 0;
        }
    }
}

#[derive(Default)]
struct QuantizationIndices {
    yac_abs: u8,
    ydc_delta: Option<i8>,
    y2dc_delta: Option<i8>,
    y2ac_delta: Option<i8>,
    uvdc_delta: Option<i8>,
    uvac_delta: Option<i8>,
}

/// TODO: Consider merging this with the MacroBlock from the decoder
#[derive(Clone, Copy, Default)]
struct MacroblockInfo {
    luma_mode: LumaMode,
    // note ideally this would be on LumaMode::B
    // since that it's where it's valid but need to change the decoder to
    // work with that as well
    luma_bpred: Option<[IntraMode; 16]>,
    chroma_mode: ChromaMode,
    // whether the macroblock uses custom segment values
    // if None, will use the frame level values
    segment_id: Option<usize>,

    coeffs_skipped: bool,
}

pub(super) type ChromaCoeffs = [i32; 16 * 4];

/// Quantized zigzag coefficients for a macroblock, stored for multi-pass encoding.
/// libwebp stores quantized coefficients from pass 1 and reuses them in pass 2+.
/// These are the final quantized values (post-trellis if applicable), ready for
/// direct token recording without re-quantization.
#[derive(Clone)]
struct QuantizedMbCoeffs {
    /// Y2 DC transform coefficients (16 values), only used for I16 mode
    y2_zigzag: [i32; 16],
    /// Y1 block coefficients (16 blocks × 16 values), zigzag order
    y1_zigzag: [[i32; 16]; 16],
    /// U block coefficients (4 blocks × 16 values), zigzag order
    u_zigzag: [[i32; 16]; 4],
    /// V block coefficients (4 blocks × 16 values), zigzag order
    v_zigzag: [[i32; 16]; 4],
}

impl QuantizedMbCoeffs {
    /// Pre-allocated zero coefficients for skipped macroblocks.
    const ZERO: Self = Self {
        y2_zigzag: [0; 16],
        y1_zigzag: [[0; 16]; 16],
        u_zigzag: [[0; 16]; 4],
        v_zigzag: [[0; 16]; 4],
    };

    /// Check if all coefficients are zero (for skip detection).
    /// Uses bitwise OR accumulator — faster than iterating with early-exit
    /// because it avoids branch mispredictions on the common non-zero case.
    #[inline]
    fn is_all_zero(&self, is_i4: bool, first_coeff_y1: usize) -> bool {
        let mut acc: u32 = 0;
        // Check Y2 (only for I16 mode)
        if !is_i4 {
            for &c in &self.y2_zigzag {
                acc |= c as u32;
            }
        }
        // Check Y1 blocks
        for block in &self.y1_zigzag {
            for &c in &block[first_coeff_y1..] {
                acc |= c as u32;
            }
        }
        // Check U blocks
        for block in &self.u_zigzag {
            for &c in block {
                acc |= c as u32;
            }
        }
        // Check V blocks
        for block in &self.v_zigzag {
            for &c in block {
                acc |= c as u32;
            }
        }
        acc == 0
    }

    /// Returns true if this MB matches libwebp's "blocky I16" gating
    /// condition for `StoreMaxDelta` (`quant_enc.c:1111`):
    /// Y2 has any nonzero AND all 16 Y1 AC coefficients are zero.
    /// In libwebp's `nz` mask this is `(nz & 0x100ffff) == 0x1000000`.
    /// Caller must already know the MB is I16 (Y2 only exists then).
    #[inline]
    fn is_blocky_i16(&self) -> bool {
        // Y1 AC = positions 1..16 of every Y1 block (DC is position 0).
        let mut y1_ac_acc: u32 = 0;
        for block in &self.y1_zigzag {
            for &c in &block[1..16] {
                y1_ac_acc |= c as u32;
            }
        }
        if y1_ac_acc != 0 {
            return false;
        }
        let mut y2_acc: u32 = 0;
        for &c in &self.y2_zigzag {
            y2_acc |= c as u32;
        }
        y2_acc != 0
    }
}

struct Vp8Encoder<'a> {
    writer: &'a mut Vec<u8>,
    frame: Frame,
    /// The encoder for the macroblock headers and the compressed frame header
    encoder: ArithmeticEncoder,
    segments: [Segment; MAX_SEGMENTS],
    segments_enabled: bool,
    segments_update_map: bool,
    segment_tree_probs: [Prob; 3],
    /// Segment ID for each macroblock (mb_width * mb_height)
    segment_map: Vec<u8>,

    loop_filter_adjustments: bool,
    macroblock_no_skip_coeff: Option<u8>,
    quantization_indices: QuantizationIndices,

    token_probs: TokenProbTables,
    /// Token statistics for adaptive probability updates
    proba_stats: ProbaStats,
    /// Updated probabilities computed from statistics
    updated_probs: Option<TokenProbTables>,
    /// Precomputed level costs for coefficient cost estimation
    level_costs: LevelCosts,
    /// Whether to use trellis quantization for better RD optimization
    do_trellis: bool,
    /// Whether to use trellis during mode selection (RD_OPT_TRELLIS_ALL, method >= 6)
    do_trellis_i4_mode: bool,
    /// Whether to use chroma error diffusion to reduce banding
    do_error_diffusion: bool,
    /// Encoding method (0-6): 0=fastest, 6=best quality
    method: u8,
    /// Spatial noise shaping strength (0-100)
    sns_strength: u8,
    /// Preprocessing options. Matches libwebp's `WebPConfig::preprocessing`.
    /// Default `Preprocessing::none()` (off), matching libwebp's default.
    smooth_segment_map: bool,
    /// Cost model selection (mode selection + trellis). Default
    /// `ZenwebpDefault` enables perceptual extensions per method level;
    /// `StrictLibwebpParity` disables them.
    cost_model: super::api::CostModel,
    /// Run a stat-collection pre-pass (m4 only — m5/m6 already saturate).
    /// Default `false`. Set via `LossyConfig::with_multi_pass_stats(true)`.
    multi_pass_stats: bool,
    /// Loop filter strength (0-100)
    filter_strength: u8,
    /// Loop filter sharpness (0-7)
    filter_sharpness: u8,
    /// Number of segments (1-4)
    num_segments: u8,
    /// Selected preset (used for Auto detection)
    preset: super::api::Preset,
    /// Partition limit (0-100): extra I4 penalty to prevent partition 0 overflow
    partition_limit: u8,

    top_complexity: Vec<Complexity>,
    left_complexity: Complexity,

    top_b_pred: Vec<IntraMode>,
    left_b_pred: [IntraMode; 4],

    macroblock_width: u16,
    macroblock_height: u16,

    /// Partitions of encoders for the macroblock coefficient data
    partitions: Vec<ArithmeticEncoder>,

    // the left borders used in prediction
    left_border_y: [u8; 16 + 1],
    left_border_u: [u8; 8 + 1],
    left_border_v: [u8; 8 + 1],

    // the top borders used in prediction
    top_border_y: Vec<u8>,
    top_border_u: Vec<u8>,
    top_border_v: Vec<u8>,

    // Error diffusion state for chroma DC coefficients
    // This implements Floyd-Steinberg-like error spreading to reduce banding
    // top_derr[mbx][channel][0..2] = errors from block above
    // left_derr[channel][0..2] = errors from block to the left
    top_derr: Vec<[[i8; 2]; 2]>,
    left_derr: [[i8; 2]; 2],

    /// Token buffer for deferred coefficient encoding (method >= 2).
    /// Stores bit-level tokens during the recording pass for later emission
    /// with optimized probability tables.
    token_buffer: Option<residuals::TokenBuffer>,
    /// Stored macroblock info from token recording pass, used to write
    /// headers without redoing mode selection.
    stored_mb_info: Vec<MacroblockInfo>,
    /// Stored quantized coefficients for token buffer approach.
    /// Mode decisions + quantized zigzag coefficients stored during encoding.
    stored_mb_coeffs: Vec<QuantizedMbCoeffs>,
    /// Maximum observed edge magnitude per segment (port of libwebp's
    /// `VP8SegmentInfo::max_edge`, `vp8i_enc.h:199`). Updated per-MB by
    /// `store_max_delta` when a "blocky" I16 macroblock is detected (Y2
    /// nonzero, all Y1 AC zero), then consumed by `adjust_filter_strength`
    /// after the encode loop to bump per-segment loop-filter levels for
    /// edge-rich segments. See libwebp `quant_enc.c:1031-1040, 1108-1113`
    /// and `filter_enc.c:198-237` (`VP8AdjustFilterStrength`). #34
    max_edge_per_segment: [i32; MAX_SEGMENTS],
}

impl<'a> Vp8Encoder<'a> {
    fn new(writer: &'a mut Vec<u8>) -> Self {
        Self {
            writer,
            frame: Frame::default(),
            encoder: ArithmeticEncoder::new(),
            segments: core::array::from_fn(|_| Segment::default()),
            segments_enabled: false,
            segments_update_map: false,
            segment_tree_probs: [255, 255, 255], // Default probs
            segment_map: Vec::new(),

            loop_filter_adjustments: false,
            macroblock_no_skip_coeff: None,
            quantization_indices: QuantizationIndices::default(),

            token_probs: Default::default(),
            proba_stats: ProbaStats::new(),
            updated_probs: None,
            level_costs: LevelCosts::new(),
            // Trellis quantization for RD-optimized coefficient selection.
            // Uses proper probability-dependent init/EOB/skip costs from LevelCosts.
            do_trellis: true,
            // Trellis during I4 mode selection (RD_OPT_TRELLIS_ALL) - method 6
            do_trellis_i4_mode: false,
            // Error diffusion improves quality in smooth gradients
            do_error_diffusion: true,
            // Default to balanced method
            method: 4,
            sns_strength: 50,
            smooth_segment_map: false,
            cost_model: super::api::CostModel::ZenwebpDefault,
            multi_pass_stats: false,
            filter_strength: 60,
            filter_sharpness: 0,
            num_segments: 4,
            preset: super::api::Preset::Default,
            partition_limit: 0,

            top_complexity: Vec::new(),
            left_complexity: Complexity::default(),

            top_b_pred: Vec::new(),
            left_b_pred: [IntraMode::default(); 4],

            macroblock_width: 0,
            macroblock_height: 0,

            partitions: vec![ArithmeticEncoder::new()],

            left_border_y: [0u8; 16 + 1],
            left_border_u: [0u8; 8 + 1],
            left_border_v: [0u8; 8 + 1],
            top_border_y: Vec::new(),
            top_border_u: Vec::new(),
            top_border_v: Vec::new(),

            // Error diffusion starts with zero
            top_derr: Vec::new(),
            left_derr: [[0; 2]; 2],

            token_buffer: None,
            stored_mb_info: Vec::new(),
            stored_mb_coeffs: Vec::new(),
            max_edge_per_segment: [0; MAX_SEGMENTS],
        }
    }

    /// Get the segment for a macroblock at (mbx, mby).
    ///
    /// When segments are enabled, looks up the segment ID from the segment map.
    /// Otherwise, returns segment 0.
    #[inline]
    fn get_segment_for_mb(&self, mbx: usize, mby: usize) -> &Segment {
        let segment_id = if self.segments_enabled && !self.segment_map.is_empty() {
            let mb_idx = mby * usize::from(self.macroblock_width) + mbx;
            self.segment_map[mb_idx] as usize
        } else {
            0
        };
        &self.segments[segment_id]
    }

    /// Get segment ID for a macroblock at (mbx, mby).
    #[inline]
    fn get_segment_id_for_mb(&self, mbx: usize, mby: usize) -> Option<usize> {
        if self.segments_enabled && !self.segment_map.is_empty() {
            let mb_idx = mby * usize::from(self.macroblock_width) + mbx;
            Some(self.segment_map[mb_idx] as usize)
        } else {
            None
        }
    }

    /// Compute updated probabilities from recorded statistics.
    ///
    /// IMPORTANT: For multi-pass encoding, this computes the optimal probabilities
    /// to use for emission. The header encoder compares against COEFF_PROBS (decoder
    /// defaults) to decide which updates to signal.
    ///
    /// Returns true if any probabilities were updated (changed from COEFF_PROBS defaults).
    /// This matches libwebp's FinalizeTokenProbas which sets dirty = has_changed where
    /// has_changed is true only if any coefficient was set to a value different from default.
    fn compute_updated_probabilities(&mut self) -> bool {
        // Always start from COEFF_PROBS (decoder defaults) for computing what to update.
        // This ensures the header signaling matches what the decoder expects.
        let mut updated = COEFF_PROBS;
        let mut has_changed = false;

        for t in 0..4 {
            for b in 0..8 {
                for c in 0..3 {
                    for p in 0..11 {
                        // Compare against COEFF_PROBS (decoder defaults), not token_probs
                        let default_prob = COEFF_PROBS[t][b][c][p];
                        let update_prob = COEFF_UPDATE_PROBS[t][b][c][p];

                        let (should_update, new_p, _savings) =
                            self.proba_stats
                                .should_update(t, b, c, p, default_prob, update_prob);

                        // Update if savings are positive, matching libwebp's approach.
                        // The signaling cost (8 bits for value + 1 bit flag) is already
                        // included in the should_update calculation.
                        if should_update {
                            updated[t][b][c][p] = new_p;
                            // has_changed is true if new_p differs from default (matching libwebp)
                            has_changed |= new_p != default_prob;
                        }
                    }
                }
            }
        }

        // Always set updated_probs with the computed values.
        // For multi-pass, this ensures the header has the final probabilities to signal.
        // If no updates are beneficial, updated will equal COEFF_PROBS.
        self.updated_probs = Some(updated);

        has_changed
    }

    /// Reset encoder state for a new encoding pass.
    /// Used by multi-pass encoding (method >= 5) to reset all state before
    /// re-encoding with updated probability/cost tables.
    fn reset_for_new_pass(&mut self) {
        // Reset complexity tracking
        for complexity in self.top_complexity.iter_mut() {
            *complexity = Complexity::default();
        }
        self.left_complexity = Complexity::default();

        // Reset B-pred tracking
        for pred in self.top_b_pred.iter_mut() {
            *pred = IntraMode::default();
        }
        self.left_b_pred = [IntraMode::default(); 4];

        // Reset border pixels to initial state
        self.left_border_y = [129u8; 16 + 1];
        self.left_border_u = [129u8; 8 + 1];
        self.left_border_v = [129u8; 8 + 1];

        for val in self.top_border_y.iter_mut() {
            *val = 127;
        }
        for val in self.top_border_u.iter_mut() {
            *val = 127;
        }
        for val in self.top_border_v.iter_mut() {
            *val = 127;
        }

        // Reset chroma error diffusion state
        self.left_derr = [[0; 2]; 2];
        for derr in self.top_derr.iter_mut() {
            *derr = [[0; 2]; 2];
        }

        // Estimate output size: ~0.3 bytes per pixel is conservative for most quality levels
        let num_pixels =
            usize::from(self.macroblock_width) * usize::from(self.macroblock_height) * 256; // 16x16 per macroblock
        let estimated_partition_size = num_pixels / 4; // ~0.25 bytes per pixel for coefficients

        // Reset partitions with pre-allocated capacity
        self.partitions = vec![ArithmeticEncoder::with_capacity(estimated_partition_size)];

        // Reset encoder (header is small, ~1KB is plenty)
        self.encoder = ArithmeticEncoder::with_capacity(1024);

        // Reset per-segment max edge tracking for this encode (#34).
        self.max_edge_per_segment = [0; MAX_SEGMENTS];
    }

    /// Update `max_edge_per_segment` for a "blocky" I16 macroblock.
    ///
    /// Port of libwebp's `StoreMaxDelta` (`quant_enc.c:1031-1040`). Looks at
    /// the first three AC coefficients of the Y2 (WHT) block — these encode
    /// the average DC differences between adjacent Y1 sub-blocks, so a large
    /// magnitude here corresponds to visible inter-block edges that the loop
    /// filter should attenuate.
    ///
    /// `y2_zigzag` is in zigzag order, matching libwebp's `y_dc_levels`
    /// (which is the output of `VP8EncQuantizeBlockWHT`, also zigzag-ordered).
    /// Indices 1, 2, 4 correspond to natural-order positions 1, 4, 2.
    #[inline]
    fn store_max_delta(&mut self, segment_id: usize, y2_zigzag: &[i32; 16]) {
        let v0 = y2_zigzag[1].unsigned_abs();
        let v1 = y2_zigzag[2].unsigned_abs();
        let v2 = y2_zigzag[4].unsigned_abs();
        let max_v = v0.max(v1).max(v2) as i32;
        if max_v > self.max_edge_per_segment[segment_id] {
            self.max_edge_per_segment[segment_id] = max_v;
        }
    }

    /// Post-encode adjustment of per-segment loop-filter strengths based on
    /// observed edge magnitudes (port of libwebp's `VP8AdjustFilterStrength`,
    /// `filter_enc.c:198-237`, called from `PostLoopFinalize` in
    /// `frame_enc.c:730`).
    ///
    /// For each segment, computes a target filter strength from the maximum
    /// observed Y2 edge magnitude scaled by the Y2 AC quantizer:
    /// `delta = (max_edge * y2.q[1]) >> 3` (the `>> 3` accounts for inverse
    /// WHT scaling). This delta is mapped to a filter level via
    /// `filter_strength_from_delta`, and the per-segment level is bumped
    /// upward only — never lowered. Finally the frame-level
    /// `filter_level` is raised to the maximum per-segment fstrength.
    ///
    /// Effect: edge-rich segments (text/charts) get stronger filtering than
    /// the analysis-time estimate, reducing blocking artifacts.
    ///
    /// Note: zenwebp stores `loopfilter_level` as a signed delta from
    /// `frame.filter_level`, while libwebp stores it as an absolute
    /// `dqm.fstrength`. We convert to absolute, bump, then recompute the
    /// delta against the (possibly raised) frame-level filter.
    ///
    /// Note vs libwebp: we do not gate on `D > min_disto` (the per-MB
    /// distortion check from `quant_enc.c:1111`). zenwebp's encoder does
    /// not currently thread that distortion through to this path. Skipping
    /// it is conservative — it lets a few extra blocky MBs contribute,
    /// which can only bump the strength upward, matching libwebp's
    /// "monotone-up" behavior.
    fn adjust_filter_strength(&mut self) {
        if self.filter_strength == 0 {
            return;
        }

        let mut absolute = [0i32; MAX_SEGMENTS];
        let frame_level = i32::from(self.frame.filter_level);
        for s in 0..MAX_SEGMENTS {
            absolute[s] = frame_level + i32::from(self.segments[s].loopfilter_level);
        }

        let mut max_level = 0i32;
        for s in 0..MAX_SEGMENTS {
            let max_edge = self.max_edge_per_segment[s];
            // y2.q[1] is the Y2 AC quantizer in libwebp's expanded matrix.
            // zenwebp stores it directly on Segment as `y2ac` (i16).
            let y2_q1 = i32::from(self.segments[s].y2ac);
            let delta_raw = max_edge.saturating_mul(y2_q1) >> 3;
            // Clamp to u8 for the table lookup (table saturates at 63 well
            // before MAX_DELTA_SIZE=64; libwebp clamps inside the function).
            let delta_u8 = delta_raw.clamp(0, 255) as u8;
            let edge_strength = i32::from(super::cost::filter_strength_from_delta(
                self.filter_sharpness,
                delta_u8,
            ));
            if edge_strength > absolute[s] {
                absolute[s] = edge_strength;
            }
            if absolute[s] > max_level {
                max_level = absolute[s];
            }
        }

        // Raise frame filter_level to the max per-segment fstrength
        // (matches libwebp `enc->filter_hdr.level = max_level`).
        let new_frame_level = max_level.clamp(0, 63) as u8;
        self.frame.filter_level = new_frame_level;

        // Recompute per-segment deltas against the new frame level. The
        // signed-6-bit field in the bitstream covers [-63, 63], which is
        // sufficient since both operands are in [0, 63].
        let new_frame_i32 = i32::from(new_frame_level);
        for s in 0..MAX_SEGMENTS {
            let delta = absolute[s] - new_frame_i32;
            self.segments[s].loopfilter_level = delta.clamp(-63, 63) as i8;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_image(
        &mut self,
        data: &[u8],
        color: PixelLayout,
        width: u16,
        height: u16,
        stride: usize,
        params: &super::api::EncoderParams,
        stop: &dyn enough::Stop,
        progress: &dyn super::api::EncodeProgress,
    ) -> Result<super::api::EncodeStats, EncodeError> {
        // Store method and configure features based on it
        self.method = params.method.min(6); // Clamp to 0-6
        // Method feature mapping (aligned with libwebp):
        //   m0-2: RD_OPT_NONE - fast mode, no RD optimization
        //   m3-4: RD_OPT_BASIC - RD scoring for mode selection, no trellis
        //   m5:   RD_OPT_TRELLIS - trellis quantization during encoding
        //   m6:   RD_OPT_TRELLIS_ALL - trellis during I4 mode selection
        self.do_trellis = self.method >= 5;
        self.do_trellis_i4_mode = self.method >= 6;
        // Store tuning parameters
        self.sns_strength = params.sns_strength.min(100);
        self.smooth_segment_map = params.smooth_segment_map;
        self.cost_model = params.cost_model;
        self.multi_pass_stats = params.multi_pass_stats;
        self.filter_strength = params.filter_strength.min(100);
        self.filter_sharpness = params.filter_sharpness.min(7);
        self.num_segments = params.num_segments.clamp(1, 4);
        self.preset = params.preset;
        self.partition_limit = params.partition_limit.unwrap_or(0).min(100);
        // For ARGB input, convert to RGBA so the standard RGBA code path handles it.
        let argb_converted;
        let (data, color) = if color == PixelLayout::Argb8 {
            let w = usize::from(width);
            let h = usize::from(height);
            let bpp = 4usize;
            let stride_bytes = stride * bpp;
            let row_bytes = w * bpp;
            let mut out = alloc::vec![0u8; w * h * 4];
            for y in 0..h {
                garb::bytes::argb_to_rgba(
                    &data[y * stride_bytes..y * stride_bytes + row_bytes],
                    &mut out[y * w * 4..(y + 1) * w * 4],
                )
                .expect("validated buffer sizes");
            }
            argb_converted = out;
            (argb_converted.as_slice(), PixelLayout::Rgba8)
        } else {
            (data, color)
        };

        let (y_bytes, u_bytes, v_bytes) = if color == PixelLayout::Yuv420 {
            // YUV420 planar data: [Y, U, V] packed into a single buffer
            let w = usize::from(width);
            let h = usize::from(height);
            let y_size = w * h;
            let uv_w = w.div_ceil(2);
            let uv_h = h.div_ceil(2);
            let uv_size = uv_w * uv_h;

            let y_plane = &data[..y_size];
            let u_plane = &data[y_size..y_size + uv_size];
            let v_plane = &data[y_size + uv_size..y_size + uv_size * 2];

            crate::decoder::yuv::import_yuv420_planes(y_plane, u_plane, v_plane, width, height)
        } else if let Some(sharp_cfg) = &params.sharp_yuv {
            crate::decoder::yuv::convert_image_sharp_yuv_with_config(
                data, color, width, height, stride, *sharp_cfg,
            )
        } else {
            match color {
                // zenyuv (SIMD Y) + gamma-corrected scalar chroma.
                // Matches libwebp chroma quality; Y is within ±2 levels of scalar.
                PixelLayout::Rgb8 | PixelLayout::Rgba8 | PixelLayout::Bgr8 | PixelLayout::Bgra8 => {
                    crate::decoder::yuv::convert_image_yuv_fast(data, color, width, height, stride)
                }
                PixelLayout::L8 => convert_image_y::<1>(data, width, height, stride),
                PixelLayout::La8 => convert_image_y::<2>(data, width, height, stride),
                PixelLayout::Yuv420 | PixelLayout::Argb8 => unreachable!(),
            }
        };

        if color != PixelLayout::Yuv420 {
            let bpp = match color {
                PixelLayout::L8 => 1usize,
                PixelLayout::La8 => 2,
                PixelLayout::Rgb8 | PixelLayout::Bgr8 => 3,
                PixelLayout::Rgba8 | PixelLayout::Bgra8 | PixelLayout::Argb8 => 4,
                PixelLayout::Yuv420 => unreachable!(),
            };
            let w = usize::from(width);
            let h = usize::from(height);
            let min_size = if h > 0 {
                stride * bpp * (h - 1) + w * bpp
            } else {
                0
            };
            assert!(
                data.len() >= min_size,
                "buffer too small: got {}, need at least {} for {}x{} stride={} {:?}",
                data.len(),
                min_size,
                w,
                h,
                stride,
                color
            );
        }

        self.setup_encoding(
            params.lossy_quality,
            width,
            height,
            y_bytes,
            u_bytes,
            v_bytes,
        );

        // Calculate initial level costs for mode selection and trellis
        if self.level_costs.is_dirty() {
            self.level_costs.calculate(&self.token_probs);
        }

        // Token buffer encoding (matching libwebp's VP8EncTokenLoop).
        //
        // Records coefficient decisions as compact tokens while collecting
        // probability statistics with mid-stream refresh. For method >= 5,
        // performs multiple passes: each pass does FULL re-encoding with
        // level_costs derived from the previous pass's observed probabilities.
        //
        // Benefits of multi-pass (matches libwebp's approach):
        // - Image-specific probability tables (not generic defaults)
        // - Better mode selection with empirical cost tables
        // - More efficient arithmetic coding with observed probabilities
        //
        // Key insight: Each pass does full re-encoding (predict → residual → DCT →
        // quantize → record). Mode decisions and quantization may differ between
        // passes because level_costs change. Borders come from reconstructed pixels,
        // so each pass has consistent predictions within itself.

        let num_mb = usize::from(self.macroblock_width) * usize::from(self.macroblock_height);

        // Number of passes based on method (mirrors libwebp's `StatLoop` behavior).
        // libwebp runs `config->pass` stat-collection passes before the final emit
        // pass, refreshing both `proba` and `level_costs` between iterations
        // (`frame_enc.c:626-684, 795-906, 840-844`). The first pass uses default
        // costs; the second pass benefits from image-tuned probabilities, which
        // makes mode selection and trellis pick options that are actually cheap
        // under the real distribution.
        //
        // Earlier in zenwebp's history multi-pass was tested without refreshing
        // `level_costs` mid-pass and reportedly hurt compression — but the
        // existing infrastructure here (lines 884-895) DOES rebuild level_costs
        // between passes, so the prior negative result no longer applies. #27.
        //
        // 2 passes at m4 only. m5/m6 already use trellis quantization, which is
        // image-adapted via the per-pass `proba_stats` accumulation; adding a
        // second pass at m5/m6 measurably regresses size (+0.6 to +0.9% on
        // CID22 — see `differences/baselines/post-batch2-27.tsv`). At m4 the
        // second pass net-helps (-0.1 to -0.4%) because the simple-quant
        // m4 path doesn't have a per-MB feedback mechanism otherwise.
        // Multi-pass stat collection is opt-in via `LossyConfig::with_multi_pass_stats(true)`
        // (default OFF, gated only at m4). m5/m6 already image-adapt via per-pass
        // `proba_stats` in trellis; adding a second pass at those tiers regresses size
        // (see #27 investigation). Multi-pass at m4 doubles encode time for ~0.1% size
        // win on photos — useful inside `target_size`/`target_zensim` search loops.
        let num_passes: usize = if self.multi_pass_stats && self.method == 4 {
            2
        } else {
            1
        };

        // Stats accumulators (populated during last pass)
        let mut final_sse_y: u64 = 0;
        let mut final_sse_u: u64 = 0;
        let mut final_sse_v: u64 = 0;
        let mut final_block_count_i4: u32 = 0;
        let mut final_block_count_i16: u32 = 0;
        let mut final_skip_mb: u32 = 0;

        for pass in 0..num_passes {
            let is_last_pass = pass == num_passes - 1;

            // Clear token buffer for this pass
            self.token_buffer = Some(residuals::TokenBuffer::with_estimated_capacity(num_mb));

            // Reset statistics only on first pass or last pass (matches libwebp).
            // For intermediate passes, statistics ACCUMULATE across passes, giving
            // a more robust probability estimate based on multiple encodings.
            if pass == 0 || is_last_pass {
                self.proba_stats.reset();
            }

            // Clear stored info - we only keep the last pass's results
            self.stored_mb_info.clear();
            self.stored_mb_info.reserve(num_mb);
            // stored_mb_coeffs is only needed for multi-pass re-encoding.
            // Since num_passes == 1, skip the 1.6MB allocation entirely.
            if num_passes > 1 {
                self.stored_mb_coeffs.clear();
                self.stored_mb_coeffs.reserve(num_mb);
            }

            if pass > 0 {
                // Pass 1+: Apply updated probabilities from previous pass
                // This gives us image-specific cost tables for mode selection and trellis
                if let Some(ref updated) = self.updated_probs {
                    self.token_probs = *updated;
                }

                // Recalculate level_costs from the new probabilities
                // This is the key to multi-pass: trellis and mode selection use
                // empirical costs, potentially making different (better) decisions
                self.level_costs.mark_dirty();
                self.level_costs.calculate(&self.token_probs);
            }

            // Reset all encoder state for this pass
            self.reset_for_new_pass();

            // Mid-stream refresh interval: roughly every total_mb/8 macroblocks
            let max_count = (num_mb / 8).max(96) as i32; // MIN_COUNT = 96 (matches libwebp)
            let mut refresh_countdown = max_count;

            let mut total_mb: u32 = 0;
            let mut skip_mb: u32 = 0;
            let mut block_count_i4: u32 = 0;
            let mut block_count_i16: u32 = 0;
            // SSE accumulators for PSNR computation
            let mut sse_y: u64 = 0;
            let mut sse_u: u64 = 0;
            let mut sse_v: u64 = 0;
            let y_stride = usize::from(self.macroblock_width) * 16;
            let uv_stride = usize::from(self.macroblock_width) * 8;
            let mut last_progress_pct: u8 = 0;

            // ===== ENCODING PASS =====
            // Each pass does full encoding: mode selection + transform + quantize + record
            for mby in 0..self.macroblock_height {
                // Reset left state for start of row
                self.left_complexity = Complexity::default();
                self.left_b_pred = [IntraMode::default(); 4];
                self.left_derr = [[0; 2]; 2]; // reset chroma error diffusion for row start
                self.left_border_y = [129u8; 16 + 1];
                self.left_border_u = [129u8; 8 + 1];
                self.left_border_v = [129u8; 8 + 1];

                for mbx in 0..self.macroblock_width {
                    // Check for cancellation every 16 macroblocks
                    if total_mb & 15 == 0 {
                        stop.check()?;
                    }

                    // Mid-stream probability refresh (like libwebp's VP8EncTokenLoop)
                    // We update probabilities mid-stream (helps compression: 1.0111x → 1.0101x)
                    // but don't recalculate level_costs (hurts compression: 1.0101x → 1.0114x)
                    refresh_countdown -= 1;
                    if refresh_countdown < 0 {
                        self.compute_updated_probabilities();
                        refresh_countdown = max_count;
                    }

                    let macroblock_info = self.choose_macroblock_info(mbx.into(), mby.into());

                    // Update b_pred context for next macroblock's mode selection
                    // This must happen during encoding pass, not just header writing
                    let mbx_usize = usize::from(mbx);
                    if let Some(bpred) = macroblock_info.luma_bpred {
                        // I4 mode: update with per-block modes
                        // top_b_pred gets the bottom row (row 3)
                        for x in 0..4 {
                            self.top_b_pred[mbx_usize * 4 + x] = bpred[3 * 4 + x];
                        }
                        // left_b_pred gets the rightmost column (column 3)
                        for y in 0..4 {
                            self.left_b_pred[y] = bpred[y * 4 + 3];
                        }
                    } else {
                        // I16 mode: all context slots get the derived intra mode
                        let intra_mode = macroblock_info
                            .luma_mode
                            .into_intra()
                            .unwrap_or(IntraMode::DC);
                        for x in 0..4 {
                            self.top_b_pred[mbx_usize * 4 + x] = intra_mode;
                        }
                        for y in 0..4 {
                            self.left_b_pred[y] = intra_mode;
                        }
                    }

                    // Transform blocks (updates border state for next macroblock)
                    let y_block_data =
                        self.transform_luma_block(mbx.into(), mby.into(), &macroblock_info);

                    let (u_block_data, v_block_data) = self.transform_chroma_blocks(
                        mbx.into(),
                        mby.into(),
                        macroblock_info.chroma_mode,
                    );

                    // Accumulate SSE for PSNR computation (source vs reconstructed)
                    sse_y += u64::from(sse_16x16_luma(
                        &self.frame.ybuf,
                        y_stride,
                        usize::from(mbx),
                        usize::from(mby),
                        &y_block_data.pred_block,
                    ));
                    sse_u += u64::from(sse_8x8_chroma(
                        &self.frame.ubuf,
                        uv_stride,
                        usize::from(mbx),
                        usize::from(mby),
                        &u_block_data.pred_block,
                    ));
                    sse_v += u64::from(sse_8x8_chroma(
                        &self.frame.vbuf,
                        uv_stride,
                        usize::from(mbx),
                        usize::from(mby),
                        &v_block_data.pred_block,
                    ));

                    // Count block types
                    if macroblock_info.luma_mode == LumaMode::B {
                        block_count_i4 += 1;
                    } else {
                        block_count_i16 += 1;
                    }

                    // Quantize and record tokens.
                    //
                    // For non-trellis methods (0-4): quantize once, check for skip,
                    // then record tokens only if non-zero. Saves redundant quantization
                    // that check_all_coeffs_zero would have done separately.
                    //
                    // For trellis methods (5-6): use integrated path since trellis
                    // quantization depends on complexity context updated per-block.
                    total_mb += 1;
                    let is_i4 = macroblock_info.luma_mode == LumaMode::B;
                    let first_coeff_y1 = if is_i4 { 0usize } else { 1 };

                    let mut mb_info = macroblock_info;
                    let store_coeffs = num_passes > 1;
                    // Segment id resolved here so both trellis/non-trellis paths
                    // can update `max_edge_per_segment` (#34).
                    let mb_segment_id = macroblock_info.segment_id.unwrap_or(0);
                    if self.do_trellis {
                        // Trellis path: integrated quantize + record (old behavior)
                        let all_zero = self.check_all_coeffs_zero(
                            &macroblock_info,
                            &y_block_data.coeffs,
                            &u_block_data.coeffs,
                            &v_block_data.coeffs,
                        );
                        if all_zero {
                            skip_mb += 1;
                            mb_info.coeffs_skipped = true;
                            self.left_complexity
                                .clear(macroblock_info.luma_mode != LumaMode::B);
                            self.top_complexity[usize::from(mbx)]
                                .clear(macroblock_info.luma_mode != LumaMode::B);
                            if store_coeffs {
                                self.stored_mb_coeffs.push(QuantizedMbCoeffs::ZERO);
                            }
                        } else {
                            let stored_coeffs = self.record_residual_tokens_storing(
                                &macroblock_info,
                                mbx as usize,
                                &y_block_data.coeffs,
                                &u_block_data.coeffs,
                                &v_block_data.coeffs,
                            );
                            // Track edge magnitude for I16 "blocky" MBs (#34).
                            if !is_i4 && stored_coeffs.is_blocky_i16() {
                                self.store_max_delta(mb_segment_id, &stored_coeffs.y2_zigzag);
                            }
                            if store_coeffs {
                                self.stored_mb_coeffs.push(stored_coeffs);
                            }
                        }
                    } else {
                        // Non-trellis path: quantize once, skip-check, record from stored.
                        // Avoids the redundant quantization in check_all_coeffs_zero.
                        let stored_coeffs = self.quantize_mb_coeffs(
                            &macroblock_info,
                            &y_block_data.coeffs,
                            &u_block_data.coeffs,
                            &v_block_data.coeffs,
                        );
                        let all_zero = stored_coeffs.is_all_zero(is_i4, first_coeff_y1);
                        if all_zero {
                            skip_mb += 1;
                            mb_info.coeffs_skipped = true;
                            self.left_complexity
                                .clear(macroblock_info.luma_mode != LumaMode::B);
                            self.top_complexity[usize::from(mbx)]
                                .clear(macroblock_info.luma_mode != LumaMode::B);
                        } else {
                            // Track edge magnitude for I16 "blocky" MBs (#34).
                            if !is_i4 && stored_coeffs.is_blocky_i16() {
                                self.store_max_delta(mb_segment_id, &stored_coeffs.y2_zigzag);
                            }
                            self.record_from_stored_coeffs(
                                &macroblock_info,
                                mbx as usize,
                                &stored_coeffs,
                            );
                        }
                        // Only store quantized coefficients when multi-pass needs them
                        if store_coeffs {
                            self.stored_mb_coeffs.push(stored_coeffs);
                        }
                    }

                    // Store macroblock info for header writing
                    self.stored_mb_info.push(mb_info);
                }

                // Report progress after each row
                let pct = ((u32::from(mby) + 1) * 100 / u32::from(self.macroblock_height)) as u8;
                if pct > last_progress_pct {
                    last_progress_pct = pct;
                    progress.on_progress(pct.min(99))?; // cap at 99, report 100 after finalize
                }
            }

            // Compute skip probability from actual data.
            // libwebp gates per-MB skip-bit emission on `skip_proba < SKIP_PROBA_THRESHOLD (250)`
            // (libwebp `src/enc/frame_enc.c:118-132` `FinalizeSkipProba` / `use_skip_proba`).
            // When fewer than ~2% of MBs are skip-eligible, the per-MB skip bit costs more than
            // it saves, so libwebp signals `use_skip_proba=0` in the frame header and omits the
            // per-MB skip bits entirely — the decoder then assumes every MB has residual data.
            //
            // libwebp can do this because its token-loop architecture emits residual tokens
            // for every MB unconditionally (an empty MB is just an EOB-at-0 per block, ~1 bit).
            // zenwebp's current architecture suppresses residual emission entirely for skipped
            // MBs, so we can only safely set `use_skip_proba=0` when **no** MB was skipped
            // (else the decoder would expect residual data we did not write).
            //
            // Conservative gate: omit the per-MB skip bit only when `prob >= 250 && skip_mb == 0`.
            // A more complete fix (force-encode the few skipped MBs as non-skip with EOB tokens
            // when prob crosses the threshold) is tracked at #25. This conservative version
            // already saves bits whenever the encoder happened to produce zero skipped MBs.
            if total_mb > 0 {
                let non_skip_mb = total_mb - skip_mb;
                let prob = ((255 * non_skip_mb + total_mb / 2) / total_mb).min(255) as u8;
                const SKIP_PROBA_THRESHOLD: u8 = 250;
                if prob >= SKIP_PROBA_THRESHOLD && skip_mb == 0 {
                    self.macroblock_no_skip_coeff = None;
                } else {
                    self.macroblock_no_skip_coeff = Some(prob.clamp(1, 254));
                }
            }

            // Finalize probabilities from this pass (used by next pass or final emission)
            self.compute_updated_probabilities();

            // Save stats from final pass
            final_sse_y = sse_y;
            final_sse_u = sse_u;
            final_sse_v = sse_v;
            final_block_count_i4 = block_count_i4;
            final_block_count_i16 = block_count_i16;
            final_skip_mb = skip_mb;
        }

        // ===== FINALIZE: write bitstream =====

        // Bump per-segment loop-filter strength based on observed edge
        // magnitudes (port of libwebp's `VP8AdjustFilterStrength`, called
        // from `PostLoopFinalize` in `frame_enc.c:730`). Must run before
        // the header writer reads `frame.filter_level` and per-segment
        // `loopfilter_level` deltas. #34
        self.adjust_filter_strength();

        // Write compressed frame header (includes probability updates)
        self.encode_compressed_frame_header();

        // Write macroblock headers from stored info.
        // Take the vec out to avoid borrow conflict with self.write_macroblock_header.
        let stored_mb_info = mem::take(&mut self.stored_mb_info);

        // Reset b-pred tracking state for header writing
        for pred in self.top_b_pred.iter_mut() {
            *pred = IntraMode::default();
        }
        self.left_b_pred = [IntraMode::default(); 4];

        let mb_w = usize::from(self.macroblock_width);
        for (idx, mb_info) in stored_mb_info.iter().enumerate() {
            let mbx = idx % mb_w;
            if mbx == 0 {
                self.left_b_pred = [IntraMode::default(); 4];
            }
            self.write_macroblock_header(mb_info, mbx);
        }

        // Emit tokens to partition using final probabilities
        let final_probs = self.updated_probs.as_ref().unwrap_or(&self.token_probs);
        let token_buf = self.token_buffer.take().unwrap();
        token_buf.emit_tokens(&mut self.partitions[0], final_probs);

        // Assemble output
        let compressed_header_encoder = mem::take(&mut self.encoder);
        let compressed_header_bytes = compressed_header_encoder.flush_and_get_buffer();

        // VP8 frame tag encodes partition 0 size in 19 bits (max 524,287 bytes).
        // Exceeding this produces a corrupt bitstream that decoders reject.
        const VP8_MAX_PARTITION0_SIZE: u32 = (1 << 19) - 1;
        let partition0_size = compressed_header_bytes.len() as u32;
        if partition0_size > VP8_MAX_PARTITION0_SIZE {
            return Err(at!(EncodeError::Partition0Overflow {
                size: partition0_size,
                max: VP8_MAX_PARTITION0_SIZE,
            })
            .into());
        }

        self.write_uncompressed_frame_header(partition0_size);

        self.writer.write_all(&compressed_header_bytes);

        self.write_partitions();

        // Clean up
        self.stored_mb_info.clear();

        // Build encoding statistics
        let num_pixels_y =
            u64::from(self.macroblock_width) * 16 * u64::from(self.macroblock_height) * 16;
        let num_pixels_uv =
            u64::from(self.macroblock_width) * 8 * u64::from(self.macroblock_height) * 8;

        let psnr_y = sse_to_psnr(final_sse_y, num_pixels_y);
        let psnr_u = sse_to_psnr(final_sse_u, num_pixels_uv);
        let psnr_v = sse_to_psnr(final_sse_v, num_pixels_uv);
        let total_sse = final_sse_y + final_sse_u + final_sse_v;
        let total_pixels = num_pixels_y + 2 * num_pixels_uv;
        let psnr_all = sse_to_psnr(total_sse, total_pixels);

        let mut stats = super::api::EncodeStats {
            psnr: [psnr_y, psnr_u, psnr_v, psnr_all, 0.0],
            block_count_i4: final_block_count_i4,
            block_count_i16: final_block_count_i16,
            block_count_skip: final_skip_mb,
            ..Default::default()
        };

        // Fill segment info
        for (i, segment) in self.segments.iter().enumerate().take(4) {
            stats.segment_quant[i] = segment.quant_index;
            stats.segment_level[i] = self.frame.filter_level;
        }

        progress.on_progress(100)?;

        Ok(stats)
    }

    /// Merge segments with identical quantizer and filter settings.
    ///
    /// This reduces the number of effective segments when different alpha regions
    /// end up with the same quantization and filter parameters. Reducing segment
    /// count can save bits in the bitstream.
    ///
    /// Ported from libwebp's SimplifySegments.
    #[allow(clippy::needless_range_loop)] // s1 indexes both seg_map and self.segments
    fn simplify_segments(&mut self) {
        // Map from old segment ID to new segment ID
        let mut seg_map = [0u8, 1, 2, 3];
        let num_segments = self.num_segments as usize;
        let mut num_final_segments = 1usize;

        // Check each segment starting from 1 to see if it matches an earlier segment
        for s1 in 1..num_segments {
            let seg1 = &self.segments[s1];
            let mut found = false;

            // Check if we already have a segment with same quant_index and loopfilter_level
            for s2 in 0..num_final_segments {
                let seg2 = &self.segments[s2];
                if seg1.quant_index == seg2.quant_index
                    && seg1.loopfilter_level == seg2.loopfilter_level
                {
                    seg_map[s1] = s2 as u8;
                    found = true;
                    break;
                }
            }

            if !found {
                // This is a new unique segment
                seg_map[s1] = num_final_segments as u8;
                if num_final_segments != s1 {
                    // Move segment data to its new position
                    self.segments[num_final_segments] = self.segments[s1].clone();
                }
                num_final_segments += 1;
            }
        }

        // If we reduced segments, remap the segment map
        if num_final_segments < num_segments {
            for seg_id in &mut self.segment_map {
                *seg_id = seg_map[*seg_id as usize];
            }
            self.num_segments = num_final_segments as u8;

            // Replicate trailing segment infos (cosmetic, required by bitstream syntax)
            for i in num_final_segments..num_segments {
                self.segments[i] = self.segments[num_final_segments - 1].clone();
            }
        }
    }

    /// Analyze image complexity and assign macroblocks to segments.
    ///
    /// This performs a DCT-based analysis pass (ported from libwebp) to:
    /// 1. Compute "alpha" (compressibility) for each macroblock using DCT histogram
    /// 2. Build a histogram of alpha values
    /// 3. Use k-means clustering to assign macroblocks to 4 segments
    /// 4. Configure per-segment quantization based on alpha
    ///
    /// Segments allow different quantization for different image regions:
    /// - Flat areas (high alpha from segment perspective) can use more aggressive quantization
    /// - Textured areas (low alpha) need finer quantization to preserve detail
    ///
    /// Ported from libwebp's VP8EncAnalyze / MBAnalyze.
    fn analyze_and_assign_segments(&mut self, base_quant_index: u8, quality: u8) {
        let y_stride = usize::from(self.macroblock_width * 16);
        let uv_stride = usize::from(self.macroblock_width * 8);
        let width = usize::from(self.frame.width);
        let height = usize::from(self.frame.height);

        // Run full DCT-based analysis pass using libwebp-compatible algorithm
        // This tests DC and TM modes for I16 and UV, computes per-MB alpha,
        // and builds the alpha histogram
        let analysis = analyze_image(
            &self.frame.ybuf,
            &self.frame.ubuf,
            &self.frame.vbuf,
            width,
            height,
            y_stride,
            uv_stride,
            self.method,
            self.sns_strength,
            self.cost_model,
        );

        // Auto-detect content type when Preset::Auto is selected.
        // Runs after analyze_image (reuses alpha histogram, nearly free).
        if self.preset == super::api::Preset::Auto {
            let content_type = classify_image_type(
                &self.frame.ybuf,
                width,
                height,
                y_stride,
                &analysis.alpha_histogram,
            );
            let (sns, filter, sharp, segs) = content_type_to_tuning(content_type);
            self.sns_strength = sns;
            self.filter_strength = filter;
            self.filter_sharpness = sharp;
            self.num_segments = segs;
        }

        // Use k-means to assign segments
        // weighted_average is computed from final cluster centers, matching libwebp
        let (centers, alpha_to_segment, mid_alpha) =
            assign_segments_kmeans(&analysis.alpha_histogram, usize::from(self.num_segments));

        // Find min and max of centers for alpha transformation
        // This matches libwebp's SetSegmentAlphas
        let min_center = centers.iter().copied().min().unwrap_or(0) as i32;
        let max_center = centers.iter().copied().max().unwrap_or(255) as i32;
        let range = if max_center == min_center {
            1 // Avoid division by zero
        } else {
            max_center - min_center
        };

        // libwebp's `SetSegmentAlphas` (`analysis_enc.c:92`) only handles the
        // degenerate `min == max` case (`if (max == min) max = min + 1;`) — no
        // additional floor. The previous `MIN_ALPHA_RANGE = 64` floor in
        // zenwebp dampened the SNS modulation by up to 6.4× on flat content
        // (gradients, skies), losing most of the per-segment quantizer spread
        // that makes the larger segments compress better. Removed in #30.
        let effective_range = range; // already >= 1 above

        // Assign segment IDs to macroblocks
        self.segment_map = analysis
            .mb_alphas
            .iter()
            .map(|&alpha| alpha_to_segment[alpha as usize])
            .collect();

        // Smooth segment map (3x3 majority filter) only when the preprocessing
        // smooth_segment_map flag explicitly opts in. libwebp gates this on `config->preprocessing & 1`
        // (`analysis_enc.c:217-218`), default OFF (`config_enc.c:48`); we match.
        // zenwebp previously smoothed unconditionally whenever multi-segment, which
        // could collapse 4 segments into 2 after `simplify_segments` and lose
        // differential-quantization savings (#26).
        if self.num_segments > 1 && self.smooth_segment_map {
            super::cost::smooth_segment_map(
                &mut self.segment_map,
                usize::from(self.macroblock_width),
                usize::from(self.macroblock_height),
            );
        }

        // Configure per-segment quantization using preset's SNS strength
        let sns_strength = self.sns_strength;

        // Compute UV quant deltas from uv_alpha average (from libwebp's VP8SetSegmentParams)
        // uv_alpha is typically ~30 (bad) to ~100 (ok to decimate UV more), centered ~60
        // Constants from libwebp quant_enc.c
        const MID_UV_ALPHA: i32 = 64;
        const MIN_UV_ALPHA: i32 = 30;
        const MAX_UV_ALPHA: i32 = 100;
        const MAX_DQ_UV: i32 = 6;
        const MIN_DQ_UV: i32 = -4;

        // Map uv_alpha to the safe maximal range of MAX/MIN_DQ_UV
        let dq_uv_ac = (analysis.uv_alpha_avg - MID_UV_ALPHA) * (MAX_DQ_UV - MIN_DQ_UV)
            / (MAX_UV_ALPHA - MIN_UV_ALPHA);
        // Rescale by user-defined SNS strength
        let dq_uv_ac = (dq_uv_ac * i32::from(sns_strength) / 100).clamp(MIN_DQ_UV, MAX_DQ_UV);

        // Boost dc-uv-quant based on sns-strength (UV is more reactive to high quants)
        let dq_uv_dc = (-4 * i32::from(sns_strength) / 100).clamp(-15, 15);

        // Write UV quant deltas to the bitstream header so the decoder knows about them.
        // Without this, the encoder quantizes UV with offset step sizes but the decoder
        // dequantizes with base step sizes, causing systematic UV reconstruction errors.
        if dq_uv_dc != 0 {
            self.quantization_indices.uvdc_delta = Some(dq_uv_dc as i8);
        }
        if dq_uv_ac != 0 {
            self.quantization_indices.uvac_delta = Some(dq_uv_ac as i8);
        }

        // libwebp uses segment 0's modulated quantizer + filter as the bitstream
        // base (`enc->base_quant = enc->dqm[0].quant;` at `quant_enc.c:404`), so
        // segment 0's delta is always 0 (saves ~9 bits per non-zero delta in the
        // segment header). zenwebp previously used the unmodulated
        // `quality_to_quant_index(quality)`, making segment 0 always carry a
        // non-zero delta when SNS modulation was active. (#30 C)
        //
        // Two passes: pass 1 computes the per-segment (quant, filter) tuples;
        // pass 2 picks segment 0's values as the bitstream base and writes
        // deltas relative to that.
        let mut seg_quant_indices = [0u8; 4];
        let mut seg_filters = [0u8; 4];
        for (seg_idx, &center) in centers.iter().enumerate() {
            let center = center as i32;
            let transformed_alpha = (255 * (center - mid_alpha) / effective_range).clamp(-127, 127);
            let beta = (255 * (center - min_center) / effective_range).clamp(0, 255) as u8;
            let seg_quant_index = compute_segment_quant(quality, transformed_alpha, sns_strength);
            let seg_filter = super::cost::compute_filter_level_with_beta(
                seg_quant_index,
                self.filter_sharpness,
                self.filter_strength,
                beta,
            );
            seg_quant_indices[seg_idx] = seg_quant_index;
            seg_filters[seg_idx] = seg_filter;
        }

        let base_quant_index_new = seg_quant_indices[0];
        let base_filter_new = seg_filters[0];
        self.quantization_indices.yac_abs = base_quant_index_new;

        // Suppress unused-variable warning on the legacy unmodulated `base_quant_index`
        // — kept in the function signature for analysis-pass API stability.
        let _ = base_quant_index;

        for (seg_idx, &center) in centers.iter().enumerate() {
            let _ = center;
            let seg_quant_index = seg_quant_indices[seg_idx];
            let seg_quant_usize = seg_quant_index as usize;

            // Compute the delta from segment 0's modulated quant.
            let delta = seg_quant_index as i8 - base_quant_index_new as i8;

            // Use the per-segment filter level computed in pass 1; delta is
            // relative to segment 0's filter (the new bitstream base).
            let seg_filter = seg_filters[seg_idx];
            let filter_delta = (seg_filter as i8) - (base_filter_new as i8);

            // Apply UV quant deltas (from libwebp's SetupMatrices)
            // UV DC quant uses dq_uv_dc offset, clamped to [0, 117]
            // UV AC quant uses dq_uv_ac offset, clamped to [0, 127]
            let uv_dc_idx = (seg_quant_usize as i32 + dq_uv_dc).clamp(0, 117) as usize;
            let uv_ac_idx = (seg_quant_usize as i32 + dq_uv_ac).clamp(0, 127) as usize;

            let mut segment = Segment {
                ydc: DC_QUANT[seg_quant_usize],
                yac: AC_QUANT[seg_quant_usize],
                y2dc: DC_QUANT[seg_quant_usize] * 2,
                // Y2 AC uses libwebp's dedicated `kAcTable2` lookup (`quant_enc.c:236`,
                // verified byte-identical to our `VP8_AC_TABLE2`). Previously we
                // synthesized the value as `kAcTable * 155/100` which deviates by up
                // to ~10% at mid-quantizer. Decoder side updated to match (#24).
                y2ac: VP8_AC_TABLE2[seg_quant_usize] as i16,
                uvdc: DC_QUANT[uv_dc_idx],
                uvac: AC_QUANT[uv_ac_idx],
                quantizer_level: delta,
                loopfilter_level: filter_delta,
                quant_index: seg_quant_index,
                ..Default::default()
            };
            segment.init_matrices(self.sns_strength, self.method, self.cost_model);
            self.segments[seg_idx] = segment;
        }

        // Simplify segments by merging those with identical quant and filter settings
        // This can reduce the number of effective segments and save bits in the bitstream
        if self.num_segments > 1 {
            self.simplify_segments();
        }

        // Compute segment tree probabilities from actual distribution
        // This matches libwebp's SetSegmentProbas
        let mut seg_counts = [0u32; 4];
        for &seg_id in &self.segment_map {
            seg_counts[seg_id as usize] += 1;
        }

        // Segment tree uses 3 probabilities for binary splits:
        // prob[0] = P(segment < 2), prob[1] = P(segment == 0 | segment < 2)
        // prob[2] = P(segment == 2 | segment >= 2)
        #[allow(clippy::manual_checked_ops)]
        let get_proba = |a: u32, b: u32| -> u8 {
            let total = a + b;
            if total == 0 {
                255 // default
            } else {
                ((255 * a + total / 2) / total) as u8
            }
        };

        self.segment_tree_probs[0] =
            get_proba(seg_counts[0] + seg_counts[1], seg_counts[2] + seg_counts[3]);
        self.segment_tree_probs[1] = get_proba(seg_counts[0], seg_counts[1]);
        self.segment_tree_probs[2] = get_proba(seg_counts[2], seg_counts[3]);

        // Only enable update_map if probabilities differ from default (255)
        let should_update_map = self.segment_tree_probs[0] != 255
            || self.segment_tree_probs[1] != 255
            || self.segment_tree_probs[2] != 255;

        // Enable segment-based encoding
        self.segments_enabled = true;
        self.segments_update_map = should_update_map;

        // Reset borders for actual encoding pass
        self.left_border_y = [129u8; 16 + 1];
        self.left_border_u = [129u8; 8 + 1];
        self.left_border_v = [129u8; 8 + 1];
    }

    // sets up the encoding of the encoder by setting all the encoder params based on the width and height
    fn setup_encoding(
        &mut self,
        lossy_quality: u8,
        width: u16,
        height: u16,
        y_buf: Vec<u8>,
        u_buf: Vec<u8>,
        v_buf: Vec<u8>,
    ) {
        // choosing the quantization quality based on the quality passed in
        if lossy_quality > 100 {
            panic!("lossy quality must be between 0 and 100");
        }

        // Use libwebp-style quality curve to match expected behavior at Q75
        // This emulates jpeg-like behavior where Q75 is "good quality"
        let quant_index: u8 = quality_to_quant_index(lossy_quality);
        let quant_index_usize: usize = quant_index as usize;

        let mb_width = width.div_ceil(16);
        let mb_height = height.div_ceil(16);
        self.macroblock_width = mb_width;
        self.macroblock_height = mb_height;

        // Compute optimal filter level based on quantization and preset tuning params
        let filter_level = super::cost::compute_filter_level(
            quant_index,
            self.filter_sharpness,
            self.filter_strength,
        );

        self.frame = Frame {
            width,
            height,

            ybuf: y_buf,
            ubuf: u_buf,
            vbuf: v_buf,

            version: 0,

            for_display: true,
            pixel_type: 0,

            filter_type: false,
            filter_level,
            sharpness_level: self.filter_sharpness,
        };

        self.top_complexity = vec![Complexity::default(); usize::from(mb_width)];
        self.top_b_pred = vec![IntraMode::default(); 4 * usize::from(mb_width)];
        self.left_b_pred = [IntraMode::default(); 4];

        self.token_probs = COEFF_PROBS;

        // Enable skip mode for zero macroblocks
        // The probability is P(not skip) - 200 means ~78% expected to have coefficients
        self.macroblock_no_skip_coeff = Some(200);

        let quantization_indices = QuantizationIndices {
            yac_abs: quant_index,
            ..Default::default()
        };
        self.quantization_indices = quantization_indices;

        // Initialize all 4 segments with base quantization first
        // This provides fallback values before segment analysis
        for seg_idx in 0..4 {
            let mut segment = Segment {
                ydc: DC_QUANT[quant_index_usize],
                yac: AC_QUANT[quant_index_usize],
                y2dc: DC_QUANT[quant_index_usize] * 2,
                y2ac: VP8_AC_TABLE2[quant_index_usize] as i16,
                uvdc: DC_QUANT[quant_index_usize],
                uvac: AC_QUANT[quant_index_usize],
                quantizer_level: 0, // No delta for base segment
                quant_index,
                ..Default::default()
            };
            segment.init_matrices(self.sns_strength, self.method, self.cost_model);
            self.segments[seg_idx] = segment;
        }

        // Segment-based quantization using DCT histogram analysis (ported from libwebp).
        // This allows different quantization for different image regions:
        // - Flat areas get more aggressive quantization
        // - Textured areas get finer quantization
        //
        // Only enable for images large enough to benefit (overhead vs gain tradeoff).
        // libwebp uses segments for images with method > 0 and multiple segments configured.
        let _total_mbs = usize::from(mb_width) * usize::from(mb_height);
        // libwebp gates segmentation on `config->emulate_jpeg_size || num_segments > 1
        // || method <= 1` (`analysis_enc.c:434-436`) — no min-MB threshold. zenwebp
        // previously skipped segmentation entirely below 256 MBs (~256x256 images),
        // losing alpha-driven quantizer differentiation on icons/thumbnails. Removed
        // the `>= 256` gate in #30.
        let use_segments = self.num_segments > 1;

        if use_segments {
            // DCT-based segment analysis and assignment.
            // For Preset::Auto, this also runs content detection and may override
            // sns_strength, filter_strength, filter_sharpness, and num_segments.
            self.analyze_and_assign_segments(quant_index, lossy_quality);

            // If Auto detection changed filter params, recompute frame filter level
            if self.preset == super::api::Preset::Auto {
                let new_filter = super::cost::compute_filter_level(
                    quant_index,
                    self.filter_sharpness,
                    self.filter_strength,
                );
                self.frame.filter_level = new_filter;
                self.frame.sharpness_level = self.filter_sharpness;
            }
        } else {
            // Disable segments for small images (overhead not worth it)
            self.segments_enabled = false;
            self.segments_update_map = false;
            self.segment_map = Vec::new();
        }

        self.left_border_y = [129u8; 16 + 1];
        self.left_border_u = [129u8; 8 + 1];
        self.left_border_v = [129u8; 8 + 1];

        self.top_border_y = vec![127u8; usize::from(self.macroblock_width) * 16 + 4];
        self.top_border_u = vec![127u8; usize::from(self.macroblock_width) * 8];
        self.top_border_v = vec![127u8; usize::from(self.macroblock_width) * 8];

        // Initialize error diffusion arrays (one entry per macroblock column)
        // [channel][position], channels are U=0, V=1
        self.top_derr = vec![[[0i8; 2]; 2]; usize::from(self.macroblock_width)];
        self.left_derr = [[0; 2]; 2];
    }
}

/// Convert SSE to PSNR in dB. Returns 99.0 for perfect reconstruction (SSE=0).
fn sse_to_psnr(sse: u64, num_pixels: u64) -> f32 {
    if sse == 0 || num_pixels == 0 {
        99.0
    } else {
        let mse = sse as f64 / num_pixels as f64;
        (10.0 * libm::log10(255.0 * 255.0 / mse)) as f32
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_frame_lossy(
    writer: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    color: PixelLayout,
    params: &super::api::EncoderParams,
    stop: &dyn enough::Stop,
    progress: &dyn super::api::EncodeProgress,
) -> super::api::EncodeResult<super::api::EncodeStats> {
    let width = width
        .try_into()
        .map_err(|_| at!(EncodeError::InvalidDimensions))?;
    let height = height
        .try_into()
        .map_err(|_| at!(EncodeError::InvalidDimensions))?;

    // Quality search: if target_size or target_psnr is set, iterate quality to converge
    if params.target_size > 0 {
        Ok(encode_with_quality_search(
            writer, data, width, height, stride, color, params, stop, progress,
        )?)
    } else if params.target_psnr > 0.0 {
        Ok(encode_with_psnr_search(
            writer, data, width, height, stride, color, params, stop, progress,
        )?)
    } else {
        // Single encoding at specified quality, with automatic partition limit retry
        encode_with_partition_retry(
            writer, data, width, height, stride, color, params, stop, progress,
        )
    }
}

/// Encode a single frame, automatically retrying with increasing partition_limit
/// if partition 0 overflows and the user didn't set an explicit limit.
#[allow(clippy::too_many_arguments)]
fn encode_with_partition_retry(
    writer: &mut Vec<u8>,
    data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
    color: PixelLayout,
    params: &super::api::EncoderParams,
    stop: &dyn enough::Stop,
    progress: &dyn super::api::EncodeProgress,
) -> super::api::EncodeResult<super::api::EncodeStats> {
    // If the user set an explicit partition_limit, use it as-is (no retry).
    if params.partition_limit.is_some() {
        let mut vp8_encoder = Vp8Encoder::new(writer);
        return Ok(
            vp8_encoder.encode_image(data, color, width, height, stride, params, stop, progress)?
        );
    }

    // Automatic mode: try encoding, retry with increasing partition_limit on overflow.
    // Escalation steps chosen to quickly find a working limit without too many retries.
    const RETRY_LIMITS: [u8; 4] = [0, 40, 70, 100];

    let mut last_overflow = None;
    for &limit in &RETRY_LIMITS {
        stop.check().map_err(|e| at!(EncodeError::from(e)))?;

        let mut trial_buf = Vec::new();
        let mut trial_params = params.clone();
        trial_params.partition_limit = Some(limit);

        let mut vp8_encoder = Vp8Encoder::new(&mut trial_buf);
        match vp8_encoder.encode_image(
            data,
            color,
            width,
            height,
            stride,
            &trial_params,
            stop,
            progress,
        ) {
            Ok(stats) => {
                writer.extend_from_slice(&trial_buf);
                return Ok(stats);
            }
            Err(e @ EncodeError::Partition0Overflow { .. }) if limit < 100 => {
                last_overflow = Some(e);
                continue;
            }
            Err(e) => return Err(at!(e)),
        }
    }

    // All retry limits exhausted — return the last overflow error
    Err(at!(last_overflow.unwrap_or(
        EncodeError::Partition0Overflow {
            size: 0,
            max: (1 << 19) - 1,
        }
    )))
}

/// Encode with quality search to meet target file size.
/// Uses secant method to converge on target size within DQ_LIMIT threshold.
#[allow(clippy::too_many_arguments)]
fn encode_with_quality_search(
    writer: &mut Vec<u8>,
    data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
    color: PixelLayout,
    params: &super::api::EncoderParams,
    stop: &dyn enough::Stop,
    progress: &dyn super::api::EncodeProgress,
) -> Result<super::api::EncodeStats, EncodeError> {
    // Initialize quality search state
    // qmin=1, qmax=100 (full range) - libwebp uses config->qmin/qmax
    let mut pass_stats = PassStats::new_for_size(params.target_size, params.lossy_quality, 1, 100);

    // Max iterations (matches libwebp's config->pass, default 6 for target_size search)
    let max_passes = (params.method + 3).max(6) as usize;
    let mut best_output: Option<Vec<u8>> = None;
    let mut best_enc_stats = super::api::EncodeStats::default();
    let mut best_diff = f64::MAX;

    for pass in 0..max_passes {
        stop.check()?;

        // Create temporary buffer for this trial encoding
        let mut trial_buffer = Vec::new();
        let mut trial_encoder = Vp8Encoder::new(&mut trial_buffer);

        // Create params with adjusted quality
        let mut trial_params = params.clone();
        trial_params.lossy_quality = libm::roundf(pass_stats.q).clamp(0.0, 100.0) as u8;

        // Encode to trial buffer
        let enc_stats = trial_encoder.encode_image(
            data,
            color,
            width,
            height,
            stride,
            &trial_params,
            stop,
            progress,
        )?;

        // Update stats with resulting size
        let output_size = trial_buffer.len() as f64;
        pass_stats.value = output_size;

        // Track best result (closest to target)
        let diff = (output_size - pass_stats.target).abs();
        if diff < best_diff {
            best_diff = diff;
            best_enc_stats = enc_stats;
            best_output = Some(trial_buffer);
        }

        // Check convergence
        let is_last = pass + 1 >= max_passes || pass_stats.is_converged();
        if is_last {
            break;
        }

        // Compute next quality to try
        pass_stats.compute_next_q();
    }

    // Write best result to output
    if let Some(output) = best_output {
        writer.extend_from_slice(&output);
    }

    Ok(best_enc_stats)
}

/// Encode with quality search to meet target PSNR.
/// Uses secant method to converge on target PSNR within DQ_LIMIT threshold.
#[allow(clippy::too_many_arguments)]
fn encode_with_psnr_search(
    writer: &mut Vec<u8>,
    data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
    color: PixelLayout,
    params: &super::api::EncoderParams,
    stop: &dyn enough::Stop,
    progress: &dyn super::api::EncodeProgress,
) -> Result<super::api::EncodeStats, EncodeError> {
    let mut pass_stats = PassStats::new_for_psnr(params.target_psnr, params.lossy_quality, 1, 100);

    let max_passes = (params.method + 3).max(6) as usize;
    let mut best_output: Option<Vec<u8>> = None;
    let mut best_enc_stats = super::api::EncodeStats::default();
    let mut best_diff = f64::MAX;

    for pass in 0..max_passes {
        stop.check()?;

        let mut trial_buffer = Vec::new();
        let mut trial_encoder = Vp8Encoder::new(&mut trial_buffer);

        let mut trial_params = params.clone();
        trial_params.lossy_quality = libm::roundf(pass_stats.q).clamp(0.0, 100.0) as u8;

        let enc_stats = trial_encoder.encode_image(
            data,
            color,
            width,
            height,
            stride,
            &trial_params,
            stop,
            progress,
        )?;

        // Use "All" PSNR (index 3) as the convergence metric
        let psnr_value = f64::from(enc_stats.psnr[3]);
        pass_stats.value = psnr_value;

        // Track best result (closest PSNR to target)
        let diff = (psnr_value - pass_stats.target).abs();
        if diff < best_diff {
            best_diff = diff;
            best_enc_stats = enc_stats;
            best_output = Some(trial_buffer);
        }

        let is_last = pass + 1 >= max_passes || pass_stats.is_converged();
        if is_last {
            break;
        }

        pass_stats.compute_next_q();
    }

    if let Some(output) = best_output {
        writer.extend_from_slice(&output);
    }

    Ok(best_enc_stats)
}
