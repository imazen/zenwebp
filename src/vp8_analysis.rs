//! Segment-based quantization analysis for VP8 encoding.
//!
//! This module provides macroblock-level analysis for adaptive quantization,
//! computing complexity metrics and assigning segments using k-means clustering.
//!
//! Ported from libwebp src/enc/analysis_enc.c

// Allow unused items - these are part of the complete libwebp analysis system
#![allow(dead_code)]
// Many loops in this file match libwebp's C patterns for clarity when comparing
#![allow(clippy::needless_range_loop)]

extern crate alloc;
use alloc::vec::Vec;

//------------------------------------------------------------------------------
// Segment-based quantization
//
// Ported from libwebp src/enc/analysis_enc.c
// Allows different quantization levels for different image regions.
//
// The analysis pass computes "alpha" (compressibility) for each macroblock
// using DCT histogram analysis, then uses k-means clustering to assign
// segments for per-region quantization.

/// Maximum alpha value for macroblock complexity
pub const MAX_ALPHA: u8 = 255;

/// Alpha scale factor (from libwebp: ALPHA_SCALE = 2 * MAX_ALPHA = 510)
const ALPHA_SCALE: u32 = 2 * MAX_ALPHA as u32;

/// Maximum coefficient threshold for histogram binning (from libwebp)
const MAX_COEFF_THRESH: usize = 31;

/// Number of k-means iterations for segment assignment
const MAX_ITERS_K_MEANS: usize = 6;

/// Number of segments
pub const NUM_SEGMENTS: usize = 4;

/// Number of intra16 modes to test during analysis (DC=0, TM=1)
const MAX_INTRA16_MODE: usize = 2;

/// Number of UV modes to test during analysis (DC=0, TM=1)
const MAX_UV_MODE: usize = 2;

/// Bytes per stride in libwebp's work buffers
const BPS: usize = 32;

/// Offset to Y plane in work buffer
const Y_OFF_ENC: usize = 0;

/// Offset to U plane in work buffer
const U_OFF_ENC: usize = 16;

/// Offset to V plane in work buffer
const V_OFF_ENC: usize = 24;

/// Offsets to each 4x4 block within a BPS-strided buffer
/// Ported from libwebp's VP8DspScan
/// The 0 * BPS expressions are intentional to match libwebp's pattern
#[allow(clippy::erasing_op, clippy::identity_op)]
const VP8_DSP_SCAN: [usize; 24] = [
    // Luma (16 blocks)
    0 + 0 * BPS,
    4 + 0 * BPS,
    8 + 0 * BPS,
    12 + 0 * BPS,
    0 + 4 * BPS,
    4 + 4 * BPS,
    8 + 4 * BPS,
    12 + 4 * BPS,
    0 + 8 * BPS,
    4 + 8 * BPS,
    8 + 8 * BPS,
    12 + 8 * BPS,
    0 + 12 * BPS,
    4 + 12 * BPS,
    8 + 12 * BPS,
    12 + 12 * BPS,
    // U (4 blocks)
    0 + 0 * BPS,
    4 + 0 * BPS,
    0 + 4 * BPS,
    4 + 4 * BPS,
    // V (4 blocks, offset by 8 from U)
    8 + 0 * BPS,
    12 + 0 * BPS,
    8 + 4 * BPS,
    12 + 4 * BPS,
];

/// Mode offsets for I16 predictions in yuv_p buffer
/// Ported from libwebp's VP8I16ModeOffsets
const I16DC16: usize = 0;
const I16TM16: usize = I16DC16 + 16;
#[allow(clippy::identity_op)] // 1 * 16 matches libwebp's pattern (row * block_size * stride)
const I16VE16: usize = 1 * 16 * BPS;
const I16HE16: usize = I16VE16 + 16;

const VP8_I16_MODE_OFFSETS: [usize; 4] = [I16DC16, I16TM16, I16VE16, I16HE16];

/// Mode offsets for chroma predictions in yuv_p buffer
/// Ported from libwebp's VP8UVModeOffsets
const C8DC8: usize = 2 * 16 * BPS;
const C8TM8: usize = C8DC8 + 16;
const C8VE8: usize = 2 * 16 * BPS + 8 * BPS;
const C8HE8: usize = C8VE8 + 16;

const VP8_UV_MODE_OFFSETS: [usize; 4] = [C8DC8, C8TM8, C8VE8, C8HE8];

/// Size of prediction buffer (I16 + Chroma + I4 preds)
const PRED_SIZE_ENC: usize = 32 * BPS + 16 * BPS + 8 * BPS;

/// Size of YUV work buffer
const YUV_SIZE_ENC: usize = BPS * 16;

/// DCT histogram result for alpha calculation
/// Ported from libwebp's VP8Histogram struct
#[derive(Default, Clone, Copy)]
pub struct DctHistogram {
    /// Maximum count in any histogram bin
    pub max_value: u32,
    /// Highest bin index with non-zero count
    pub last_non_zero: usize,
}

impl DctHistogram {
    /// Initialize histogram with default values (from libwebp's InitHistogram)
    pub fn new() -> Self {
        Self {
            max_value: 0,
            last_non_zero: 1,
        }
    }

    /// Compute histogram data from a distribution array
    /// Ported from libwebp's VP8SetHistogramData
    pub fn from_distribution(distribution: &[u32; MAX_COEFF_THRESH + 1]) -> Self {
        let mut max_value: u32 = 0;
        let mut last_non_zero: usize = 1;

        for (k, &count) in distribution.iter().enumerate() {
            if count > 0 {
                if count > max_value {
                    max_value = count;
                }
                last_non_zero = k;
            }
        }

        Self {
            max_value,
            last_non_zero,
        }
    }

    /// Get alpha value from histogram
    /// Ported from libwebp's GetAlpha
    pub fn get_alpha(&self) -> i32 {
        if self.max_value > 1 {
            (ALPHA_SCALE * self.last_non_zero as u32 / self.max_value) as i32
        } else {
            0
        }
    }
}

/// Forward DCT on a 4x4 block (difference between source and prediction)
/// Ported from libwebp's VP8FTransform (FTransform_C)
#[allow(clippy::identity_op)] // 0 + offset pattern matches libwebp's code structure
fn forward_dct_4x4(src: &[u8], pred: &[u8], src_stride: usize, pred_stride: usize) -> [i16; 16] {
    let mut tmp = [0i32; 16];
    let mut out = [0i16; 16];

    // Horizontal pass: compute differences and first butterfly
    for i in 0..4 {
        let src_row = i * src_stride;
        let pred_row = i * pred_stride;

        let d0 = i32::from(src[src_row]) - i32::from(pred[pred_row]);
        let d1 = i32::from(src[src_row + 1]) - i32::from(pred[pred_row + 1]);
        let d2 = i32::from(src[src_row + 2]) - i32::from(pred[pred_row + 2]);
        let d3 = i32::from(src[src_row + 3]) - i32::from(pred[pred_row + 3]);

        let a0 = d0 + d3;
        let a1 = d1 + d2;
        let a2 = d1 - d2;
        let a3 = d0 - d3;

        tmp[0 + i * 4] = (a0 + a1) * 8;
        tmp[2 + i * 4] = (a0 - a1) * 8;
        tmp[1 + i * 4] = (a2 * 2217 + a3 * 5352 + 1812) >> 9;
        tmp[3 + i * 4] = (a3 * 2217 - a2 * 5352 + 937) >> 9;
    }

    // Vertical pass
    for i in 0..4 {
        let a0 = tmp[0 + i] + tmp[12 + i];
        let a1 = tmp[4 + i] + tmp[8 + i];
        let a2 = tmp[4 + i] - tmp[8 + i];
        let a3 = tmp[0 + i] - tmp[12 + i];

        out[0 + i] = ((a0 + a1 + 7) >> 4) as i16;
        out[8 + i] = ((a0 - a1 + 7) >> 4) as i16;
        out[4 + i] = (((a2 * 2217 + a3 * 5352 + 12000) >> 16) + if a3 != 0 { 1 } else { 0 }) as i16;
        out[12 + i] = ((a3 * 2217 - a2 * 5352 + 51000) >> 16) as i16;
    }

    out
}

/// Collect DCT histogram for a range of 4x4 blocks
/// Ported from libwebp's CollectHistogram_C / VP8CollectHistogram
///
/// # Arguments
/// * `src` - Source pixels (BPS stride)
/// * `pred` - Prediction pixels (BPS stride)
/// * `start_block` - First block index (in VP8DspScan)
/// * `end_block` - One past last block index
pub fn collect_histogram_bps(
    src: &[u8],
    pred: &[u8],
    start_block: usize,
    end_block: usize,
) -> DctHistogram {
    let mut distribution = [0u32; MAX_COEFF_THRESH + 1];

    for &scan_off in &VP8_DSP_SCAN[start_block..end_block] {
        let src_off = scan_off;
        let pred_off = scan_off;

        let dct_out = forward_dct_4x4(&src[src_off..], &pred[pred_off..], BPS, BPS);

        for coeff in dct_out.iter() {
            let v = (coeff.unsigned_abs() >> 3) as usize;
            let clipped = v.min(MAX_COEFF_THRESH);
            distribution[clipped] += 1;
        }
    }

    DctHistogram::from_distribution(&distribution)
}

/// Finalize alpha value by inverting and clipping
/// Ported from libwebp's FinalAlphaValue
#[inline]
fn final_alpha_value(alpha: i32) -> u8 {
    let alpha = MAX_ALPHA as i32 - alpha;
    alpha.clamp(0, MAX_ALPHA as i32) as u8
}

//------------------------------------------------------------------------------
// Prediction functions for analysis
// Ported from libwebp's VP8EncPredLuma16 and VP8EncPredChroma8

/// Generate DC prediction for 16x16 luma block into BPS-strided buffer
/// Ported from libwebp's DCMode (size=16, round=16, shift=5)
fn pred_luma16_dc(dst: &mut [u8], left: Option<&[u8]>, top: Option<&[u8]>) {
    let dc_val = match (top, left) {
        (Some(top), Some(left)) => {
            // Both borders: sum all 32 samples
            let dc: u32 = top[..16].iter().map(|&x| u32::from(x)).sum::<u32>()
                + left[..16].iter().map(|&x| u32::from(x)).sum::<u32>();
            ((dc + 16) >> 5) as u8
        }
        (Some(top), None) => {
            // Top only: sum 16 samples, double, then shift by 5
            let mut dc: u32 = top[..16].iter().map(|&x| u32::from(x)).sum();
            dc += dc; // Double
            ((dc + 16) >> 5) as u8
        }
        (None, Some(left)) => {
            // Left only: sum 16 samples, double, then shift by 5
            let mut dc: u32 = left[..16].iter().map(|&x| u32::from(x)).sum();
            dc += dc; // Double
            ((dc + 16) >> 5) as u8
        }
        (None, None) => {
            // No borders: use 128
            0x80u8
        }
    };

    // Fill 16x16 block
    for y in 0..16 {
        for x in 0..16 {
            dst[y * BPS + x] = dc_val;
        }
    }
}

/// Fill a block with a constant value (BPS stride)
fn fill_block(dst: &mut [u8], value: u8, size: usize) {
    for y in 0..size {
        for x in 0..size {
            dst[y * BPS + x] = value;
        }
    }
}

/// Vertical prediction: copy top row to all rows
fn vertical_pred(dst: &mut [u8], top: Option<&[u8]>, size: usize) {
    if let Some(top) = top {
        for y in 0..size {
            for x in 0..size {
                dst[y * BPS + x] = top[x];
            }
        }
    } else {
        fill_block(dst, 127, size);
    }
}

/// Horizontal prediction: copy left column to all columns
fn horizontal_pred(dst: &mut [u8], left: Option<&[u8]>, size: usize) {
    if let Some(left) = left {
        for y in 0..size {
            for x in 0..size {
                dst[y * BPS + x] = left[y];
            }
        }
    } else {
        fill_block(dst, 129, size);
    }
}

/// Generate TM (TrueMotion) prediction for 16x16 luma block
/// Ported from libwebp's TrueMotion
///
/// left_with_corner[0] = top-left corner, left_with_corner[1..17] = left pixels
fn pred_luma16_tm(dst: &mut [u8], left_with_corner: Option<&[u8]>, top: Option<&[u8]>) {
    match (left_with_corner, top) {
        (Some(left), Some(top)) => {
            // Both borders: compute TrueMotion
            // left[0] is top-left corner, left[1..17] are left pixels
            let tl = i32::from(left[0]);
            for y in 0..16 {
                let l = i32::from(left[1 + y]);
                for x in 0..16 {
                    let t = i32::from(top[x]);
                    dst[y * BPS + x] = (l + t - tl).clamp(0, 255) as u8;
                }
            }
        }
        (Some(left), None) => {
            // Left only: use horizontal prediction
            horizontal_pred(dst, Some(&left[1..17]), 16);
        }
        (None, Some(top)) => {
            // Top only: use vertical prediction
            vertical_pred(dst, Some(top), 16);
        }
        (None, None) => {
            // Neither: fill with 129 (not 127 like VerticalPred default)
            fill_block(dst, 129, 16);
        }
    }
}

/// Generate all I16 predictions into yuv_p buffer
/// Ported from libwebp's VP8EncPredLuma16
///
/// left_with_corner: full y_left array where [0]=corner, [1..17]=left pixels
fn make_luma16_preds(yuv_p: &mut [u8], left_with_corner: Option<&[u8]>, top: Option<&[u8]>) {
    // DC prediction at I16DC16 (only needs left pixels, not corner)
    let left_only = left_with_corner.map(|l| &l[1..17]);
    pred_luma16_dc(&mut yuv_p[I16DC16..], left_only, top);

    // TM prediction at I16TM16 (needs corner at [0] and left pixels at [1..17])
    pred_luma16_tm(&mut yuv_p[I16TM16..], left_with_corner, top);

    // Note: V and H predictions not implemented since we only test DC and TM (MAX_INTRA16_MODE=2)
}

/// Generate DC prediction for 8x8 chroma block into BPS-strided buffer
/// Ported from libwebp's DCMode (size=8, round=8, shift=4)
fn pred_chroma8_dc(dst: &mut [u8], left: Option<&[u8]>, top: Option<&[u8]>) {
    let dc_val = match (top, left) {
        (Some(top), Some(left)) => {
            // Both borders: sum all 16 samples
            let mut dc: u32 = 0;
            for i in 0..8 {
                dc += u32::from(top[i]);
                dc += u32::from(left[i]);
            }
            ((dc + 8) >> 4) as u8
        }
        (Some(top), None) => {
            // Top only: sum 8 samples, double, then shift by 4
            let mut dc: u32 = 0;
            for i in 0..8 {
                dc += u32::from(top[i]);
            }
            dc += dc; // Double
            ((dc + 8) >> 4) as u8
        }
        (None, Some(left)) => {
            // Left only: sum 8 samples, double, then shift by 4
            let mut dc: u32 = 0;
            for i in 0..8 {
                dc += u32::from(left[i]);
            }
            dc += dc; // Double
            ((dc + 8) >> 4) as u8
        }
        (None, None) => {
            // No borders: use 128
            0x80u8
        }
    };

    // Fill 8x8 block
    for y in 0..8 {
        for x in 0..8 {
            dst[y * BPS + x] = dc_val;
        }
    }
}

/// Generate TM prediction for 8x8 chroma block
/// Ported from libwebp's TrueMotion
///
/// left_with_corner: [0]=corner, [1..9]=left pixels (matching u_left/v_left layout)
fn pred_chroma8_tm(dst: &mut [u8], left_with_corner: Option<&[u8]>, top: Option<&[u8]>) {
    match (left_with_corner, top) {
        (Some(left), Some(top)) => {
            // Both borders: compute TrueMotion
            // left[0] is top-left corner, left[1..9] are left pixels
            let tl = i32::from(left[0]);
            for y in 0..8 {
                let l = i32::from(left[1 + y]);
                for x in 0..8 {
                    let t = i32::from(top[x]);
                    dst[y * BPS + x] = (l + t - tl).clamp(0, 255) as u8;
                }
            }
        }
        (Some(left), None) => {
            // Left only: use horizontal prediction
            horizontal_pred(dst, Some(&left[1..9]), 8);
        }
        (None, Some(top)) => {
            // Top only: use vertical prediction
            vertical_pred(dst, Some(top), 8);
        }
        (None, None) => {
            // Neither: fill with 129
            fill_block(dst, 129, 8);
        }
    }
}

/// Generate all chroma predictions into yuv_p buffer
/// Ported from libwebp's VP8EncPredChroma8
///
/// Note: In libwebp, U and V predictions are interleaved in the buffer.
/// u_left_with_corner/v_left_with_corner: [0]=corner, [1..9]=left pixels
/// uv_top: U at [0..8], V at [8..16]
fn make_chroma8_preds(
    yuv_p: &mut [u8],
    u_left_with_corner: Option<&[u8]>,
    v_left_with_corner: Option<&[u8]>,
    uv_top: Option<&[u8]>,
) {
    // Extract U and V top borders if available
    let (u_top, v_top) = if let Some(uv_top) = uv_top {
        (Some(&uv_top[0..8]), Some(&uv_top[8..16]))
    } else {
        (None, None)
    };

    // For DC prediction, extract just the left pixels (no corner)
    let u_left_only = u_left_with_corner.map(|l| &l[1..9]);
    let v_left_only = v_left_with_corner.map(|l| &l[1..9]);

    // DC prediction for U at C8DC8
    pred_chroma8_dc(&mut yuv_p[C8DC8..], u_left_only, u_top);

    // DC prediction for V at C8DC8 + 8 (V is 8 pixels to the right of U)
    pred_chroma8_dc(&mut yuv_p[C8DC8 + 8..], v_left_only, v_top);

    // TM prediction for U at C8TM8 (needs corner)
    pred_chroma8_tm(&mut yuv_p[C8TM8..], u_left_with_corner, u_top);

    // TM prediction for V at C8TM8 + 8
    pred_chroma8_tm(&mut yuv_p[C8TM8 + 8..], v_left_with_corner, v_top);
}

//------------------------------------------------------------------------------
// Macroblock analysis
// Ported from libwebp's analysis_enc.c

/// Import a block of pixels into BPS-strided work buffer
/// Ported from libwebp's ImportBlock
fn import_block(src: &[u8], src_stride: usize, dst: &mut [u8], w: usize, h: usize, size: usize) {
    for y in 0..h {
        // Copy row
        for x in 0..w {
            dst[y * BPS + x] = src[y * src_stride + x];
        }
        // Replicate last pixel if width < size
        if w < size {
            let last_pixel = dst[y * BPS + w - 1];
            for x in w..size {
                dst[y * BPS + x] = last_pixel;
            }
        }
    }
    // Replicate last row if height < size
    for y in h..size {
        for x in 0..size {
            dst[y * BPS + x] = dst[(h - 1) * BPS + x];
        }
    }
}

/// Import vertical line into left border array
fn import_line(src: &[u8], src_stride: usize, dst: &mut [u8], len: usize, total_len: usize) {
    for i in 0..len {
        dst[i] = src[i * src_stride];
    }
    // Replicate last value
    let last = dst[len.saturating_sub(1).max(0)];
    for i in len..total_len {
        dst[i] = last;
    }
}

/// Analysis iterator state
/// Simplified version of libwebp's VP8EncIterator
pub struct AnalysisIterator {
    /// Current macroblock x position
    pub x: usize,
    /// Current macroblock y position
    pub y: usize,
    /// Image width in macroblocks
    pub mb_w: usize,
    /// Image height in macroblocks
    pub mb_h: usize,
    /// Picture width in pixels
    pub width: usize,
    /// Picture height in pixels
    pub height: usize,
    /// YUV input work buffer (BPS stride)
    pub yuv_in: Vec<u8>,
    /// Prediction work buffer (BPS stride)
    pub yuv_p: Vec<u8>,
    /// Left Y boundary samples (17 bytes: [-1] at index 0, [0..15] at indices 1..17)
    pub y_left: [u8; 17],
    /// Left U boundary samples (9 bytes)
    pub u_left: [u8; 9],
    /// Left V boundary samples (9 bytes)
    pub v_left: [u8; 9],
    /// Top Y boundary samples (mb_w * 16 + 4)
    pub y_top: Vec<u8>,
    /// Top UV boundary samples (mb_w * 16, interleaved U[0..8] V[8..16])
    pub uv_top: Vec<u8>,
}

impl AnalysisIterator {
    /// Create a new analysis iterator
    pub fn new(width: usize, height: usize) -> Self {
        let mb_w = width.div_ceil(16);
        let mb_h = height.div_ceil(16);

        Self {
            x: 0,
            y: 0,
            mb_w,
            mb_h,
            width,
            height,
            yuv_in: vec![0u8; YUV_SIZE_ENC],
            yuv_p: vec![0u8; PRED_SIZE_ENC],
            y_left: [129u8; 17],
            u_left: [129u8; 9],
            v_left: [129u8; 9],
            y_top: vec![127u8; mb_w * 16 + 4],
            uv_top: vec![127u8; mb_w * 16],
        }
    }

    /// Reset iterator to start
    pub fn reset(&mut self) {
        self.x = 0;
        self.y = 0;
        self.init_left();
        self.init_top();
    }

    /// Initialize left border to defaults
    fn init_left(&mut self) {
        let corner = if self.y > 0 { 129u8 } else { 127u8 };
        self.y_left[0] = corner;
        self.u_left[0] = corner;
        self.v_left[0] = corner;
        for i in 1..17 {
            self.y_left[i] = 129;
        }
        for i in 1..9 {
            self.u_left[i] = 129;
            self.v_left[i] = 129;
        }
    }

    /// Initialize top border to defaults
    fn init_top(&mut self) {
        self.y_top.fill(127);
        self.uv_top.fill(127);
    }

    /// Set iterator to start of row y
    pub fn set_row(&mut self, y: usize) {
        self.x = 0;
        self.y = y;
        self.init_left();
    }

    /// Import source samples into work buffer
    /// Ported from libwebp's VP8IteratorImport
    pub fn import(
        &mut self,
        y_src: &[u8],
        u_src: &[u8],
        v_src: &[u8],
        y_stride: usize,
        uv_stride: usize,
    ) {
        let x = self.x;
        let y = self.y;

        let y_offset = y * 16 * y_stride + x * 16;
        let uv_offset = y * 8 * uv_stride + x * 8;

        let w = (self.width - x * 16).min(16);
        let h = (self.height - y * 16).min(16);
        let uv_w = w.div_ceil(2);
        let uv_h = h.div_ceil(2);

        // Import Y
        import_block(
            &y_src[y_offset..],
            y_stride,
            &mut self.yuv_in[Y_OFF_ENC..],
            w,
            h,
            16,
        );

        // Import U
        import_block(
            &u_src[uv_offset..],
            uv_stride,
            &mut self.yuv_in[U_OFF_ENC..],
            uv_w,
            uv_h,
            8,
        );

        // Import V
        import_block(
            &v_src[uv_offset..],
            uv_stride,
            &mut self.yuv_in[V_OFF_ENC..],
            uv_w,
            uv_h,
            8,
        );

        // Import boundary samples
        if x == 0 {
            self.init_left();
        } else {
            // Top-left corner
            if y == 0 {
                self.y_left[0] = 127;
                self.u_left[0] = 127;
                self.v_left[0] = 127;
            } else {
                self.y_left[0] = y_src[y_offset - 1 - y_stride];
                self.u_left[0] = u_src[uv_offset - 1 - uv_stride];
                self.v_left[0] = v_src[uv_offset - 1 - uv_stride];
            }

            // Left column
            import_line(
                &y_src[y_offset - 1..],
                y_stride,
                &mut self.y_left[1..],
                h,
                16,
            );
            import_line(
                &u_src[uv_offset - 1..],
                uv_stride,
                &mut self.u_left[1..],
                uv_h,
                8,
            );
            import_line(
                &v_src[uv_offset - 1..],
                uv_stride,
                &mut self.v_left[1..],
                uv_h,
                8,
            );
        }

        // Top row
        if y == 0 {
            // First row: use 127
            for i in 0..16 {
                self.y_top[x * 16 + i] = 127;
            }
            for i in 0..8 {
                self.uv_top[x * 16 + i] = 127;
                self.uv_top[x * 16 + 8 + i] = 127;
            }
        } else {
            // Import from source
            for i in 0..w {
                self.y_top[x * 16 + i] = y_src[y_offset - y_stride + i];
            }
            // Replicate last pixel
            let last_y = self.y_top[x * 16 + w - 1];
            for i in w..16 {
                self.y_top[x * 16 + i] = last_y;
            }

            for i in 0..uv_w {
                self.uv_top[x * 16 + i] = u_src[uv_offset - uv_stride + i];
                self.uv_top[x * 16 + 8 + i] = v_src[uv_offset - uv_stride + i];
            }
            // Replicate
            let last_u = self.uv_top[x * 16 + uv_w - 1];
            let last_v = self.uv_top[x * 16 + 8 + uv_w - 1];
            for i in uv_w..8 {
                self.uv_top[x * 16 + i] = last_u;
                self.uv_top[x * 16 + 8 + i] = last_v;
            }
        }
    }

    /// Advance to next macroblock
    /// Returns true if more macroblocks remain
    pub fn next(&mut self) -> bool {
        self.x += 1;
        if self.x >= self.mb_w {
            self.set_row(self.y + 1);
        }
        self.y < self.mb_h
    }

    /// Check if iteration is done
    pub fn is_done(&self) -> bool {
        self.y >= self.mb_h
    }

    /// Get left boundary for Y with corner at index 0
    /// Returns full y_left array: [0]=corner, [1..17]=left pixels
    fn get_y_left_with_corner(&self) -> Option<&[u8]> {
        if self.x > 0 {
            Some(&self.y_left[..])
        } else {
            None
        }
    }

    /// Get top boundary for Y
    fn get_y_top(&self) -> Option<&[u8]> {
        if self.y > 0 {
            Some(&self.y_top[self.x * 16..self.x * 16 + 16])
        } else {
            None
        }
    }

    /// Get left boundary for U with corner at index 0
    /// Returns full u_left array: [0]=corner, [1..9]=left pixels
    fn get_u_left_with_corner(&self) -> Option<&[u8]> {
        if self.x > 0 {
            Some(&self.u_left[..])
        } else {
            None
        }
    }

    /// Get left boundary for V with corner at index 0
    /// Returns full v_left array: [0]=corner, [1..9]=left pixels
    fn get_v_left_with_corner(&self) -> Option<&[u8]> {
        if self.x > 0 {
            Some(&self.v_left[..])
        } else {
            None
        }
    }

    /// Get top boundary for UV (interleaved)
    fn get_uv_top(&self) -> Option<&[u8]> {
        if self.y > 0 {
            Some(&self.uv_top[self.x * 16..self.x * 16 + 16])
        } else {
            None
        }
    }

    /// Analyze best I16 mode and return alpha
    /// Ported from libwebp's MBAnalyzeBestIntra16Mode
    pub fn analyze_best_intra16_mode(&mut self) -> (i32, usize) {
        let mut best_alpha = -1i32;
        let mut best_mode = 0usize;

        // Extract boundary data before mutable borrow
        let has_left = self.x > 0;
        let has_top = self.y > 0;
        let y_left_copy: [u8; 17] = self.y_left;
        let y_top_start = self.x * 16;

        // Generate all predictions
        // Pass copies/slices to avoid borrow conflicts
        let y_left_opt = if has_left {
            Some(&y_left_copy[..])
        } else {
            None
        };
        let y_top_opt = if has_top {
            Some(&self.y_top[y_top_start..y_top_start + 16])
        } else {
            None
        };
        make_luma16_preds(&mut self.yuv_p, y_left_opt, y_top_opt);

        // Test DC (mode 0) and TM (mode 1)
        for mode in 0..MAX_INTRA16_MODE {
            let pred_offset = VP8_I16_MODE_OFFSETS[mode];

            // Collect histogram comparing yuv_in+Y_OFF vs yuv_p+mode_offset
            let histo = collect_histogram_with_offset(
                &self.yuv_in,
                Y_OFF_ENC,
                &self.yuv_p,
                pred_offset,
                0,
                16,
            );

            let alpha = histo.get_alpha();
            if alpha > best_alpha {
                best_alpha = alpha;
                best_mode = mode;
            }
        }

        (best_alpha, best_mode)
    }

    /// Analyze best UV mode and return alpha
    /// Ported from libwebp's MBAnalyzeBestUVMode
    pub fn analyze_best_uv_mode(&mut self) -> (i32, usize) {
        let mut best_alpha = -1i32;
        let mut smallest_alpha = i32::MAX;
        let mut best_mode = 0usize;

        // Extract boundary data before mutable borrow
        let has_left = self.x > 0;
        let has_top = self.y > 0;
        let u_left_copy: [u8; 9] = self.u_left;
        let v_left_copy: [u8; 9] = self.v_left;
        let uv_top_start = self.x * 16;

        // Generate all chroma predictions
        let u_left_opt = if has_left {
            Some(&u_left_copy[..])
        } else {
            None
        };
        let v_left_opt = if has_left {
            Some(&v_left_copy[..])
        } else {
            None
        };
        let uv_top_opt = if has_top {
            Some(&self.uv_top[uv_top_start..uv_top_start + 16])
        } else {
            None
        };
        make_chroma8_preds(&mut self.yuv_p, u_left_opt, v_left_opt, uv_top_opt);

        // Test DC (mode 0) and TM (mode 1)
        for mode in 0..MAX_UV_MODE {
            let pred_offset = VP8_UV_MODE_OFFSETS[mode];

            // Collect histogram for U+V blocks (blocks 16-24 in VP8DspScan)
            let histo = collect_histogram_with_offset(
                &self.yuv_in,
                U_OFF_ENC,
                &self.yuv_p,
                pred_offset,
                16,
                24,
            );

            let alpha = histo.get_alpha();
            if alpha > best_alpha {
                best_alpha = alpha;
            }
            // Best prediction mode is the one with smallest alpha
            if mode == 0 || alpha < smallest_alpha {
                smallest_alpha = alpha;
                best_mode = mode;
            }
        }

        (best_alpha, best_mode)
    }
}

/// Collect histogram with different source and prediction offsets
/// This allows comparing yuv_in at src_off with yuv_p at pred_off
fn collect_histogram_with_offset(
    src_buf: &[u8],
    src_base: usize,
    pred_buf: &[u8],
    pred_base: usize,
    start_block: usize,
    end_block: usize,
) -> DctHistogram {
    let mut distribution = [0u32; MAX_COEFF_THRESH + 1];

    for j in start_block..end_block {
        let scan_off = VP8_DSP_SCAN[j];
        let src_off = src_base + scan_off;
        let pred_off = pred_base + scan_off;

        let dct_out = forward_dct_4x4(&src_buf[src_off..], &pred_buf[pred_off..], BPS, BPS);

        for coeff in dct_out.iter() {
            let v = (coeff.unsigned_abs() >> 3) as usize;
            let clipped = v.min(MAX_COEFF_THRESH);
            distribution[clipped] += 1;
        }
    }

    DctHistogram::from_distribution(&distribution)
}

/// Analyze a single macroblock and return its alpha
/// Ported from libwebp's MBAnalyze
pub fn analyze_macroblock(it: &mut AnalysisIterator) -> u8 {
    let (best_alpha, _best_mode) = it.analyze_best_intra16_mode();
    let (best_uv_alpha, _uv_mode) = it.analyze_best_uv_mode();

    // Final susceptibility mix
    let alpha = (3 * best_alpha + best_uv_alpha + 2) >> 2;

    // Finalize: invert and clip
    final_alpha_value(alpha)
}

/// Run full analysis pass on image and return alpha histogram + per-MB alphas
/// Ported from libwebp's VP8EncAnalyze / DoSegmentsJob
pub fn analyze_image(
    y_src: &[u8],
    u_src: &[u8],
    v_src: &[u8],
    width: usize,
    height: usize,
    y_stride: usize,
    uv_stride: usize,
) -> (Vec<u8>, [u32; 256]) {
    let mut it = AnalysisIterator::new(width, height);
    it.reset();

    let total_mbs = it.mb_w * it.mb_h;
    let mut mb_alphas = vec![0u8; total_mbs];
    let mut alpha_histogram = [0u32; 256];

    let mut mb_idx = 0;
    loop {
        it.import(y_src, u_src, v_src, y_stride, uv_stride);

        let alpha = analyze_macroblock(&mut it);
        mb_alphas[mb_idx] = alpha;
        alpha_histogram[alpha as usize] += 1;

        mb_idx += 1;
        if !it.next() {
            break;
        }
    }

    (mb_alphas, alpha_histogram)
}

// Keep old function signatures for compatibility but mark as deprecated
#[allow(dead_code)]
#[deprecated(note = "Use analyze_image instead")]
pub fn collect_dct_histogram(
    _src: &[u8],
    _pred: &[u8],
    _src_stride: usize,
    _pred_stride: usize,
    _block_offsets: &[(usize, usize)],
) -> DctHistogram {
    DctHistogram::new()
}

#[allow(dead_code)]
#[deprecated(note = "Use analyze_image instead")]
pub fn compute_mb_alpha(_dct_coeffs: &[i32; 256]) -> u8 {
    128
}

/// Assign macroblocks to segments using k-means clustering on alpha values.
///
/// # Arguments
/// * `alphas` - Alpha histogram (count of macroblocks with each alpha value)
/// * `num_segments` - Number of segments to use (1-4)
///
/// # Returns
/// (centers, map, weighted_average) where:
/// - centers[i] = alpha center for segment i
/// - map[alpha] = segment index for that alpha value
/// - weighted_average = weighted average of centers (for SetSegmentAlphas)
pub fn assign_segments_kmeans(
    alphas: &[u32; 256],
    num_segments: usize,
) -> ([u8; NUM_SEGMENTS], [u8; 256], i32) {
    let num_segments = num_segments.min(NUM_SEGMENTS);
    let mut centers = [0u8; NUM_SEGMENTS];
    let mut map = [0u8; 256];

    // Find min and max alpha with non-zero count
    let mut min_a = 0usize;
    let mut max_a = MAX_ALPHA as usize;

    for (n, &count) in alphas.iter().enumerate() {
        if count > 0 {
            min_a = n;
            break;
        }
    }
    for n in (min_a..=MAX_ALPHA as usize).rev() {
        if alphas[n] > 0 {
            max_a = n;
            break;
        }
    }

    let range_a = max_a.saturating_sub(min_a);

    // Initialize centers evenly spread across the range
    for (k, center) in centers.iter_mut().enumerate().take(num_segments) {
        let n = 1 + 2 * k;
        *center = (min_a + (n * range_a) / (2 * num_segments)) as u8;
    }

    // K-means iterations
    let mut accum = [0u32; NUM_SEGMENTS];
    let mut dist_accum = [0u32; NUM_SEGMENTS];
    let mut weighted_average = 0i32;
    let mut total_weight = 0u32;

    for _ in 0..MAX_ITERS_K_MEANS {
        // Reset accumulators
        for i in 0..num_segments {
            accum[i] = 0;
            dist_accum[i] = 0;
        }

        // Assign each alpha value to nearest center
        let mut current_center = 0usize;
        for a in min_a..=max_a {
            if alphas[a] > 0 {
                // Find nearest center
                while current_center + 1 < num_segments {
                    let d_curr = (a as i32 - centers[current_center] as i32).abs();
                    let d_next = (a as i32 - centers[current_center + 1] as i32).abs();
                    if d_next < d_curr {
                        current_center += 1;
                    } else {
                        break;
                    }
                }
                map[a] = current_center as u8;
                dist_accum[current_center] += a as u32 * alphas[a];
                accum[current_center] += alphas[a];
            }
        }

        // Move centers to center of their clouds
        // Also compute weighted_average from final centers (as libwebp does)
        let mut displaced = 0i32;
        weighted_average = 0;
        total_weight = 0;
        for n in 0..num_segments {
            if accum[n] > 0 {
                let new_center = ((dist_accum[n] + accum[n] / 2) / accum[n]) as u8;
                displaced += (centers[n] as i32 - new_center as i32).abs();
                centers[n] = new_center;
                // libwebp computes weighted_average from final centers
                weighted_average += new_center as i32 * accum[n] as i32;
                total_weight += accum[n];
            }
        }

        // Early exit if centers have converged
        if displaced < 5 {
            break;
        }
    }

    // Finalize weighted_average with rounding (matching libwebp)
    if total_weight > 0 {
        weighted_average = (weighted_average + total_weight as i32 / 2) / total_weight as i32;
    } else {
        weighted_average = 128;
    }

    // Fill unused segments with last valid center
    for i in num_segments..NUM_SEGMENTS {
        centers[i] = centers[num_segments - 1];
    }

    (centers, map, weighted_average)
}

/// Compute per-segment quantization using libwebp's formula.
///
/// This matches VP8SetSegmentParams in libwebp/src/enc/quant_enc.c
///
/// # Arguments
/// * `base_quant` - Base quantizer index (0-127), computed from quality
/// * `segment_alpha` - Transformed alpha for this segment, in range [-127, 127].
///   Computed as: 255 * (center - mid) / (max - min).
///   Positive = easier to compress, negative = harder.
/// * `sns_strength` - SNS strength (0-100), higher = more segment differentiation
///
/// # Returns
/// Adjusted quantizer index for this segment
pub fn compute_segment_quant(base_quant: u8, segment_alpha: i32, sns_strength: u8) -> u8 {
    // libwebp constant: scaling between SNS strength and quantizer modulation
    const SNS_TO_DQ: f64 = 0.9;

    // Amplitude of quantization modulation
    // amp = SNS_TO_DQ * sns_strength / 100 / 128
    let amp = SNS_TO_DQ * (sns_strength as f64) / 100.0 / 128.0;

    // Exponent for power-law modulation
    // segment_alpha is in [-127, 127] range
    // Positive alpha (easy) -> expn < 1 -> higher compression -> higher quant
    // Negative alpha (hard) -> expn > 1 -> lower compression -> lower quant
    let expn = 1.0 - amp * (segment_alpha as f64);

    // Ensure expn is positive (as asserted in libwebp)
    if expn <= 0.0 {
        return base_quant;
    }

    // Compression factor from base_quant
    // Since base_quant = 127 * (1 - c_base), we have c_base = 1 - base_quant/127
    let c_base = 1.0 - (base_quant as f64 / 127.0);

    // Apply power-law modulation
    let c = crate::fast_math::pow(c_base, expn);

    // Convert back to quantizer index
    let q = (127.0 * (1.0 - c)) as i32;
    q.clamp(0, 127) as u8
}
