//! Prediction functions for analysis pass.
//!
//! These generate intra predictions for luma (16x16) and chroma (8x8) blocks
//! for use in DCT histogram analysis. Only DC and TM modes are used during
//! analysis (libwebp's MAX_INTRA16_MODE = 2).
//!
//! ## SIMD Optimization Opportunities
//!
//! - `pred_luma16_dc`, `pred_chroma8_dc`: Horizontal sum + block fill
//! - `pred_luma16_tm`, `pred_chroma8_tm`: TrueMotion prediction with clamping
//! - `fill_block`: Memset-like operation

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use super::{BPS, C8DC8, C8TM8, I16DC16, I16TM16};

//------------------------------------------------------------------------------
// Block fill helpers

/// Fill a block with a constant value (BPS stride)
#[inline]
fn fill_block(dst: &mut [u8], value: u8, size: usize) {
    for y in 0..size {
        for x in 0..size {
            dst[y * BPS + x] = value;
        }
    }
}

/// Vertical prediction: copy top row to all rows
#[inline]
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
#[inline]
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

//------------------------------------------------------------------------------
// Luma 16x16 predictions

/// Generate DC prediction for 16x16 luma block
/// Ported from libwebp's DCMode (size=16, round=16, shift=5)
pub fn pred_luma16_dc(dst: &mut [u8], left: Option<&[u8]>, top: Option<&[u8]>) {
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

/// Generate TM (TrueMotion) prediction for 16x16 luma block
/// Ported from libwebp's TrueMotion
///
/// left_with_corner[0] = top-left corner, left_with_corner[1..17] = left pixels
pub fn pred_luma16_tm(dst: &mut [u8], left_with_corner: Option<&[u8]>, top: Option<&[u8]>) {
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
pub fn make_luma16_preds(yuv_p: &mut [u8], left_with_corner: Option<&[u8]>, top: Option<&[u8]>) {
    // DC prediction at I16DC16 (only needs left pixels, not corner)
    let left_only = left_with_corner.map(|l| &l[1..17]);
    pred_luma16_dc(&mut yuv_p[I16DC16..], left_only, top);

    // TM prediction at I16TM16 (needs corner at [0] and left pixels at [1..17])
    pred_luma16_tm(&mut yuv_p[I16TM16..], left_with_corner, top);

    // Note: V and H predictions not implemented since we only test DC and TM (MAX_INTRA16_MODE=2)
}

//------------------------------------------------------------------------------
// Chroma 8x8 predictions

/// Generate DC prediction for 8x8 chroma block into BPS-strided buffer
/// Ported from libwebp's DCMode (size=8, round=8, shift=4)
pub fn pred_chroma8_dc(dst: &mut [u8], left: Option<&[u8]>, top: Option<&[u8]>) {
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
pub fn pred_chroma8_tm(dst: &mut [u8], left_with_corner: Option<&[u8]>, top: Option<&[u8]>) {
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
pub fn make_chroma8_preds(
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
