//! NEON SIMD implementations for encoder distortion and utility functions.
//!
//! Ported from the x86 SSE2 versions in simd_sse.rs.

#[cfg(target_arch = "aarch64")]
use archmage::{arcane, rite, NeonToken};

#[cfg(target_arch = "aarch64")]
use safe_unaligned_simd::aarch64 as simd_mem;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use super::prediction::{CHROMA_BLOCK_SIZE, CHROMA_STRIDE, LUMA_BLOCK_SIZE, LUMA_STRIDE};

// =============================================================================
// SSE (Sum of Squared Errors) functions
// =============================================================================

/// NEON horizontal sum of four i32 values in a uint32x4_t
#[cfg(target_arch = "aarch64")]
#[rite]
fn hsum_u32x4(_token: NeonToken, v: uint32x4_t) -> u32 {
    vaddvq_u32(v)
}

/// SSE between two 4x4 blocks (16 bytes each)
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn sse4x4_neon(_token: NeonToken, a: &[u8; 16], b: &[u8; 16]) -> u32 {
    sse4x4_inner(_token, a, b)
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn sse4x4_inner(_token: NeonToken, a: &[u8; 16], b: &[u8; 16]) -> u32 {
    let a_vec = simd_mem::vld1q_u8(a);
    let b_vec = simd_mem::vld1q_u8(b);

    // Absolute difference
    let diff = vabdq_u8(a_vec, b_vec);

    // Widen to u16 and square
    let diff_lo = vmovl_u8(vget_low_u8(diff));
    let diff_hi = vmovl_u8(vget_high_u8(diff));
    let sq_lo = vmull_u16(vget_low_u16(diff_lo), vget_low_u16(diff_lo));
    let sq_hi = vmull_u16(vget_high_u16(diff_lo), vget_high_u16(diff_lo));
    let sq_lo2 = vmull_u16(vget_low_u16(diff_hi), vget_low_u16(diff_hi));
    let sq_hi2 = vmull_u16(vget_high_u16(diff_hi), vget_high_u16(diff_hi));

    // Sum all
    let sum = vaddq_u32(sq_lo, sq_hi);
    let sum = vaddq_u32(sum, sq_lo2);
    let sum = vaddq_u32(sum, sq_hi2);
    hsum_u32x4(_token, sum)
}

/// SSE with fused prediction + residual reconstruction
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn sse4x4_with_residual_neon(
    _token: NeonToken,
    src: &[u8; 16],
    pred: &[u8; 16],
    residual: &[i32; 16],
) -> u32 {
    sse4x4_with_residual_inner(_token, src, pred, residual)
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn sse4x4_with_residual_inner(
    _token: NeonToken,
    src: &[u8; 16],
    pred: &[u8; 16],
    residual: &[i32; 16],
) -> u32 {
    let src_vec = simd_mem::vld1q_u8(src);
    let pred_vec = simd_mem::vld1q_u8(pred);

    // Widen prediction to i16
    let pred_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pred_vec)));
    let pred_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(pred_vec)));

    // Load residuals and pack to i16
    let r0 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&residual[0..4]).unwrap());
    let r1 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&residual[4..8]).unwrap());
    let r2 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&residual[8..12]).unwrap());
    let r3 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&residual[12..16]).unwrap());
    let res_lo = vcombine_s16(vmovn_s32(r0), vmovn_s32(r1));
    let res_hi = vcombine_s16(vmovn_s32(r2), vmovn_s32(r3));

    // Reconstruct: pred + residual, clamp to [0, 255]
    let rec_lo = vaddq_s16(pred_lo, res_lo);
    let rec_hi = vaddq_s16(pred_hi, res_hi);
    let rec = vcombine_u8(vqmovun_s16(rec_lo), vqmovun_s16(rec_hi));

    // SSE between src and reconstructed
    let diff = vabdq_u8(src_vec, rec);
    let diff_lo = vmovl_u8(vget_low_u8(diff));
    let diff_hi = vmovl_u8(vget_high_u8(diff));
    let sq_lo = vmull_u16(vget_low_u16(diff_lo), vget_low_u16(diff_lo));
    let sq_hi = vmull_u16(vget_high_u16(diff_lo), vget_high_u16(diff_lo));
    let sq_lo2 = vmull_u16(vget_low_u16(diff_hi), vget_low_u16(diff_hi));
    let sq_hi2 = vmull_u16(vget_high_u16(diff_hi), vget_high_u16(diff_hi));

    let sum = vaddq_u32(sq_lo, sq_hi);
    let sum = vaddq_u32(sum, sq_lo2);
    let sum = vaddq_u32(sum, sq_hi2);
    hsum_u32x4(_token, sum)
}

/// SSE for 16x16 luma macroblock
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn sse_16x16_luma_neon(
    _token: NeonToken,
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    sse_16x16_luma_inner(_token, src_y, src_width, mbx, mby, pred)
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn sse_16x16_luma_inner(
    _token: NeonToken,
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    let mut acc = vdupq_n_u32(0);
    let src_base = mby * 16 * src_width + mbx * 16;

    for row in 0..16 {
        let src_off = src_base + row * src_width;
        let pred_off = (1 + row) * LUMA_STRIDE + 1; // Skip border
        let src_row = <&[u8; 16]>::try_from(&src_y[src_off..src_off + 16]).unwrap();
        let pred_row = <&[u8; 16]>::try_from(&pred[pred_off..pred_off + 16]).unwrap();

        let s = simd_mem::vld1q_u8(src_row);
        let p = simd_mem::vld1q_u8(pred_row);
        let diff = vabdq_u8(s, p);
        let d_lo = vmovl_u8(vget_low_u8(diff));
        let d_hi = vmovl_u8(vget_high_u8(diff));
        acc = vmlal_u16(acc, vget_low_u16(d_lo), vget_low_u16(d_lo));
        acc = vmlal_u16(acc, vget_high_u16(d_lo), vget_high_u16(d_lo));
        acc = vmlal_u16(acc, vget_low_u16(d_hi), vget_low_u16(d_hi));
        acc = vmlal_u16(acc, vget_high_u16(d_hi), vget_high_u16(d_hi));
    }

    hsum_u32x4(_token, acc)
}

/// SSE for 8x8 chroma block
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn sse_8x8_chroma_neon(
    _token: NeonToken,
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    sse_8x8_chroma_inner(_token, src_uv, src_width, mbx, mby, pred)
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn sse_8x8_chroma_inner(
    _token: NeonToken,
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    let chroma_width = src_width / 2;
    let mut acc = vdupq_n_u32(0);
    let src_base = mby * 8 * chroma_width + mbx * 8;

    for row in 0..8 {
        let src_off = src_base + row * chroma_width;
        let pred_off = (1 + row) * CHROMA_STRIDE + 1;

        let src_row = <&[u8; 8]>::try_from(&src_uv[src_off..src_off + 8]).unwrap();
        let pred_row = <&[u8; 8]>::try_from(&pred[pred_off..pred_off + 8]).unwrap();

        let s = simd_mem::vld1_u8(src_row);
        let p = simd_mem::vld1_u8(pred_row);
        let diff = vabd_u8(s, p);
        let d = vmovl_u8(diff);
        acc = vmlal_u16(acc, vget_low_u16(d), vget_low_u16(d));
        acc = vmlal_u16(acc, vget_high_u16(d), vget_high_u16(d));
    }

    hsum_u32x4(_token, acc)
}

// =============================================================================
// Spectral distortion (TDisto / Hadamard transform)
// =============================================================================

/// Fused TDisto for two 4x4 blocks: |weighted_hadamard(b) - weighted_hadamard(a)| >> 5
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn tdisto_4x4_fused_neon(
    _token: NeonToken,
    a: &[u8],
    b: &[u8],
    stride: usize,
    w: &[u16; 16],
) -> i32 {
    tdisto_4x4_fused_inner(_token, a, b, stride, w)
}

#[cfg(target_arch = "aarch64")]
#[rite]
pub(crate) fn tdisto_4x4_fused_inner(
    _token: NeonToken,
    a: &[u8],
    b: &[u8],
    stride: usize,
    w: &[u16; 16],
) -> i32 {
    // Load 4 bytes per row for both blocks, combine into i16 vectors
    // Layout: [a0 a1 a2 a3 b0 b1 b2 b3] per row

    // Helper: load 4 bytes from slice, widen to i16
    macro_rules! load_row_pair {
        ($src_a:expr, $src_b:expr, $off:expr) => {{
            let a_bytes = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(
                    &[
                        $src_a[$off],
                        $src_a[$off + 1],
                        $src_a[$off + 2],
                        $src_a[$off + 3],
                        $src_b[$off],
                        $src_b[$off + 1],
                        $src_b[$off + 2],
                        $src_b[$off + 3],
                    ][..],
                )
                .unwrap(),
            );
            vreinterpretq_s16_u16(vmovl_u8(a_bytes))
        }};
    }

    let mut tmp0 = load_row_pair!(a, b, 0);
    let mut tmp1 = load_row_pair!(a, b, stride);
    let mut tmp2 = load_row_pair!(a, b, stride * 2);
    let mut tmp3 = load_row_pair!(a, b, stride * 3);
    // tmp0 = [a00 a01 a02 a03 b00 b01 b02 b03] as i16

    // Vertical Hadamard
    {
        let va0 = vaddq_s16(tmp0, tmp2);
        let va1 = vaddq_s16(tmp1, tmp3);
        let va2 = vsubq_s16(tmp1, tmp3);
        let va3 = vsubq_s16(tmp0, tmp2);
        let vb0 = vaddq_s16(va0, va1);
        let vb1 = vaddq_s16(va3, va2);
        let vb2 = vsubq_s16(va3, va2);
        let vb3 = vsubq_s16(va0, va1);

        // Transpose both 4x4 blocks using NEON zip operations
        // vzipq returns x2 tuple types; destructure before reinterpret
        let t01 = vzipq_s16(vb0, vb1);
        let t23 = vzipq_s16(vb2, vb3);
        let r0 = vzipq_s32(
            vreinterpretq_s32_s16(t01.0),
            vreinterpretq_s32_s16(t23.0),
        );
        let r1 = vzipq_s32(
            vreinterpretq_s32_s16(t01.1),
            vreinterpretq_s32_s16(t23.1),
        );
        tmp0 = vreinterpretq_s16_s32(r0.0);
        tmp1 = vreinterpretq_s16_s32(r0.1);
        tmp2 = vreinterpretq_s16_s32(r1.0);
        tmp3 = vreinterpretq_s16_s32(r1.1);
    }

    // Horizontal Hadamard
    let ha0 = vaddq_s16(tmp0, tmp2);
    let ha1 = vaddq_s16(tmp1, tmp3);
    let ha2 = vsubq_s16(tmp1, tmp3);
    let ha3 = vsubq_s16(tmp0, tmp2);
    let hb0 = vaddq_s16(ha0, ha1);
    let hb1 = vaddq_s16(ha3, ha2);
    let hb2 = vsubq_s16(ha3, ha2);
    let hb3 = vsubq_s16(ha0, ha1);

    // Separate A (low 4) and B (high 4), take absolute values
    let a_01 = vcombine_s16(vget_low_s16(hb0), vget_low_s16(hb1));
    let a_23 = vcombine_s16(vget_low_s16(hb2), vget_low_s16(hb3));
    let b_01 = vcombine_s16(vget_high_s16(hb0), vget_high_s16(hb1));
    let b_23 = vcombine_s16(vget_high_s16(hb2), vget_high_s16(hb3));

    let a_abs_01 = vabsq_s16(a_01);
    let a_abs_23 = vabsq_s16(a_23);
    let b_abs_01 = vabsq_s16(b_01);
    let b_abs_23 = vabsq_s16(b_23);

    // Load weights
    let w_0 = simd_mem::vld1q_u16(<&[u16; 8]>::try_from(&w[0..8]).unwrap());
    let w_8 = simd_mem::vld1q_u16(<&[u16; 8]>::try_from(&w[8..16]).unwrap());
    let w_0s = vreinterpretq_s16_u16(w_0);
    let w_8s = vreinterpretq_s16_u16(w_8);

    // Weighted multiply-accumulate: sum(abs_coeff * weight)
    // Use vmull + vmlal for multiply-accumulate to i32
    let a_prod_01 = vmull_s16(vget_low_s16(a_abs_01), vget_low_s16(w_0s));
    let a_prod_01 = vmlal_s16(a_prod_01, vget_high_s16(a_abs_01), vget_high_s16(w_0s));
    let a_prod_23 = vmull_s16(vget_low_s16(a_abs_23), vget_low_s16(w_8s));
    let a_prod_23 = vmlal_s16(a_prod_23, vget_high_s16(a_abs_23), vget_high_s16(w_8s));

    let b_prod_01 = vmull_s16(vget_low_s16(b_abs_01), vget_low_s16(w_0s));
    let b_prod_01 = vmlal_s16(b_prod_01, vget_high_s16(b_abs_01), vget_high_s16(w_0s));
    let b_prod_23 = vmull_s16(vget_low_s16(b_abs_23), vget_low_s16(w_8s));
    let b_prod_23 = vmlal_s16(b_prod_23, vget_high_s16(b_abs_23), vget_high_s16(w_8s));

    // Horizontal sum
    let a_sum = vaddq_s32(a_prod_01, a_prod_23);
    let b_sum = vaddq_s32(b_prod_01, b_prod_23);

    let sum_a = vaddvq_s32(a_sum);
    let sum_b = vaddvq_s32(b_sum);

    (sum_b - sum_a).abs() >> 5
}

// =============================================================================
// Flatness check functions
// =============================================================================

/// Check if a 16x16 source block is all one color
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn is_flat_source_16_neon(
    _token: NeonToken,
    src: &[u8],
    stride: usize,
) -> bool {
    is_flat_source_16_inner(_token, src, stride)
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn is_flat_source_16_inner(
    _token: NeonToken,
    src: &[u8],
    stride: usize,
) -> bool {
    // Compare all bytes against the first byte
    let first = vdupq_n_u8(src[0]);
    let mut all_eq = vdupq_n_u8(0xff);

    for row in 0..16 {
        let off = row * stride;
        let row_data = simd_mem::vld1q_u8(
            <&[u8; 16]>::try_from(&src[off..off + 16]).unwrap(),
        );
        let eq = vceqq_u8(row_data, first);
        all_eq = vandq_u8(all_eq, eq);
    }

    // Check if all lanes are 0xFF (all equal)
    vminvq_u8(all_eq) == 0xff
}

/// Check if coefficients are "flat" (few non-zero AC coefficients)
#[cfg(target_arch = "aarch64")]
#[arcane]
pub(crate) fn is_flat_coeffs_neon(
    _token: NeonToken,
    levels: &[i16],
    num_blocks: usize,
    thresh: i32,
) -> bool {
    is_flat_coeffs_inner(_token, levels, num_blocks, thresh)
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn is_flat_coeffs_inner(
    _token: NeonToken,
    levels: &[i16],
    num_blocks: usize,
    thresh: i32,
) -> bool {
    let zero = vdupq_n_s16(0);
    let mut count = 0i32;

    for block in 0..num_blocks {
        let off = block * 16;
        if off + 16 > levels.len() {
            break;
        }
        let v = simd_mem::vld1q_s16(
            <&[i16; 8]>::try_from(&levels[off..off + 8]).unwrap(),
        );
        let v2 = simd_mem::vld1q_s16(
            <&[i16; 8]>::try_from(&levels[off + 8..off + 16]).unwrap(),
        );
        // Count lanes where v != 0
        let ne0 = vmvnq_u16(vceqq_s16(v, zero));
        let ne1 = vmvnq_u16(vceqq_s16(v2, zero));
        // Each non-zero lane becomes 0xFFFF; AND with 1 to get 0 or 1
        let one = vdupq_n_u16(1);
        let c0 = vandq_u16(ne0, one);
        let c1 = vandq_u16(ne1, one);
        let total = vaddq_u16(c0, c1);
        // Skip DC (position 0): subtract 1 if DC was non-zero
        let dc_nz = if levels[off] != 0 { 1i32 } else { 0 };
        count += vaddvq_u16(total) as i32 - dc_nz;
    }

    count <= thresh
}
