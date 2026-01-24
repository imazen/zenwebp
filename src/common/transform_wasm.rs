//! WASM SIMD128 intrinsics for DCT/IDCT transforms.
//!
//! Uses v128 type and i16x8/i32x4 intrinsics matching the x86 SSE2 implementation.
//! All intrinsics used here are safe in Rust 1.89+.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

#[cfg(target_arch = "wasm32")]
use archmage::{arcane, Simd128Token};

// =============================================================================
// Public dispatch functions
// =============================================================================

/// Forward DCT using WASM SIMD128
#[cfg(target_arch = "wasm32")]
pub(crate) fn dct4x4_wasm(token: Simd128Token, block: &mut [i32; 16]) {
    dct4x4_wasm_impl(token, block);
}

/// Inverse DCT using WASM SIMD128
#[cfg(target_arch = "wasm32")]
pub(crate) fn idct4x4_wasm(token: Simd128Token, block: &mut [i32]) {
    debug_assert!(block.len() >= 16);
    idct4x4_wasm_impl(token, block);
}

// =============================================================================
// WASM SIMD128 Implementation
// =============================================================================

/// Load 4 i32 values into a v128 using safe lane constructors
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn load_i32x4(block: &[i32], offset: usize) -> v128 {
    i32x4(
        block[offset],
        block[offset + 1],
        block[offset + 2],
        block[offset + 3],
    )
}

/// Store v128 as 4 i32 values using safe lane extraction
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn store_i32x4(block: &mut [i32], offset: usize, v: v128) {
    block[offset] = i32x4_extract_lane::<0>(v);
    block[offset + 1] = i32x4_extract_lane::<1>(v);
    block[offset + 2] = i32x4_extract_lane::<2>(v);
    block[offset + 3] = i32x4_extract_lane::<3>(v);
}

/// Load 8 i16 values into a v128
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn load_i16x8_from_i32(block: &[i32], indices: [usize; 8]) -> v128 {
    i16x8(
        block[indices[0]] as i16,
        block[indices[1]] as i16,
        block[indices[2]] as i16,
        block[indices[3]] as i16,
        block[indices[4]] as i16,
        block[indices[5]] as i16,
        block[indices[6]] as i16,
        block[indices[7]] as i16,
    )
}

/// Store 8 i16 values to array
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn store_i16x8(out: &mut [i16], offset: usize, v: v128) {
    out[offset] = i16x8_extract_lane::<0>(v);
    out[offset + 1] = i16x8_extract_lane::<1>(v);
    out[offset + 2] = i16x8_extract_lane::<2>(v);
    out[offset + 3] = i16x8_extract_lane::<3>(v);
    out[offset + 4] = i16x8_extract_lane::<4>(v);
    out[offset + 5] = i16x8_extract_lane::<5>(v);
    out[offset + 6] = i16x8_extract_lane::<6>(v);
    out[offset + 7] = i16x8_extract_lane::<7>(v);
}

/// Forward DCT implementation using WASM SIMD128
#[cfg(target_arch = "wasm32")]
#[arcane]
fn dct4x4_wasm_impl(_token: Simd128Token, block: &mut [i32; 16]) {
    // Convert i32 block to i16 and reorganize into libwebp's interleaved layout:
    // row01 = [r0c0, r0c1, r1c0, r1c1, r0c2, r0c3, r1c2, r1c3]
    // row23 = [r2c0, r2c1, r3c0, r3c1, r2c2, r2c3, r3c2, r3c3]
    let row01 = load_i16x8_from_i32(block, [0, 1, 4, 5, 2, 3, 6, 7]);
    let row23 = load_i16x8_from_i32(block, [8, 9, 12, 13, 10, 11, 14, 15]);

    // Forward transform pass 1 (rows)
    let (v01, v32) = ftransform_pass1_wasm(row01, row23);

    // Forward transform pass 2 (columns)
    let mut out16 = [0i16; 16];
    ftransform_pass2_wasm(&v01, &v32, &mut out16);

    // Convert i16 output back to i32
    for i in 0..16 {
        block[i] = out16[i] as i32;
    }
}

/// FTransform Pass 1 - matches libwebp FTransformPass1_SSE2
#[cfg(target_arch = "wasm32")]
#[inline]
fn ftransform_pass1_wasm(in01: v128, in23: v128) -> (v128, v128) {
    // Constants matching libwebp exactly
    let k937 = i32x4_splat(937);
    let k1812 = i32x4_splat(1812);
    let k88p = i16x8(8, 8, 8, 8, 8, 8, 8, 8);
    let k88m = i16x8(8, -8, 8, -8, 8, -8, 8, -8);
    let k5352_2217p = i16x8(5352, 2217, 5352, 2217, 5352, 2217, 5352, 2217);
    let k5352_2217m = i16x8(2217, -5352, 2217, -5352, 2217, -5352, 2217, -5352);

    // Shuffle high words: swap pairs in high 64-bits
    // shufflehi(in01, 0b10_11_00_01) = 00 01 10 11 03 02 13 12
    let shuf01_p = i16x8_shuffle::<0, 1, 2, 3, 5, 4, 7, 6>(in01, in01);
    let shuf23_p = i16x8_shuffle::<0, 1, 2, 3, 5, 4, 7, 6>(in23, in23);

    // Interleave to separate low/high parts
    let s01 = i64x2_shuffle::<0, 2>(shuf01_p, shuf23_p);
    let s32 = i64x2_shuffle::<1, 3>(shuf01_p, shuf23_p);

    // a01 = [d0+d3, d1+d2], a32 = [d0-d3, d1-d2]
    let a01 = i16x8_add(s01, s32);
    let a32 = i16x8_sub(s01, s32);

    // tmp0 = (a0 + a1) << 3, tmp2 = (a0 - a1) << 3
    let tmp0 = i32x4_dot_i16x8(a01, k88p);
    let tmp2 = i32x4_dot_i16x8(a01, k88m);

    // tmp1 = (a3*5352 + a2*2217 + 1812) >> 9
    // tmp3 = (a3*2217 - a2*5352 + 937) >> 9
    let tmp1_1 = i32x4_dot_i16x8(a32, k5352_2217p);
    let tmp3_1 = i32x4_dot_i16x8(a32, k5352_2217m);
    let tmp1_2 = i32x4_add(tmp1_1, k1812);
    let tmp3_2 = i32x4_add(tmp3_1, k937);
    let tmp1 = i32x4_shr(tmp1_2, 9);
    let tmp3 = i32x4_shr(tmp3_2, 9);

    // Pack back to i16 with saturation
    let s03 = i16x8_narrow_i32x4(tmp0, tmp2);
    let s12 = i16x8_narrow_i32x4(tmp1, tmp3);

    // Interleave to get proper output order
    let s_lo = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(s03, s12);
    let s_hi = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(s03, s12);
    let v23 = i32x4_shuffle::<2, 3, 6, 7>(s_lo, s_hi);
    let out01 = i32x4_shuffle::<0, 1, 4, 5>(s_lo, s_hi);
    let out32 = i32x4_shuffle::<1, 0, 3, 2>(v23, v23);

    (out01, out32)
}

/// FTransform Pass 2 - matches libwebp FTransformPass2_SSE2
#[cfg(target_arch = "wasm32")]
#[inline]
fn ftransform_pass2_wasm(v01: &v128, v32: &v128, out: &mut [i16; 16]) {
    let zero = i16x8_splat(0);
    let seven = i16x8_splat(7);
    let k5352_2217 = i16x8(2217, 5352, 2217, 5352, 2217, 5352, 2217, 5352);
    let k2217_5352 = i16x8(-5352, 2217, -5352, 2217, -5352, 2217, -5352, 2217);
    let k12000_plus_one = i32x4_splat(12000 + (1 << 16));
    let k51000 = i32x4_splat(51000);

    // a3 = v0 - v3, a2 = v1 - v2
    let a32 = i16x8_sub(*v01, *v32);
    let a22 = i64x2_shuffle::<1, 1>(a32, a32);

    let b23 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a22, a32);
    let c1 = i32x4_dot_i16x8(b23, k5352_2217);
    let c3 = i32x4_dot_i16x8(b23, k2217_5352);
    let d1 = i32x4_add(c1, k12000_plus_one);
    let d3 = i32x4_add(c3, k51000);
    let e1 = i32x4_shr(d1, 16);
    let e3 = i32x4_shr(d3, 16);

    let f1 = i16x8_narrow_i32x4(e1, e1);
    let f3 = i16x8_narrow_i32x4(e3, e3);

    // g1 = f1 + (a3 != 0)
    let cmp = i16x8_eq(a32, zero);
    let g1 = i16x8_add(f1, cmp);

    // a0 = v0 + v3, a1 = v1 + v2
    let a01 = i16x8_add(*v01, *v32);
    let a01_plus_7 = i16x8_add(a01, seven);
    let a11 = i64x2_shuffle::<1, 1>(a01, a01);
    let c0 = i16x8_add(a01_plus_7, a11);
    let c2 = i16x8_sub(a01_plus_7, a11);

    let d0 = i16x8_shr(c0, 4);
    let d2 = i16x8_shr(c2, 4);

    // Combine outputs
    let d0_g1 = i64x2_shuffle::<0, 2>(d0, g1);
    let d2_f3 = i64x2_shuffle::<0, 2>(d2, f3);

    // Store results
    store_i16x8(out, 0, d0_g1);
    store_i16x8(out, 8, d2_f3);
}

/// Inverse DCT implementation using WASM SIMD128
#[cfg(target_arch = "wasm32")]
#[arcane]
fn idct4x4_wasm_impl(_token: Simd128Token, block: &mut [i32]) {
    // libwebp IDCT constants
    let k1k2 = i16x8(20091, 20091, 20091, 20091, -30068, -30068, -30068, -30068);
    let k2k1 = i16x8(-30068, -30068, -30068, -30068, 20091, 20091, 20091, 20091);
    let zero_four = i16x8(4, 4, 4, 4, 0, 0, 0, 0);

    // Load i32 values and pack to i16
    let i32_0 = load_i32x4(block, 0);
    let i32_1 = load_i32x4(block, 4);
    let i32_2 = load_i32x4(block, 8);
    let i32_3 = load_i32x4(block, 12);

    // Pack i32 to i16 with saturation
    let in01 = i16x8_narrow_i32x4(i32_0, i32_1);
    let in23 = i16x8_narrow_i32x4(i32_2, i32_3);

    // Vertical pass
    let (t01, t23) = itransform_pass_wasm(in01, in23, k1k2, k2k1);

    // Horizontal pass with rounding
    let (out01, out23) = itransform_pass2_wasm(t01, t23, k1k2, k2k1, zero_four);

    // Unpack i16 back to i32 using sign extension
    let zero = i16x8_splat(0);

    let sign01_lo = i16x8_gt(zero, out01);
    let out_0 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(out01, sign01_lo);
    let out_1 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(out01, sign01_lo);

    let sign23_lo = i16x8_gt(zero, out23);
    let out_2 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(out23, sign23_lo);
    let out_3 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(out23, sign23_lo);

    // Store results
    store_i32x4(block, 0, out_0);
    store_i32x4(block, 4, out_1);
    store_i32x4(block, 8, out_2);
    store_i32x4(block, 12, out_3);
}

/// ITransform vertical pass
#[cfg(target_arch = "wasm32")]
#[inline]
fn itransform_pass_wasm(in01: v128, in23: v128, k1k2: v128, k2k1: v128) -> (v128, v128) {
    let in1 = i64x2_shuffle::<1, 1>(in01, in01);
    let in3 = i64x2_shuffle::<1, 1>(in23, in23);

    let a_d3 = i16x8_add(in01, in23);
    let b_c3 = i16x8_sub(in01, in23);

    // Use q15mulr_sat for the multiply-high operation
    let c1d1 = i16x8_q15mulr_sat(in1, k2k1);
    let c2d2 = i16x8_q15mulr_sat(in3, k1k2);
    let c3 = i64x2_shuffle::<1, 1>(b_c3, b_c3);
    let c4 = i16x8_sub(c1d1, c2d2);
    let c = i16x8_add(c3, c4);
    let d4u = i16x8_add(c1d1, c2d2);
    let du = i16x8_add(a_d3, d4u);
    let d = i64x2_shuffle::<1, 1>(du, du);

    let comb_ab = i64x2_shuffle::<0, 2>(a_d3, b_c3);
    let comb_dc = i64x2_shuffle::<0, 2>(d, c);

    let tmp01 = i16x8_add(comb_ab, comb_dc);
    let tmp32 = i16x8_sub(comb_ab, comb_dc);
    let tmp23 = i32x4_shuffle::<2, 3, 0, 1>(tmp32, tmp32);

    let transpose_0 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(tmp01, tmp23);
    let transpose_1 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(tmp01, tmp23);

    let t01 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(transpose_0, transpose_1);
    let t23 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(transpose_0, transpose_1);

    (t01, t23)
}

/// ITransform horizontal pass with final shift
#[cfg(target_arch = "wasm32")]
#[inline]
fn itransform_pass2_wasm(
    t01: v128,
    t23: v128,
    k1k2: v128,
    k2k1: v128,
    zero_four: v128,
) -> (v128, v128) {
    let t1 = i64x2_shuffle::<1, 1>(t01, t01);
    let t3 = i64x2_shuffle::<1, 1>(t23, t23);

    let dc = i16x8_add(t01, zero_four);

    let a_d3 = i16x8_add(dc, t23);
    let b_c3 = i16x8_sub(dc, t23);

    let c1d1 = i16x8_q15mulr_sat(t1, k2k1);
    let c2d2 = i16x8_q15mulr_sat(t3, k1k2);
    let c3 = i64x2_shuffle::<1, 1>(b_c3, b_c3);
    let c4 = i16x8_sub(c1d1, c2d2);
    let c = i16x8_add(c3, c4);
    let d4u = i16x8_add(c1d1, c2d2);
    let du = i16x8_add(a_d3, d4u);
    let d = i64x2_shuffle::<1, 1>(du, du);

    let comb_ab = i64x2_shuffle::<0, 2>(a_d3, b_c3);
    let comb_dc = i64x2_shuffle::<0, 2>(d, c);

    let tmp01 = i16x8_add(comb_ab, comb_dc);
    let tmp32 = i16x8_sub(comb_ab, comb_dc);
    let tmp23 = i32x4_shuffle::<2, 3, 0, 1>(tmp32, tmp32);

    // Final shift >> 3
    let shifted01 = i16x8_shr(tmp01, 3);
    let shifted23 = i16x8_shr(tmp23, 3);

    // Transpose back
    let transpose_0 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(shifted01, shifted23);
    let transpose_1 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(shifted01, shifted23);

    let out01 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(transpose_0, transpose_1);
    let out23 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(transpose_0, transpose_1);

    (out01, out23)
}
