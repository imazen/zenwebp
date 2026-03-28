//! DCT/IDCT and WHT/IWHT transforms with multi-platform SIMD.
//!
//! All platform variants (scalar, SSE2, NEON, WASM SIMD128) in one file.

#![allow(clippy::too_many_arguments)]
// Allow dead code when std is disabled - some functions are encoder-only
#![cfg_attr(not(feature = "std"), allow(dead_code))]
// Allow dead code when std is disabled - some functions are encoder-only
#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use core::arch::x86_64::*;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use archmage::intrinsics::x86_64 as simd_mem;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use archmage::{SimdToken, X64V3Token, arcane, rite};

#[cfg(target_arch = "aarch64")]
use archmage::intrinsics::aarch64 as simd_mem;
#[cfg(target_arch = "aarch64")]
use archmage::{NeonToken, arcane, rite};
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "wasm32")]
use archmage::{Wasm128Token, arcane};
#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

use core::convert::TryFrom;

// ============================================================================
// Array splitting helpers — zero-cost conversion from [T; 16] to [T; 4] rows
// ============================================================================

/// Split `&[T; 16]` into four `&[T; 4]` references (rows 0-3).
/// Compiles to no-ops — just pointer arithmetic.
#[inline(always)]
fn rows4<T>(a: &[T; 16]) -> (&[T; 4], &[T; 4], &[T; 4], &[T; 4]) {
    let (r0, rest) = a.split_first_chunk::<4>().unwrap();
    let (r1, rest) = rest.split_first_chunk::<4>().unwrap();
    let (r2, rest) = rest.split_first_chunk::<4>().unwrap();
    let r3: &[T; 4] = rest.try_into().unwrap();
    (r0, r1, r2, r3)
}

/// Split `&mut [T; 16]` into four `&mut [T; 4]` references (rows 0-3).
#[inline(always)]
fn rows4_mut<T>(a: &mut [T; 16]) -> (&mut [T; 4], &mut [T; 4], &mut [T; 4], &mut [T; 4]) {
    let (r0, rest) = a.split_first_chunk_mut::<4>().unwrap();
    let (r1, rest) = rest.split_first_chunk_mut::<4>().unwrap();
    let (r2, rest) = rest.split_first_chunk_mut::<4>().unwrap();
    let r3: &mut [T; 4] = rest.try_into().unwrap();
    (r0, r1, r2, r3)
}

/// Split `&[T; 16]` into two `&[T; 8]` halves.
#[inline(always)]
fn halves8<T>(a: &[T; 16]) -> (&[T; 8], &[T; 8]) {
    let (lo, rest) = a.split_first_chunk::<8>().unwrap();
    let hi: &[T; 8] = rest.try_into().unwrap();
    (lo, hi)
}

/// Split `&mut [T; 16]` into two `&mut [T; 8]` halves.
#[inline(always)]
fn halves8_mut<T>(a: &mut [T; 16]) -> (&mut [T; 8], &mut [T; 8]) {
    let (lo, rest) = a.split_first_chunk_mut::<8>().unwrap();
    let hi: &mut [T; 8] = rest.try_into().unwrap();
    (lo, hi)
}

/// 16 bit fixed point version of cos(PI/8) * sqrt(2) - 1
const CONST1: i64 = 20091;
/// 16 bit fixed point version of sin(PI/8) * sqrt(2)
const CONST2: i64 = 35468;

/// DC-only inverse transform: fills all 16 positions with (DC+4)>>3
/// Used when a block has only DC coefficient (no AC), avoiding full IDCT.
/// The input block[0] contains the quantized DC value; AC positions are ignored.
#[inline(always)]
pub(crate) fn idct4x4_dc(block: &mut [i32; 16]) {
    let dc = (block[0] + 4) >> 3;
    block.fill(dc);
}

// inverse discrete cosine transform, used in decoding
#[inline(always)]
pub(crate) fn idct4x4(block: &mut [i32; 16]) {
    idct4x4_intrinsics(block);
}

/// Inverse DCT with pre-summoned SIMD token (avoids per-call token summoning)
#[inline(always)]
#[allow(dead_code)] // May be useful for alternative decode paths
pub(crate) fn idct4x4_with_token(
    block: &mut [i32; 16],
    simd_token: super::prediction::SimdTokenType,
) {
    idct4x4_intrinsics_with_token(block, simd_token);
}

#[archmage::autoversion(cfg(simd))]
#[allow(dead_code)]
pub(crate) fn idct4x4_scalar(block: &mut [i32; 16]) {
    // The intermediate results may overflow the types, so we stretch the type.
    fn fetch(block: &[i32], idx: usize) -> i64 {
        i64::from(block[idx])
    }

    for i in 0usize..4 {
        let a1 = fetch(block, i) + fetch(block, 8 + i);
        let b1 = fetch(block, i) - fetch(block, 8 + i);

        let t1 = (fetch(block, 4 + i) * CONST2) >> 16;
        let t2 = fetch(block, 12 + i) + ((fetch(block, 12 + i) * CONST1) >> 16);
        let c1 = t1 - t2;

        let t1 = fetch(block, 4 + i) + ((fetch(block, 4 + i) * CONST1) >> 16);
        let t2 = (fetch(block, 12 + i) * CONST2) >> 16;
        let d1 = t1 + t2;

        block[i] = (a1 + d1) as i32;
        block[4 + i] = (b1 + c1) as i32;
        block[4 * 3 + i] = (a1 - d1) as i32;
        block[4 * 2 + i] = (b1 - c1) as i32;
    }

    for i in 0usize..4 {
        let a1 = fetch(block, 4 * i) + fetch(block, 4 * i + 2);
        let b1 = fetch(block, 4 * i) - fetch(block, 4 * i + 2);

        let t1 = (fetch(block, 4 * i + 1) * CONST2) >> 16;
        let t2 = fetch(block, 4 * i + 3) + ((fetch(block, 4 * i + 3) * CONST1) >> 16);
        let c1 = t1 - t2;

        let t1 = fetch(block, 4 * i + 1) + ((fetch(block, 4 * i + 1) * CONST1) >> 16);
        let t2 = (fetch(block, 4 * i + 3) * CONST2) >> 16;
        let d1 = t1 + t2;

        block[4 * i] = ((a1 + d1 + 4) >> 3) as i32;
        block[4 * i + 3] = ((a1 - d1 + 4) >> 3) as i32;
        block[4 * i + 1] = ((b1 + c1 + 4) >> 3) as i32;
        block[4 * i + 2] = ((b1 - c1 + 4) >> 3) as i32;
    }
}

// 14.3 inverse walsh-hadamard transform, used in decoding
#[archmage::autoversion(cfg(simd))]
pub(crate) fn iwht4x4(block: &mut [i32; 16]) {

    for i in 0usize..4 {
        let a1 = block[i] + block[12 + i];
        let b1 = block[4 + i] + block[8 + i];
        let c1 = block[4 + i] - block[8 + i];
        let d1 = block[i] - block[12 + i];

        block[i] = a1 + b1;
        block[4 + i] = c1 + d1;
        block[8 + i] = a1 - b1;
        block[12 + i] = d1 - c1;
    }

    for block in block.chunks_exact_mut(4) {
        let a1 = block[0] + block[3];
        let b1 = block[1] + block[2];
        let c1 = block[1] - block[2];
        let d1 = block[0] - block[3];

        let a2 = a1 + b1;
        let b2 = c1 + d1;
        let c2 = a1 - b1;
        let d2 = d1 - c1;

        block[0] = (a2 + 3) >> 3;
        block[1] = (b2 + 3) >> 3;
        block[2] = (c2 + 3) >> 3;
        block[3] = (d2 + 3) >> 3;
    }
}

#[archmage::autoversion(cfg(simd))]
pub(crate) fn wht4x4(block: &mut [i32; 16]) {
    // The intermediate results may overflow the types, so we stretch the type.
    fn fetch(block: &[i32], idx: usize) -> i64 {
        i64::from(block[idx])
    }

    // vertical
    for i in 0..4 {
        let a = fetch(block, i * 4) + fetch(block, i * 4 + 3);
        let b = fetch(block, i * 4 + 1) + fetch(block, i * 4 + 2);
        let c = fetch(block, i * 4 + 1) - fetch(block, i * 4 + 2);
        let d = fetch(block, i * 4) - fetch(block, i * 4 + 3);

        block[i * 4] = (a + b) as i32;
        block[i * 4 + 1] = (c + d) as i32;
        block[i * 4 + 2] = (a - b) as i32;
        block[i * 4 + 3] = (d - c) as i32;
    }

    // horizontal
    for i in 0..4 {
        let a1 = fetch(block, i) + fetch(block, i + 12);
        let b1 = fetch(block, i + 4) + fetch(block, i + 8);
        let c1 = fetch(block, i + 4) - fetch(block, i + 8);
        let d1 = fetch(block, i) - fetch(block, i + 12);

        let a2 = a1 + b1;
        let b2 = c1 + d1;
        let c2 = a1 - b1;
        let d2 = d1 - c1;

        let a3 = (a2 + if a2 > 0 { 1 } else { 0 }) / 2;
        let b3 = (b2 + if b2 > 0 { 1 } else { 0 }) / 2;
        let c3 = (c2 + if c2 > 0 { 1 } else { 0 }) / 2;
        let d3 = (d2 + if d2 > 0 { 1 } else { 0 }) / 2;

        block[i] = a3 as i32;
        block[i + 4] = b3 as i32;
        block[i + 8] = c3 as i32;
        block[i + 12] = d3 as i32;
    }
}

#[inline(always)]
pub(crate) fn dct4x4(block: &mut [i32; 16]) {
    dct4x4_intrinsics(block);
}

// Scalar DCT implementation for reference and non-SIMD builds
#[archmage::autoversion(cfg(simd))]
#[allow(dead_code)]
pub(crate) fn dct4x4_scalar(block: &mut [i32; 16]) {
    // The intermediate results may overflow the types, so we stretch the type.
    fn fetch(block: &[i32], idx: usize) -> i64 {
        i64::from(block[idx])
    }

    // vertical
    for i in 0..4 {
        let a = (fetch(block, i * 4) + fetch(block, i * 4 + 3)) * 8;
        let b = (fetch(block, i * 4 + 1) + fetch(block, i * 4 + 2)) * 8;
        let c = (fetch(block, i * 4 + 1) - fetch(block, i * 4 + 2)) * 8;
        let d = (fetch(block, i * 4) - fetch(block, i * 4 + 3)) * 8;

        block[i * 4] = (a + b) as i32;
        block[i * 4 + 2] = (a - b) as i32;
        block[i * 4 + 1] = ((c * 2217 + d * 5352 + 14500) >> 12) as i32;
        block[i * 4 + 3] = ((d * 2217 - c * 5352 + 7500) >> 12) as i32;
    }

    // horizontal
    for i in 0..4 {
        let a = fetch(block, i) + fetch(block, i + 12);
        let b = fetch(block, i + 4) + fetch(block, i + 8);
        let c = fetch(block, i + 4) - fetch(block, i + 8);
        let d = fetch(block, i) - fetch(block, i + 12);

        block[i] = ((a + b + 7) >> 4) as i32;
        block[i + 8] = ((a - b + 7) >> 4) as i32;
        block[i + 4] = (((c * 2217 + d * 5352 + 12000) >> 16) + if d != 0 { 1 } else { 0 }) as i32;
        block[i + 12] = ((d * 2217 - c * 5352 + 51000) >> 16) as i32;
    }
}

// ============================================================================
// SSE2 implementations (from transform_simd_intrinsics.rs)
// ============================================================================

// =============================================================================
// Public dispatch functions
// =============================================================================

/// Forward DCT with dynamic dispatch to best available implementation
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
pub(crate) fn dct4x4_intrinsics(block: &mut [i32; 16]) {
    if let Some(token) = X64V3Token::summon() {
        dct4x4_entry(token, block);
    } else {
        dct4x4_scalar(block);
    }
}

/// Forward DCT - non-x86 fallback
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
pub(crate) fn dct4x4_intrinsics(block: &mut [i32; 16]) {
    #[cfg(target_arch = "wasm32")]
    {
        use archmage::{SimdToken, Wasm128Token};
        if let Some(token) = Wasm128Token::summon() {
            dct4x4_wasm(token, block);
        } else {
            dct4x4_scalar(block);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::{NeonToken, SimdToken};
        if let Some(token) = NeonToken::try_new() {
            dct4x4_neon(token, block);
        } else {
            dct4x4_scalar(block);
        }
    }
    #[cfg(not(any(target_arch = "wasm32", target_arch = "aarch64")))]
    {
        dct4x4_scalar(block);
    }
}

/// Inverse DCT with dynamic dispatch
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
pub(crate) fn idct4x4_intrinsics(block: &mut [i32; 16]) {
    if let Some(token) = X64V3Token::summon() {
        idct4x4_entry(token, block);
    } else {
        idct4x4_scalar(block);
    }
}

/// Inverse DCT with pre-summoned token (avoids per-call token summoning)
/// Note: Currently unused - decoder uses fused IDCT+add_residue.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn idct4x4_intrinsics_with_token(
    block: &mut [i32; 16],
    simd_token: crate::common::prediction::SimdTokenType,
) {
    if let Some(token) = simd_token {
        idct4x4_entry(token, block);
    } else {
        idct4x4_scalar(block);
    }
}

/// Inverse DCT - non-x86 fallback
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
#[inline]
pub(crate) fn idct4x4_intrinsics(block: &mut [i32; 16]) {
    #[cfg(target_arch = "wasm32")]
    {
        use archmage::{SimdToken, Wasm128Token};
        if let Some(token) = Wasm128Token::summon() {
            idct4x4_wasm(token, block);
        } else {
            idct4x4_scalar(block);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::{NeonToken, SimdToken};
        if let Some(token) = NeonToken::try_new() {
            idct4x4_neon(token, block);
        } else {
            idct4x4_scalar(block);
        }
    }
    #[cfg(not(any(target_arch = "wasm32", target_arch = "aarch64")))]
    {
        idct4x4_scalar(block);
    }
}

/// Inverse DCT with pre-summoned token - non-x86 fallback (ignores token)
/// Note: Currently unused - decoder uses fused IDCT+add_residue.
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(dead_code)]
#[inline]
pub(crate) fn idct4x4_intrinsics_with_token(
    block: &mut [i32; 16],
    _simd_token: crate::common::prediction::SimdTokenType,
) {
    idct4x4_intrinsics(block);
}

/// Process two blocks at once
#[allow(dead_code)]
pub(crate) fn dct4x4_two_intrinsics(block1: &mut [i32; 16], block2: &mut [i32; 16]) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        // SSE4.1 implies SSE2; summon() is now fast (no env var check)
        if let Some(token) = X64V3Token::summon() {
            dct4x4_two_entry(token, block1, block2);
        } else {
            dct4x4_scalar(block1);
            dct4x4_scalar(block2);
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        use archmage::{SimdToken, Wasm128Token};
        if let Some(token) = Wasm128Token::summon() {
            dct4x4_wasm(token, block1);
            dct4x4_wasm(token, block2);
        } else {
            dct4x4_scalar(block1);
            dct4x4_scalar(block2);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::{NeonToken, SimdToken};
        if let Some(token) = NeonToken::try_new() {
            dct4x4_neon(token, block1);
            dct4x4_neon(token, block2);
        } else {
            dct4x4_scalar(block1);
            dct4x4_scalar(block2);
        }
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "wasm32",
        target_arch = "aarch64"
    )))]
    {
        dct4x4_scalar(block1);
        dct4x4_scalar(block2);
    }
}

// =============================================================================
// SSE2 Implementation - matches libwebp's FTransform/ITransform
// =============================================================================

/// Entry shim for dct4x4_sse2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[arcane]
fn dct4x4_entry(_token: X64V3Token, block: &mut [i32; 16]) {
    dct4x4_sse2(_token, block);
}

/// Forward DCT using SSE2 with i16 layout matching libwebp's FTransform
/// Uses _mm_madd_epi16 for efficient multiply-accumulate
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
fn dct4x4_sse2(_token: X64V3Token, block: &mut [i32; 16]) {
    // Convert i32 block to i16 and reorganize into libwebp's interleaved layout:
    // row01 = [r0c0, r0c1, r1c0, r1c1, r0c2, r0c3, r1c2, r1c3]
    // row23 = [r2c0, r2c1, r3c0, r3c1, r2c2, r2c3, r3c2, r3c3]
    let row01 = _mm_set_epi16(
        block[7] as i16, // r1c3
        block[6] as i16, // r1c2
        block[3] as i16, // r0c3
        block[2] as i16, // r0c2
        block[5] as i16, // r1c1
        block[4] as i16, // r1c0
        block[1] as i16, // r0c1
        block[0] as i16, // r0c0
    );
    let row23 = _mm_set_epi16(
        block[15] as i16, // r3c3
        block[14] as i16, // r3c2
        block[11] as i16, // r2c3
        block[10] as i16, // r2c2
        block[13] as i16, // r3c1
        block[12] as i16, // r3c0
        block[9] as i16,  // r2c1
        block[8] as i16,  // r2c0
    );

    // Forward transform pass 1 (rows)
    let (v01, v32) = ftransform_pass1_i16(_token, row01, row23);

    // Forward transform pass 2 (columns)
    let mut out16 = [0i16; 16];
    ftransform_pass2_i16(_token, &v01, &v32, &mut out16);

    // Convert i16 output back to i32 using SIMD sign extension
    let zero = _mm_setzero_si128();
    let (out16_lo, out16_hi) = halves8(&out16);
    let out01 = simd_mem::_mm_loadu_si128(out16_lo);
    let out23 = simd_mem::_mm_loadu_si128(out16_hi);

    // Sign extend i16 to i32
    let sign01 = _mm_cmpgt_epi16(zero, out01);
    let sign23 = _mm_cmpgt_epi16(zero, out23);

    let out_0 = _mm_unpacklo_epi16(out01, sign01);
    let out_1 = _mm_unpackhi_epi16(out01, sign01);
    let out_2 = _mm_unpacklo_epi16(out23, sign23);
    let out_3 = _mm_unpackhi_epi16(out23, sign23);

    let (r0, r1, r2, r3) = rows4_mut(block);
    simd_mem::_mm_storeu_si128(r0, out_0);
    simd_mem::_mm_storeu_si128(r1, out_1);
    simd_mem::_mm_storeu_si128(r2, out_2);
    simd_mem::_mm_storeu_si128(r3, out_3);
}

/// FTransform Pass 1 - matches libwebp FTransformPass1_SSE2 exactly
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
fn ftransform_pass1_i16(_token: X64V3Token, in01: __m128i, in23: __m128i) -> (__m128i, __m128i) {
    // Constants matching libwebp exactly
    let k937 = _mm_set1_epi32(937);
    let k1812 = _mm_set1_epi32(1812);
    let k88p = _mm_set_epi16(8, 8, 8, 8, 8, 8, 8, 8);
    let k88m = _mm_set_epi16(-8, 8, -8, 8, -8, 8, -8, 8);
    let k5352_2217p = _mm_set_epi16(2217, 5352, 2217, 5352, 2217, 5352, 2217, 5352);
    let k5352_2217m = _mm_set_epi16(-5352, 2217, -5352, 2217, -5352, 2217, -5352, 2217);

    // in01 = 00 01 10 11 02 03 12 13
    // in23 = 20 21 30 31 22 23 32 33
    // Shuffle high words: swap pairs in high 64-bits
    let shuf01_p = _mm_shufflehi_epi16(in01, 0b10_11_00_01); // _MM_SHUFFLE(2,3,0,1)
    let shuf23_p = _mm_shufflehi_epi16(in23, 0b10_11_00_01);
    // 00 01 10 11 03 02 13 12
    // 20 21 30 31 23 22 33 32

    // Interleave to separate low/high parts
    let s01 = _mm_unpacklo_epi64(shuf01_p, shuf23_p);
    let s32 = _mm_unpackhi_epi64(shuf01_p, shuf23_p);
    // s01 = 00 01 10 11 20 21 30 31  (columns 0,1 for all rows)
    // s32 = 03 02 13 12 23 22 33 32  (columns 3,2 reversed for all rows)

    // a01 = [d0+d3, d1+d2] for each row pair
    // a32 = [d0-d3, d1-d2] for each row pair
    let a01 = _mm_add_epi16(s01, s32);
    let a32 = _mm_sub_epi16(s01, s32);

    // tmp0 = (a0 + a1) << 3 via madd with [8,8] - produces i32
    // tmp2 = (a0 - a1) << 3 via madd with [-8,8]
    let tmp0 = _mm_madd_epi16(a01, k88p);
    let tmp2 = _mm_madd_epi16(a01, k88m);

    // tmp1 = (a3*5352 + a2*2217 + 1812) >> 9
    // tmp3 = (a3*2217 - a2*5352 + 937) >> 9
    // Note: a32 has [d0-d3, d1-d2] = [a3, a2] for the formulas
    let tmp1_1 = _mm_madd_epi16(a32, k5352_2217p);
    let tmp3_1 = _mm_madd_epi16(a32, k5352_2217m);
    let tmp1_2 = _mm_add_epi32(tmp1_1, k1812);
    let tmp3_2 = _mm_add_epi32(tmp3_1, k937);
    let tmp1 = _mm_srai_epi32(tmp1_2, 9);
    let tmp3 = _mm_srai_epi32(tmp3_2, 9);

    // Pack back to i16
    let s03 = _mm_packs_epi32(tmp0, tmp2); // [out0_r0, out0_r1, out0_r2, out0_r3, out2_r0, ...]
    let s12 = _mm_packs_epi32(tmp1, tmp3); // [out1_r0, out1_r1, out1_r2, out1_r3, out3_r0, ...]

    // Interleave to get proper output order
    let s_lo = _mm_unpacklo_epi16(s03, s12); // 0 1 0 1 0 1 0 1
    let s_hi = _mm_unpackhi_epi16(s03, s12); // 2 3 2 3 2 3 2 3
    let v23 = _mm_unpackhi_epi32(s_lo, s_hi);
    let out01 = _mm_unpacklo_epi32(s_lo, s_hi);
    let out32 = _mm_shuffle_epi32(v23, 0b01_00_11_10); // _MM_SHUFFLE(1,0,3,2)

    (out01, out32)
}

/// FTransform Pass 2 - matches libwebp FTransformPass2_SSE2 exactly
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
fn ftransform_pass2_i16(_token: X64V3Token, v01: &__m128i, v32: &__m128i, out: &mut [i16; 16]) {
    let zero = _mm_setzero_si128();
    let seven = _mm_set1_epi16(7);
    let k5352_2217 = _mm_set_epi16(5352, 2217, 5352, 2217, 5352, 2217, 5352, 2217);
    let k2217_5352 = _mm_set_epi16(2217, -5352, 2217, -5352, 2217, -5352, 2217, -5352);
    let k12000_plus_one = _mm_set1_epi32(12000 + (1 << 16));
    let k51000 = _mm_set1_epi32(51000);

    // a3 = v0 - v3, a2 = v1 - v2
    let a32 = _mm_sub_epi16(*v01, *v32);
    let a22 = _mm_unpackhi_epi64(a32, a32);

    let b23 = _mm_unpacklo_epi16(a22, a32);
    let c1 = _mm_madd_epi16(b23, k5352_2217);
    let c3 = _mm_madd_epi16(b23, k2217_5352);
    let d1 = _mm_add_epi32(c1, k12000_plus_one);
    let d3 = _mm_add_epi32(c3, k51000);
    let e1 = _mm_srai_epi32(d1, 16);
    let e3 = _mm_srai_epi32(d3, 16);

    let f1 = _mm_packs_epi32(e1, e1);
    let f3 = _mm_packs_epi32(e3, e3);

    // g1 = f1 + (a3 != 0) - the +1 is already in k12000_plus_one
    // cmpeq returns 0xFFFF for equal, 0 for not equal
    // Adding 0xFFFF (-1) when equal subtracts 1, which cancels the +1 we added
    let g1 = _mm_add_epi16(f1, _mm_cmpeq_epi16(a32, zero));

    // a0 = v0 + v3, a1 = v1 + v2
    let a01 = _mm_add_epi16(*v01, *v32);
    let a01_plus_7 = _mm_add_epi16(a01, seven);
    let a11 = _mm_unpackhi_epi64(a01, a01);
    let c0 = _mm_add_epi16(a01_plus_7, a11);
    let c2 = _mm_sub_epi16(a01_plus_7, a11);

    let d0 = _mm_srai_epi16(c0, 4);
    let d2 = _mm_srai_epi16(c2, 4);

    // Combine outputs: row 0 and 1 in first register, row 2 and 3 in second
    let d0_g1 = _mm_unpacklo_epi64(d0, g1);
    let d2_f3 = _mm_unpacklo_epi64(d2, f3);

    let (out_lo, out_hi) = halves8_mut(out);
    simd_mem::_mm_storeu_si128(out_lo, d0_g1);
    simd_mem::_mm_storeu_si128(out_hi, d2_f3);
}

/// Entry shim for dct4x4_two_sse2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[arcane]
#[allow(dead_code)]
fn dct4x4_two_entry(_token: X64V3Token, block1: &mut [i32; 16], block2: &mut [i32; 16]) {
    dct4x4_two_sse2(_token, block1, block2);
}

/// Process two blocks at once
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
#[allow(dead_code)]
fn dct4x4_two_sse2(_token: X64V3Token, block1: &mut [i32; 16], block2: &mut [i32; 16]) {
    // Process both blocks using the single-block implementation
    // AVX2 can potentially do this more efficiently with 256-bit registers
    dct4x4_sse2(_token, block1);
    dct4x4_sse2(_token, block2);
}

// =============================================================================
// Fused Residual + DCT (matches libwebp's FTransform2_SSE2)
// =============================================================================

/// Fused residual computation + DCT for two adjacent 4x4 blocks
/// Takes u8 source and reference with stride, outputs i16 coefficients.
/// Matches libwebp's FTransform2_SSE2 for maximum performance.
///
/// # Arguments
/// * `src` - Source pixels (8 bytes per row = 2 blocks side by side)
/// * `ref_` - Reference/prediction pixels (same layout)
/// * `src_stride` - Stride for source rows
/// * `ref_stride` - Stride for reference rows
/// * `out` - Output: 32 i16 coefficients (2 blocks × 16 coeffs)
/// Entry shim for ftransform2_sse2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[arcane]
fn ftransform2_entry(
    _token: X64V3Token,
    src: &[u8],
    ref_: &[u8],
    src_stride: usize,
    ref_stride: usize,
    out: &mut [i16; 32],
) {
    ftransform2_sse2(_token, src, ref_, src_stride, ref_stride, out);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
pub(crate) fn ftransform2_sse2(
    _token: X64V3Token,
    src: &[u8],
    ref_: &[u8],
    src_stride: usize,
    ref_stride: usize,
    out: &mut [i16; 32],
) {
    // One bounds check at entry eliminates all interior checks
    let src_min = src_stride * 3 + 8;
    let ref_min = ref_stride * 3 + 8;
    assert!(src.len() >= src_min && ref_.len() >= ref_min);

    let zero = _mm_setzero_si128();

    // Load 8 bytes per row from src (2 blocks × 4 bytes)
    let src0 =
        simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&src[..8]).unwrap());
    let src1 =
        simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&src[src_stride..src_stride + 8]).unwrap());
    let src2 = simd_mem::_mm_loadu_si64(
        <&[u8; 8]>::try_from(&src[src_stride * 2..src_stride * 2 + 8]).unwrap(),
    );
    let src3 = simd_mem::_mm_loadu_si64(
        <&[u8; 8]>::try_from(&src[src_stride * 3..src_stride * 3 + 8]).unwrap(),
    );

    // Load 8 bytes per row from ref
    let ref0 =
        simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&ref_[..8]).unwrap());
    let ref1 =
        simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&ref_[ref_stride..ref_stride + 8]).unwrap());
    let ref2 = simd_mem::_mm_loadu_si64(
        <&[u8; 8]>::try_from(&ref_[ref_stride * 2..ref_stride * 2 + 8]).unwrap(),
    );
    let ref3 = simd_mem::_mm_loadu_si64(
        <&[u8; 8]>::try_from(&ref_[ref_stride * 3..ref_stride * 3 + 8]).unwrap(),
    );

    // Convert u8 to i16 (zero-extend)
    let src_0 = _mm_unpacklo_epi8(src0, zero);
    let src_1 = _mm_unpacklo_epi8(src1, zero);
    let src_2 = _mm_unpacklo_epi8(src2, zero);
    let src_3 = _mm_unpacklo_epi8(src3, zero);

    let ref_0 = _mm_unpacklo_epi8(ref0, zero);
    let ref_1 = _mm_unpacklo_epi8(ref1, zero);
    let ref_2 = _mm_unpacklo_epi8(ref2, zero);
    let ref_3 = _mm_unpacklo_epi8(ref3, zero);

    // Compute difference (src - ref)
    // Each diff register holds: [blk0_col0, blk0_col1, blk0_col2, blk0_col3, blk1_col0, ...]
    let diff0 = _mm_sub_epi16(src_0, ref_0);
    let diff1 = _mm_sub_epi16(src_1, ref_1);
    let diff2 = _mm_sub_epi16(src_2, ref_2);
    let diff3 = _mm_sub_epi16(src_3, ref_3);

    // Reorganize for processing two blocks
    // shuf01l/h splits low/high halves (block 0 and block 1)
    let shuf01l = _mm_unpacklo_epi32(diff0, diff1);
    let shuf23l = _mm_unpacklo_epi32(diff2, diff3);
    let shuf01h = _mm_unpackhi_epi32(diff0, diff1);
    let shuf23h = _mm_unpackhi_epi32(diff2, diff3);

    // First pass for block 0
    let (v01l, v32l) = ftransform_pass1_i16(_token, shuf01l, shuf23l);

    // First pass for block 1
    let (v01h, v32h) = ftransform_pass1_i16(_token, shuf01h, shuf23h);

    // Second pass for both blocks
    let mut out0 = [0i16; 16];
    let mut out1 = [0i16; 16];
    ftransform_pass2_i16(_token, &v01l, &v32l, &mut out0);
    ftransform_pass2_i16(_token, &v01h, &v32h, &mut out1);

    // Copy to output
    out[..16].copy_from_slice(&out0);
    out[16..].copy_from_slice(&out1);
}

/// Public dispatch function for fused residual + DCT of two adjacent blocks
pub(crate) fn ftransform2_from_u8(
    src: &[u8],
    ref_: &[u8],
    src_stride: usize,
    ref_stride: usize,
    out: &mut [i16; 32],
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if let Some(token) = X64V3Token::summon() {
            ftransform2_entry(token, src, ref_, src_stride, ref_stride, out);
            return;
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        use archmage::{SimdToken, Wasm128Token};
        if let Some(token) = Wasm128Token::summon() {
            // Process two blocks using single-block WASM DCT
            for block in 0..2 {
                let mut block_data = [0i32; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        let src_val = src[y * src_stride + block * 4 + x] as i32;
                        let ref_val = ref_[y * ref_stride + block * 4 + x] as i32;
                        block_data[y * 4 + x] = src_val - ref_val;
                    }
                }
                dct4x4_wasm(token, &mut block_data);
                for (i, &val) in block_data.iter().enumerate() {
                    out[block * 16 + i] = val as i16;
                }
            }
            return;
        }
    }
    // Scalar fallback
    ftransform2_scalar(src, ref_, src_stride, ref_stride, out);
}

/// Scalar fallback for fused residual + DCT
fn ftransform2_scalar(
    src: &[u8],
    ref_: &[u8],
    src_stride: usize,
    ref_stride: usize,
    out: &mut [i16; 32],
) {
    // Process two blocks
    for block in 0..2 {
        let mut block_data = [0i32; 16];
        // Compute residual for this 4x4 block
        for y in 0..4 {
            for x in 0..4 {
                let src_val = src[y * src_stride + block * 4 + x] as i32;
                let ref_val = ref_[y * ref_stride + block * 4 + x] as i32;
                block_data[y * 4 + x] = src_val - ref_val;
            }
        }
        // Apply DCT
        dct4x4_scalar(&mut block_data);
        // Convert to i16
        for (i, &val) in block_data.iter().enumerate() {
            out[block * 16 + i] = val as i16;
        }
    }
}

/// Entry shim for idct4x4_sse2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[arcane]
pub(crate) fn idct4x4_entry(_token: X64V3Token, block: &mut [i32; 16]) {
    idct4x4_sse2(_token, block);
}

/// Inverse DCT using SSE2 - matches libwebp ITransform_One_SSE2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
pub(crate) fn idct4x4_sse2(_token: X64V3Token, block: &mut [i32; 16]) {
    // libwebp IDCT constants:
    // K1 = sqrt(2) * cos(pi/8) * 65536 = 85627 => k1 = 85627 - 65536 = 20091
    // K2 = sqrt(2) * sin(pi/8) * 65536 = 35468 => k2 = 35468 - 65536 = -30068
    let k1k2 = _mm_set_epi16(-30068, -30068, -30068, -30068, 20091, 20091, 20091, 20091);
    let k2k1 = _mm_set_epi16(20091, 20091, 20091, 20091, -30068, -30068, -30068, -30068);
    let zero_four = _mm_set_epi16(0, 0, 0, 0, 4, 4, 4, 4);

    // Load i32 values and pack to i16 using SIMD
    let (r0, r1, r2, r3) = rows4(block);
    let i32_0 = simd_mem::_mm_loadu_si128(r0);
    let i32_1 = simd_mem::_mm_loadu_si128(r1);
    let i32_2 = simd_mem::_mm_loadu_si128(r2);
    let i32_3 = simd_mem::_mm_loadu_si128(r3);

    // Pack i32 to i16 with saturation (values should be small enough)
    let in01 = _mm_packs_epi32(i32_0, i32_1);
    let in23 = _mm_packs_epi32(i32_2, i32_3);

    // Vertical pass
    let (t01, t23) = itransform_pass_sse2(_token, in01, in23, k1k2, k2k1);

    // Horizontal pass with rounding
    let (out01, out23) = itransform_pass2_sse2(_token, t01, t23, k1k2, k2k1, zero_four);

    // Unpack i16 back to i32 using sign extension
    let zero = _mm_setzero_si128();

    // Sign extend low 4 i16 values to i32
    let sign01_lo = _mm_cmpgt_epi16(zero, out01); // all 1s where negative
    let out_0 = _mm_unpacklo_epi16(out01, sign01_lo);
    let out_1 = _mm_unpackhi_epi16(out01, sign01_lo);

    let sign23_lo = _mm_cmpgt_epi16(zero, out23);
    let out_2 = _mm_unpacklo_epi16(out23, sign23_lo);
    let out_3 = _mm_unpackhi_epi16(out23, sign23_lo);

    // Store results
    let (r0, r1, r2, r3) = rows4_mut(block);
    simd_mem::_mm_storeu_si128(r0, out_0);
    simd_mem::_mm_storeu_si128(r1, out_1);
    simd_mem::_mm_storeu_si128(r2, out_2);
    simd_mem::_mm_storeu_si128(r3, out_3);
}

/// ITransform vertical pass - matches libwebp
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
fn itransform_pass_sse2(
    _token: X64V3Token,
    in01: __m128i,
    in23: __m128i,
    k1k2: __m128i,
    k2k1: __m128i,
) -> (__m128i, __m128i) {
    // in01 = a00 a10 a20 a30 | a01 a11 a21 a31
    // in23 = a02 a12 a22 a32 | a03 a13 a23 a33

    let in1 = _mm_unpackhi_epi64(in01, in01);
    let in3 = _mm_unpackhi_epi64(in23, in23);

    // a = in0 + in2, b = in0 - in2
    let a_d3 = _mm_add_epi16(in01, in23);
    let b_c3 = _mm_sub_epi16(in01, in23);

    // c = MUL(in1, K2) - MUL(in3, K1)
    // d = MUL(in1, K1) + MUL(in3, K2)
    // Using the trick: MUL(x, K) = x + MUL(x, k) where K = k + 65536
    let c1d1 = _mm_mulhi_epi16(in1, k2k1);
    let c2d2 = _mm_mulhi_epi16(in3, k1k2);
    let c3 = _mm_unpackhi_epi64(b_c3, b_c3);
    let c4 = _mm_sub_epi16(c1d1, c2d2);
    let c = _mm_add_epi16(c3, c4);
    let d4u = _mm_add_epi16(c1d1, c2d2);
    let du = _mm_add_epi16(a_d3, d4u);
    let d = _mm_unpackhi_epi64(du, du);

    // Combine and transpose
    let comb_ab = _mm_unpacklo_epi64(a_d3, b_c3);
    let comb_dc = _mm_unpacklo_epi64(d, c);

    let tmp01 = _mm_add_epi16(comb_ab, comb_dc);
    let tmp32 = _mm_sub_epi16(comb_ab, comb_dc);
    let tmp23 = _mm_shuffle_epi32(tmp32, 0b01_00_11_10);

    let transpose_0 = _mm_unpacklo_epi16(tmp01, tmp23);
    let transpose_1 = _mm_unpackhi_epi16(tmp01, tmp23);

    let t01 = _mm_unpacklo_epi16(transpose_0, transpose_1);
    let t23 = _mm_unpackhi_epi16(transpose_0, transpose_1);

    (t01, t23)
}

/// ITransform horizontal pass with final shift
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
fn itransform_pass2_sse2(
    _token: X64V3Token,
    t01: __m128i,
    t23: __m128i,
    k1k2: __m128i,
    k2k1: __m128i,
    zero_four: __m128i,
) -> (__m128i, __m128i) {
    let t1 = _mm_unpackhi_epi64(t01, t01);
    let t3 = _mm_unpackhi_epi64(t23, t23);

    // Add rounding constant to DC
    let dc = _mm_add_epi16(t01, zero_four);

    let a_d3 = _mm_add_epi16(dc, t23);
    let b_c3 = _mm_sub_epi16(dc, t23);

    let c1d1 = _mm_mulhi_epi16(t1, k2k1);
    let c2d2 = _mm_mulhi_epi16(t3, k1k2);
    let c3 = _mm_unpackhi_epi64(b_c3, b_c3);
    let c4 = _mm_sub_epi16(c1d1, c2d2);
    let c = _mm_add_epi16(c3, c4);
    let d4u = _mm_add_epi16(c1d1, c2d2);
    let du = _mm_add_epi16(a_d3, d4u);
    let d = _mm_unpackhi_epi64(du, du);

    let comb_ab = _mm_unpacklo_epi64(a_d3, b_c3);
    let comb_dc = _mm_unpacklo_epi64(d, c);

    let tmp01 = _mm_add_epi16(comb_ab, comb_dc);
    let tmp32 = _mm_sub_epi16(comb_ab, comb_dc);
    let tmp23 = _mm_shuffle_epi32(tmp32, 0b01_00_11_10);

    // Final shift >> 3
    let shifted01 = _mm_srai_epi16(tmp01, 3);
    let shifted23 = _mm_srai_epi16(tmp23, 3);

    // Transpose back
    let transpose_0 = _mm_unpacklo_epi16(shifted01, shifted23);
    let transpose_1 = _mm_unpackhi_epi16(shifted01, shifted23);

    let out01 = _mm_unpacklo_epi16(transpose_0, transpose_1);
    let out23 = _mm_unpackhi_epi16(transpose_0, transpose_1);

    (out01, out23)
}

// =============================================================================
// Fused IDCT + Add Residue (like libwebp's ITransform_SSE2)
// =============================================================================

/// Fused IDCT + add residue + clamp, writing directly to output buffer.
/// This is the hot path optimization - avoids separate IDCT and add_residue calls.
///
/// Takes: coefficients (i32[16]), prediction block, output location
/// Does: IDCT(coeffs) + prediction → output (clamped to 0-255)
/// Also clears the coefficient block to zeros.
/// Entry shim for idct_add_residue_sse2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[allow(clippy::too_many_arguments)]
#[arcane]
pub(crate) fn idct_add_residue_entry(
    _token: X64V3Token,
    coeffs: &mut [i32; 16],
    pred_block: &[u8],
    pred_stride: usize,
    out_block: &mut [u8],
    out_stride: usize,
    pred_y0: usize,
    pred_x0: usize,
    out_y0: usize,
    out_x0: usize,
) {
    idct_add_residue_sse2(
        _token,
        coeffs,
        pred_block,
        pred_stride,
        out_block,
        out_stride,
        pred_y0,
        pred_x0,
        out_y0,
        out_x0,
    );
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
#[allow(clippy::too_many_arguments)]
pub(crate) fn idct_add_residue_sse2(
    _token: X64V3Token,
    coeffs: &mut [i32; 16],
    pred_block: &[u8],
    pred_stride: usize,
    out_block: &mut [u8],
    out_stride: usize,
    pred_y0: usize,
    pred_x0: usize,
    out_y0: usize,
    out_x0: usize,
) {
    // IDCT constants
    let k1k2 = _mm_set_epi16(-30068, -30068, -30068, -30068, 20091, 20091, 20091, 20091);
    let k2k1 = _mm_set_epi16(20091, 20091, 20091, 20091, -30068, -30068, -30068, -30068);
    let zero_four = _mm_set_epi16(0, 0, 0, 0, 4, 4, 4, 4);
    let zero = _mm_setzero_si128();

    // Load i32 coefficients and pack to i16
    let (c0, c1, c2, c3) = rows4(coeffs);
    let i32_0 = simd_mem::_mm_loadu_si128(c0);
    let i32_1 = simd_mem::_mm_loadu_si128(c1);
    let i32_2 = simd_mem::_mm_loadu_si128(c2);
    let i32_3 = simd_mem::_mm_loadu_si128(c3);

    let in01 = _mm_packs_epi32(i32_0, i32_1);
    let in23 = _mm_packs_epi32(i32_2, i32_3);

    // Vertical pass
    let (t01, t23) = itransform_pass_sse2(_token, in01, in23, k1k2, k2k1);

    // Horizontal pass with rounding → residual in i16
    let (res01, res23) = itransform_pass2_sse2(_token, t01, t23, k1k2, k2k1, zero_four);

    // res01 = row0[0-3] row1[0-3] as i16
    // res23 = row2[0-3] row3[0-3] as i16

    // Load 4 prediction bytes per row, extend to i16, add residual, pack to u8
    macro_rules! process_row {
        ($res:expr, $row_idx:expr, $hi:expr) => {{
            // Extract this row's residuals (4 x i16)
            let residual = if $hi {
                _mm_unpackhi_epi64($res, $res)
            } else {
                $res
            };

            // Load 4 prediction bytes
            let pred_pos = (pred_y0 + $row_idx) * pred_stride + pred_x0;
            let pred_bytes: [u8; 4] = pred_block[pred_pos..pred_pos + 4].try_into().unwrap();
            let pred_i32 = i32::from_ne_bytes(pred_bytes);
            let pred_vec = _mm_cvtsi32_si128(pred_i32);
            // Extend u8 → i16 (zero extend, then treat as signed for addition)
            let pred_i16 = _mm_unpacklo_epi8(pred_vec, zero);

            // Add residual to prediction (i16 + i16 → i16)
            let sum = _mm_add_epi16(pred_i16, residual);

            // Pack to u8 with saturation (clamp to 0-255)
            let packed = _mm_packus_epi16(sum, sum);

            // Store 4 bytes to output
            let out_pos = (out_y0 + $row_idx) * out_stride + out_x0;
            let result = _mm_cvtsi128_si32(packed) as u32;
            out_block[out_pos..out_pos + 4].copy_from_slice(&result.to_ne_bytes());
        }};
    }

    process_row!(res01, 0, false);
    process_row!(res01, 1, true);
    process_row!(res23, 2, false);
    process_row!(res23, 3, true);

    // Clear coefficient block
    let (c0, c1, c2, c3) = rows4_mut(coeffs);
    simd_mem::_mm_storeu_si128(c0, zero);
    simd_mem::_mm_storeu_si128(c1, zero);
    simd_mem::_mm_storeu_si128(c2, zero);
    simd_mem::_mm_storeu_si128(c3, zero);
}

/// Entry shim for idct_add_residue_dc_sse2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[allow(clippy::too_many_arguments)]
#[arcane]
pub(crate) fn idct_add_residue_dc_entry(
    _token: X64V3Token,
    coeffs: &mut [i32; 16],
    pred_block: &[u8],
    pred_stride: usize,
    out_block: &mut [u8],
    out_stride: usize,
    pred_y0: usize,
    pred_x0: usize,
    out_y0: usize,
    out_x0: usize,
) {
    idct_add_residue_dc_sse2(
        _token,
        coeffs,
        pred_block,
        pred_stride,
        out_block,
        out_stride,
        pred_y0,
        pred_x0,
        out_y0,
        out_x0,
    );
}

/// DC-only fused IDCT + add residue - when only DC coefficient is non-zero.
/// Much faster than full IDCT.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
#[allow(clippy::too_many_arguments)]
pub(crate) fn idct_add_residue_dc_sse2(
    _token: X64V3Token,
    coeffs: &mut [i32; 16],
    pred_block: &[u8],
    pred_stride: usize,
    out_block: &mut [u8],
    out_stride: usize,
    pred_y0: usize,
    pred_x0: usize,
    out_y0: usize,
    out_x0: usize,
) {
    // DC-only: output = prediction + ((DC + 4) >> 3) for all 16 pixels
    let dc = coeffs[0];
    let dc_adj = ((dc + 4) >> 3) as i16;
    let dc_vec = _mm_set1_epi16(dc_adj);
    let zero = _mm_setzero_si128();

    for row in 0..4 {
        // Load 4 prediction bytes
        let pred_pos = (pred_y0 + row) * pred_stride + pred_x0;
        let pred_bytes: [u8; 4] = pred_block[pred_pos..pred_pos + 4].try_into().unwrap();
        let pred_i32 = i32::from_ne_bytes(pred_bytes);
        let pred_vec = _mm_cvtsi32_si128(pred_i32);
        let pred_i16 = _mm_unpacklo_epi8(pred_vec, zero);

        // Add DC to prediction
        let sum = _mm_add_epi16(pred_i16, dc_vec);

        // Pack to u8 with saturation
        let packed = _mm_packus_epi16(sum, sum);

        // Store 4 bytes
        let out_pos = (out_y0 + row) * out_stride + out_x0;
        let result = _mm_cvtsi128_si32(packed) as u32;
        out_block[out_pos..out_pos + 4].copy_from_slice(&result.to_ne_bytes());
    }

    // Clear coefficient block
    coeffs.fill(0);
}

/// Wrapper with token type parameter for decoder use (separate pred/output)
/// Note: Currently unused - decoder uses in-place version instead.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[allow(dead_code, clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn idct_add_residue_with_token(
    token: X64V3Token,
    coeffs: &mut [i32; 16],
    pred_block: &[u8],
    pred_stride: usize,
    out_block: &mut [u8],
    out_stride: usize,
    pred_y0: usize,
    pred_x0: usize,
    out_y0: usize,
    out_x0: usize,
    dc_only: bool,
) {
    if dc_only {
        idct_add_residue_dc_entry(
            token,
            coeffs,
            pred_block,
            pred_stride,
            out_block,
            out_stride,
            pred_y0,
            pred_x0,
            out_y0,
            out_x0,
        );
    } else {
        idct_add_residue_entry(
            token,
            coeffs,
            pred_block,
            pred_stride,
            out_block,
            out_stride,
            pred_y0,
            pred_x0,
            out_y0,
            out_x0,
        );
    }
}

/// In-place fused IDCT + add residue for decoder hot path (entry point).
/// Wrapper for callers that don't have a target_feature context.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[arcane]
pub(crate) fn idct_add_residue_inplace_sse2(
    _token: X64V3Token,
    coeffs: &mut [i32; 16],
    block: &mut [u8],
    y0: usize,
    x0: usize,
    stride: usize,
    dc_only: bool,
) {
    idct_add_residue_inplace_sse2_inner(_token, coeffs, block, y0, x0, stride, dc_only);
}

/// In-place fused IDCT + add residue — `#[rite]` version for inlining into
/// `#[arcane]` callers (e.g., the prediction+IDCT pipeline).
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
pub(crate) fn idct_add_residue_inplace_sse2_inner(
    _token: X64V3Token,
    coeffs: &mut [i32; 16],
    block: &mut [u8],
    y0: usize,
    x0: usize,
    stride: usize,
    dc_only: bool,
) {
    // Subslice starting at first pixel row; single bounds check eliminates all interior checks.
    // The region covers 3*stride+4 bytes (4 rows of 4 pixels at stride offsets).
    let base = y0 * stride + x0;
    let region = &mut block[base..base + 3 * stride + 4];
    let s1 = stride;
    let s2 = stride * 2;
    let s3 = stride * 3;

    if dc_only {
        // DC-only fast path
        let dc = coeffs[0];
        let dc_adj = ((dc + 4) >> 3) as i16;
        let dc_vec = _mm_set1_epi16(dc_adj);
        let zero = _mm_setzero_si128();

        for &off in &[0, s1, s2, s3] {
            let pred_bytes: [u8; 4] = region[off..off + 4].try_into().unwrap();
            let pred_vec = _mm_cvtsi32_si128(i32::from_ne_bytes(pred_bytes));
            let pred_i16 = _mm_unpacklo_epi8(pred_vec, zero);
            let sum = _mm_add_epi16(pred_i16, dc_vec);
            let packed = _mm_packus_epi16(sum, sum);
            let result = _mm_cvtsi128_si32(packed) as u32;
            region[off..off + 4].copy_from_slice(&result.to_ne_bytes());
        }
    } else {
        // Full IDCT path
        let k1k2 = _mm_set_epi16(-30068, -30068, -30068, -30068, 20091, 20091, 20091, 20091);
        let k2k1 = _mm_set_epi16(20091, 20091, 20091, 20091, -30068, -30068, -30068, -30068);
        let zero_four = _mm_set_epi16(0, 0, 0, 0, 4, 4, 4, 4);
        let zero = _mm_setzero_si128();

        // Load and pack coefficients
        let (c0, c1, c2, c3) = rows4(coeffs);
        let i32_0 = simd_mem::_mm_loadu_si128(c0);
        let i32_1 = simd_mem::_mm_loadu_si128(c1);
        let i32_2 = simd_mem::_mm_loadu_si128(c2);
        let i32_3 = simd_mem::_mm_loadu_si128(c3);

        let in01 = _mm_packs_epi32(i32_0, i32_1);
        let in23 = _mm_packs_epi32(i32_2, i32_3);

        // Vertical + horizontal passes → residuals in i16
        let (t01, t23) = itransform_pass_sse2(_token, in01, in23, k1k2, k2k1);
        let (res01, res23) = itransform_pass2_sse2(_token, t01, t23, k1k2, k2k1, zero_four);

        // Process each row: load pred, add residual, store
        macro_rules! process_row {
            ($res:expr, $off:expr, $hi:expr) => {{
                let residual = if $hi {
                    _mm_unpackhi_epi64($res, $res)
                } else {
                    $res
                };
                let pred_bytes: [u8; 4] = region[$off..$off + 4].try_into().unwrap();
                let pred_vec = _mm_cvtsi32_si128(i32::from_ne_bytes(pred_bytes));
                let pred_i16 = _mm_unpacklo_epi8(pred_vec, zero);
                let sum = _mm_add_epi16(pred_i16, residual);
                let packed = _mm_packus_epi16(sum, sum);
                let result = _mm_cvtsi128_si32(packed) as u32;
                region[$off..$off + 4].copy_from_slice(&result.to_ne_bytes());
            }};
        }

        process_row!(res01, 0, false);
        process_row!(res01, s1, true);
        process_row!(res23, s2, false);
        process_row!(res23, s3, true);
    }

    // Clear coefficient block
    coeffs.fill(0);
}

/// Wrapper for in-place fused IDCT + add residue
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
pub(crate) fn idct_add_residue_inplace_with_token(
    token: X64V3Token,
    coeffs: &mut [i32; 16],
    block: &mut [u8],
    y0: usize,
    x0: usize,
    stride: usize,
    dc_only: bool,
) {
    idct_add_residue_inplace_sse2(token, coeffs, block, y0, x0, stride, dc_only);
}

// =============================================================================
// In-place fused IDCT + add_residue for encoder
// =============================================================================

/// In-place fused IDCT + add residue with dispatch (no token needed).
/// Reads prediction from `block[y0*stride+x0..]`, performs IDCT on coefficients,
/// adds residual to prediction, clamps 0-255, stores back, and clears coefficients.
///
/// DC-only fast path when `dc_only` is true (skips full IDCT, just adds constant).
#[inline(always)]
pub(crate) fn idct_add_residue_inplace(
    coeffs: &mut [i32; 16],
    block: &mut [u8],
    y0: usize,
    x0: usize,
    stride: usize,
    dc_only: bool,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if let Some(token) = X64V3Token::summon() {
            idct_add_residue_inplace_sse2(token, coeffs, block, y0, x0, stride, dc_only);
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::{NeonToken, SimdToken};
        if let Some(token) = NeonToken::summon() {
            idct_add_residue_inplace_neon(token, coeffs, block, y0, x0, stride, dc_only);
            return;
        }
    }
    // Scalar fallback: IDCT then add
    if dc_only {
        let dc = coeffs[0];
        let dc_adj = (dc + 4) >> 3;
        for row in 0..4 {
            let pos = (y0 + row) * stride + x0;
            for col in 0..4 {
                let p = block[pos + col] as i32;
                block[pos + col] = (p + dc_adj).clamp(0, 255) as u8;
            }
        }
    } else {
        idct4x4_scalar(coeffs);
        let mut pos = y0 * stride + x0;
        for row in coeffs.chunks(4) {
            for (p, &a) in block[pos..][..4].iter_mut().zip(row.iter()) {
                *p = (a + i32::from(*p)).clamp(0, 255) as u8;
            }
            pos += stride;
        }
    }
    coeffs.fill(0);
}

// =============================================================================
// Fused Residual + DCT for single 4x4 block from flat u8 arrays
// =============================================================================

/// Fused residual computation + DCT for a single 4x4 block.
/// Takes flat u8 source and reference arrays (stride=4), outputs i32 coefficients.
///
/// Replaces: manual residual loop + dct4x4() in I4 inner loop.
/// Benefit: avoids i32 intermediate for residuals; computes directly in i16.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
pub(crate) fn ftransform_from_u8_4x4(src: &[u8; 16], ref_: &[u8; 16]) -> [i32; 16] {
    if let Some(token) = X64V3Token::summon() {
        ftransform_from_u8_4x4_entry(token, src, ref_)
    } else {
        ftransform_from_u8_4x4_scalar(src, ref_)
    }
}

/// Non-x86 fallback: try NEON on aarch64, else scalar
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
#[inline(always)]
pub(crate) fn ftransform_from_u8_4x4(src: &[u8; 16], ref_: &[u8; 16]) -> [i32; 16] {
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::{NeonToken, SimdToken};
        if let Some(token) = NeonToken::summon() {
            return ftransform_from_u8_4x4_neon(token, src, ref_);
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        use archmage::{SimdToken, Wasm128Token};
        if let Some(token) = Wasm128Token::summon() {
            return ftransform_from_u8_4x4_wasm(token, src, ref_);
        }
    }
    ftransform_from_u8_4x4_scalar(src, ref_)
}

/// Scalar implementation of fused residual+DCT
pub(crate) fn ftransform_from_u8_4x4_scalar(src: &[u8; 16], ref_: &[u8; 16]) -> [i32; 16] {
    let mut block = [0i32; 16];
    for i in 0..16 {
        block[i] = src[i] as i32 - ref_[i] as i32;
    }
    dct4x4_scalar(&mut block);
    block
}

/// Entry shim for ftransform_from_u8_4x4_sse2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[arcane]
fn ftransform_from_u8_4x4_entry(_token: X64V3Token, src: &[u8; 16], ref_: &[u8; 16]) -> [i32; 16] {
    ftransform_from_u8_4x4_sse2(_token, src, ref_)
}

/// SSE2 fused residual+DCT for single 4x4 block from flat u8[16] arrays.
/// Loads bytes, computes diff as i16, runs DCT pass1+pass2, outputs i32[16].
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
pub(crate) fn ftransform_from_u8_4x4_sse2(
    _token: X64V3Token,
    src: &[u8; 16],
    ref_: &[u8; 16],
) -> [i32; 16] {
    let zero = _mm_setzero_si128();

    // Load all 16 source bytes and 16 reference bytes as single 128-bit loads
    let src_all = simd_mem::_mm_loadu_si128(src);
    let ref_all = simd_mem::_mm_loadu_si128(ref_);

    // Zero-extend u8 to i16
    let src_lo = _mm_unpacklo_epi8(src_all, zero); // src[0..8] as i16
    let src_hi = _mm_unpackhi_epi8(src_all, zero); // src[8..16] as i16
    let ref_lo = _mm_unpacklo_epi8(ref_all, zero); // ref[0..8] as i16
    let ref_hi = _mm_unpackhi_epi8(ref_all, zero); // ref[8..16] as i16

    // Compute difference (src - ref) as i16 (range -255..255, fits i16)
    let diff_lo = _mm_sub_epi16(src_lo, ref_lo); // diff[0..8]
    let diff_hi = _mm_sub_epi16(src_hi, ref_hi); // diff[8..16]

    // diff_lo = [d00, d01, d02, d03, d10, d11, d12, d13]
    // diff_hi = [d20, d21, d22, d23, d30, d31, d32, d33]
    // Reorganize into libwebp's interleaved layout for ftransform_pass1_i16:
    // in01 = [r0c0, r0c1, r1c0, r1c1, r0c2, r0c3, r1c2, r1c3]
    // in23 = [r2c0, r2c1, r3c0, r3c1, r2c2, r2c3, r3c2, r3c3]

    // diff_lo already has rows 0,1 interleaved as pairs of 32-bit (2 i16 per pair)
    // Need: [d00,d01,d10,d11, d02,d03,d12,d13]
    // diff_lo = [d00,d01,d02,d03, d10,d11,d12,d13]
    // Interleave 32-bit words: unpacklo_epi32 + unpackhi_epi32
    let in01 = _mm_unpacklo_epi32(diff_lo, _mm_unpackhi_epi64(diff_lo, diff_lo));
    // = [d00,d01, d10,d11, d02,d03, d12,d13] ✓

    // Same for rows 2,3
    let in23 = _mm_unpacklo_epi32(diff_hi, _mm_unpackhi_epi64(diff_hi, diff_hi));
    // = [d20,d21, d30,d31, d22,d23, d32,d33] ✓

    // Forward transform pass 1 (rows)
    let (v01, v32) = ftransform_pass1_i16(_token, in01, in23);

    // Forward transform pass 2 (columns)
    let mut out16 = [0i16; 16];
    ftransform_pass2_i16(_token, &v01, &v32, &mut out16);

    // Convert i16 output to i32 using SIMD sign extension
    let (out16_lo, out16_hi) = halves8(&out16);
    let out01 = simd_mem::_mm_loadu_si128(out16_lo);
    let out23 = simd_mem::_mm_loadu_si128(out16_hi);

    let sign01 = _mm_cmpgt_epi16(zero, out01);
    let sign23 = _mm_cmpgt_epi16(zero, out23);

    let out_0 = _mm_unpacklo_epi16(out01, sign01);
    let out_1 = _mm_unpackhi_epi16(out01, sign01);
    let out_2 = _mm_unpacklo_epi16(out23, sign23);
    let out_3 = _mm_unpackhi_epi16(out23, sign23);

    let mut result = [0i32; 16];
    let (r0, r1, r2, r3) = rows4_mut(&mut result);
    simd_mem::_mm_storeu_si128(r0, out_0);
    simd_mem::_mm_storeu_si128(r1, out_1);
    simd_mem::_mm_storeu_si128(r2, out_2);
    simd_mem::_mm_storeu_si128(r3, out_3);
    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests_simd {
    use super::*;

    #[test]
    fn test_dct_intrinsics_matches_scalar() {
        let input: [i32; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];

        let mut scalar_block = input;
        dct4x4_scalar(&mut scalar_block);

        let mut intrinsics_block = input;
        dct4x4_intrinsics(&mut intrinsics_block);

        assert_eq!(
            scalar_block, intrinsics_block,
            "Intrinsics DCT doesn't match scalar.\nScalar: {:?}\nIntrinsics: {:?}",
            scalar_block, intrinsics_block
        );
    }

    #[test]
    fn test_idct_intrinsics_matches_scalar() {
        let mut input: [i32; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];
        dct4x4_scalar(&mut input);

        let mut scalar_block = input;
        idct4x4_scalar(&mut scalar_block);

        let mut intrinsics_block = input;
        idct4x4_intrinsics(&mut intrinsics_block);

        assert_eq!(
            scalar_block, intrinsics_block,
            "Intrinsics IDCT doesn't match scalar.\nScalar: {:?}\nIntrinsics: {:?}",
            scalar_block, intrinsics_block
        );
    }

    #[test]
    fn test_dct_two_intrinsics() {
        let input1: [i32; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];
        let input2: [i32; 16] = [
            100, 50, 25, 75, 200, 150, 100, 50, 25, 75, 125, 175, 225, 200, 150, 100,
        ];

        let mut scalar1 = input1;
        let mut scalar2 = input2;
        dct4x4_scalar(&mut scalar1);
        dct4x4_scalar(&mut scalar2);

        let mut intrinsics1 = input1;
        let mut intrinsics2 = input2;
        dct4x4_two_intrinsics(&mut intrinsics1, &mut intrinsics2);

        assert_eq!(scalar1, intrinsics1, "Two-block intrinsics block1 mismatch");
        assert_eq!(scalar2, intrinsics2, "Two-block intrinsics block2 mismatch");
    }

    #[test]
    fn test_roundtrip() {
        let original: [i32; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];

        let mut block = original;
        dct4x4_intrinsics(&mut block);
        idct4x4_intrinsics(&mut block);

        // Should get back original (or very close due to rounding)
        assert_eq!(original, block, "Roundtrip failed");
    }

    #[test]
    fn test_ftransform_from_u8_4x4() {
        // Create test data: flat 4x4 blocks
        let src: [u8; 16] = [
            100, 108, 116, 124, 132, 140, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220,
        ];
        let ref_: [u8; 16] = [128; 16];

        // Use fused function
        let simd_result = ftransform_from_u8_4x4(&src, &ref_);

        // Compute expected result using scalar path
        let mut expected = [0i32; 16];
        for i in 0..16 {
            expected[i] = src[i] as i32 - ref_[i] as i32;
        }
        dct4x4_scalar(&mut expected);

        assert_eq!(
            simd_result, expected,
            "ftransform_from_u8_4x4 mismatch.\nSIMD: {:?}\nExpected: {:?}",
            simd_result, expected
        );
    }

    #[test]
    fn test_ftransform_from_u8_4x4_varied() {
        // Test with varied pixel values
        let src: [u8; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];
        let ref_: [u8; 16] = [
            100, 50, 200, 80, 60, 130, 170, 140, 230, 210, 120, 220, 200, 20, 70, 110,
        ];

        let simd_result = ftransform_from_u8_4x4(&src, &ref_);

        let mut expected = [0i32; 16];
        for i in 0..16 {
            expected[i] = src[i] as i32 - ref_[i] as i32;
        }
        dct4x4_scalar(&mut expected);

        assert_eq!(
            simd_result, expected,
            "ftransform_from_u8_4x4 varied mismatch.\nSIMD: {:?}\nExpected: {:?}",
            simd_result, expected
        );
    }

    #[test]
    fn test_ftransform2_from_u8() {
        // Create test data: 2 adjacent 4x4 blocks (8 bytes per row)
        // Each row is 8 bytes, 4 rows total
        const STRIDE: usize = 16; // Use larger stride to test stride handling
        let mut src = [128u8; STRIDE * 4];
        let mut ref_ = [128u8; STRIDE * 4];

        // Set up some test pattern in first 8 columns of each row
        for y in 0..4 {
            for x in 0..8 {
                src[y * STRIDE + x] = (y * 8 + x) as u8 + 100;
                ref_[y * STRIDE + x] = 128;
            }
        }

        // Use ftransform2_from_u8
        let mut out_simd = [0i16; 32];
        ftransform2_from_u8(&src, &ref_, STRIDE, STRIDE, &mut out_simd);

        // Compute expected result using scalar path
        let mut expected = [0i16; 32];
        for block in 0..2 {
            let mut block_data = [0i32; 16];
            for y in 0..4 {
                for x in 0..4 {
                    let src_val = src[y * STRIDE + block * 4 + x] as i32;
                    let ref_val = ref_[y * STRIDE + block * 4 + x] as i32;
                    block_data[y * 4 + x] = src_val - ref_val;
                }
            }
            dct4x4_scalar(&mut block_data);
            for (i, &val) in block_data.iter().enumerate() {
                expected[block * 16 + i] = val as i16;
            }
        }

        assert_eq!(
            out_simd, expected,
            "ftransform2_from_u8 mismatch.\nSIMD: {:?}\nExpected: {:?}",
            out_simd, expected
        );
    }
}

#[cfg(all(test, feature = "_benchmarks"))]
mod benchmarks {
    use super::*;
    use test::Bencher;

    const TEST_BLOCKS: [[i32; 16]; 4] = [
        [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ],
        [
            100, 50, 25, 75, 200, 150, 100, 50, 25, 75, 125, 175, 225, 200, 150, 100,
        ],
        [
            12, 34, 56, 78, 90, 12, 34, 56, 78, 90, 12, 34, 56, 78, 90, 12,
        ],
        [
            255, 0, 128, 64, 192, 32, 224, 16, 240, 8, 248, 4, 252, 2, 254, 1,
        ],
    ];

    #[bench]
    fn bench_dct_intrinsics(b: &mut Bencher) {
        b.iter(|| {
            for input in &TEST_BLOCKS {
                let mut block = *input;
                test::black_box(dct4x4_intrinsics(&mut block));
            }
        });
    }

    #[bench]
    fn bench_idct_intrinsics(b: &mut Bencher) {
        let mut dct_blocks = TEST_BLOCKS;
        for block in &mut dct_blocks {
            dct4x4_scalar(block);
        }

        b.iter(|| {
            for input in &dct_blocks {
                let mut block = *input;
                test::black_box(idct4x4_intrinsics(&mut block));
            }
        });
    }

    #[bench]
    fn bench_dct_two_intrinsics(b: &mut Bencher) {
        b.iter(|| {
            let mut block1 = TEST_BLOCKS[0];
            let mut block2 = TEST_BLOCKS[1];
            test::black_box(dct4x4_two_intrinsics(&mut block1, &mut block2));
            let mut block3 = TEST_BLOCKS[2];
            let mut block4 = TEST_BLOCKS[3];
            test::black_box(dct4x4_two_intrinsics(&mut block3, &mut block4));
        });
    }
}

// ============================================================================
// NEON implementations (from transform_aarch64.rs)
// ============================================================================
#[cfg(target_arch = "aarch64")]
mod neon_transform {
    use super::*;

    #[arcane]
    pub(crate) fn dct4x4_neon(_token: NeonToken, block: &mut [i32; 16]) {
        dct4x4_neon_inner(_token, block);
    }

    #[rite]
    fn dct4x4_neon_inner(_token: NeonToken, block: &mut [i32; 16]) {
        // Load i32[16] as 4 × i32x4, then narrow to i16x4
        let (b0, b1, b2, b3) = rows4(block);
        let r0 = simd_mem::vld1q_s32(b0);
        let r1 = simd_mem::vld1q_s32(b1);
        let r2 = simd_mem::vld1q_s32(b2);
        let r3 = simd_mem::vld1q_s32(b3);

        let d0 = vmovn_s32(r0);
        let d1 = vmovn_s32(r1);
        let d2 = vmovn_s32(r2);
        let d3 = vmovn_s32(r3);

        // Transpose 4x4 for first pass
        let (t0t1, t3t2) = transpose_4x4_s16_neon(_token, d0, d1, d2, d3);

        // First DCT pass (includes internal transpose at end)
        let (p0p1, p3p2) = forward_pass_1_neon(_token, t0t1, t3t2);

        // Second DCT pass with final rounding → i32 output
        // Data is already transposed by pass 1's internal transpose
        let out = forward_pass_2_neon(_token, p0p1, p3p2);

        let (b0, b1, b2, b3) = rows4_mut(block);
        simd_mem::vst1q_s32(b0, out[0]);
        simd_mem::vst1q_s32(b1, out[1]);
        simd_mem::vst1q_s32(b2, out[2]);
        simd_mem::vst1q_s32(b3, out[3]);
    }

    /// Transpose 4x4 i16 matrix (matches libwebp Transpose4x4_S16_NEON).
    /// Input: 4 rows as i16x4. Output: (col01, col32) as two i16x8.
    /// col01 = col0 | col1, col32 = col3 | col2

    #[rite]
    fn transpose_4x4_s16_neon(
        _token: NeonToken,
        a: int16x4_t,
        b: int16x4_t,
        c: int16x4_t,
        d: int16x4_t,
    ) -> (int16x8_t, int16x8_t) {
        let ab = vtrn_s16(a, b);
        let cd = vtrn_s16(c, d);
        let tmp02 = vtrn_s32(vreinterpret_s32_s16(ab.0), vreinterpret_s32_s16(cd.0));
        let tmp13 = vtrn_s32(vreinterpret_s32_s16(ab.1), vreinterpret_s32_s16(cd.1));
        let out01 = vreinterpretq_s16_s64(vcombine_s64(
            vreinterpret_s64_s32(tmp02.0),
            vreinterpret_s64_s32(tmp13.0),
        ));
        let out32 = vreinterpretq_s16_s64(vcombine_s64(
            vreinterpret_s64_s32(tmp13.1),
            vreinterpret_s64_s32(tmp02.1),
        ));
        (out01, out32)
    }

    /// Forward DCT pass 1 (from libwebp FTransform_NEON, first pass).

    #[rite]
    fn forward_pass_1_neon(
        _token: NeonToken,
        d0d1: int16x8_t,
        d3d2: int16x8_t,
    ) -> (int16x8_t, int16x8_t) {
        let k_cst937 = vdupq_n_s32(937);
        let k_cst1812 = vdupq_n_s32(1812);

        let a0a1 = vaddq_s16(d0d1, d3d2);
        let a3a2 = vsubq_s16(d0d1, d3d2);

        let a0a1_2 = vshlq_n_s16::<3>(a0a1);
        let tmp0 = vadd_s16(vget_low_s16(a0a1_2), vget_high_s16(a0a1_2));
        let tmp2 = vsub_s16(vget_low_s16(a0a1_2), vget_high_s16(a0a1_2));

        let a3_2217 = vmull_n_s16(vget_low_s16(a3a2), 2217);
        let a2_2217 = vmull_n_s16(vget_high_s16(a3a2), 2217);
        let a2_p_a3 = vmlal_n_s16(a2_2217, vget_low_s16(a3a2), 5352);
        let a3_m_a2 = vmlsl_n_s16(a3_2217, vget_high_s16(a3a2), 5352);

        let tmp1 = vshrn_n_s32::<9>(vaddq_s32(a2_p_a3, k_cst1812));
        let tmp3 = vshrn_n_s32::<9>(vaddq_s32(a3_m_a2, k_cst937));

        transpose_4x4_s16_neon(_token, tmp0, tmp1, tmp2, tmp3)
    }

    /// Forward DCT pass 2 with final rounding. Returns i32x4[4].

    #[rite]
    fn forward_pass_2_neon(_token: NeonToken, d0d1: int16x8_t, d3d2: int16x8_t) -> [int32x4_t; 4] {
        let k_cst12000 = vdupq_n_s32(12000 + (1 << 16));
        let k_cst51000 = vdupq_n_s32(51000);

        let a0a1 = vaddq_s16(d0d1, d3d2);
        let a3a2 = vsubq_s16(d0d1, d3d2);

        let a0_k7 = vadd_s16(vget_low_s16(a0a1), vdup_n_s16(7));
        let out0 = vshr_n_s16::<4>(vadd_s16(a0_k7, vget_high_s16(a0a1)));
        let out2 = vshr_n_s16::<4>(vsub_s16(a0_k7, vget_high_s16(a0a1)));

        let a3_2217 = vmull_n_s16(vget_low_s16(a3a2), 2217);
        let a2_2217 = vmull_n_s16(vget_high_s16(a3a2), 2217);
        let a2_p_a3 = vmlal_n_s16(a2_2217, vget_low_s16(a3a2), 5352);
        let a3_m_a2 = vmlsl_n_s16(a3_2217, vget_high_s16(a3a2), 5352);

        let tmp1 = vaddhn_s32(a2_p_a3, k_cst12000);
        let out3 = vaddhn_s32(a3_m_a2, k_cst51000);

        // out1 += (a3 != 0): ceq returns all-ones (0xFFFF = -1), so add -1 when eq
        let a3_eq_0 = vreinterpret_s16_u16(vceq_s16(vget_low_s16(a3a2), vdup_n_s16(0)));
        let out1 = vadd_s16(tmp1, a3_eq_0);

        // Widen i16x4 → i32x4
        [
            vmovl_s16(out0),
            vmovl_s16(out1),
            vmovl_s16(out2),
            vmovl_s16(out3),
        ]
    }

    // =============================================================================
    // Inverse DCT — ported from libwebp's TransformPass_NEON + ITransformOne_NEON
    // =============================================================================

    // VP8 IDCT constants for vqdmulh doubling multiply-high:
    // kC1 = 20091 (cos(PI/8)*sqrt(2)-1, fixed-point)
    // kC2_half = 17734 (sin(PI/8)*sqrt(2) / 2 for vqdmulh)
    const KC1: i16 = 20091;
    const KC2_HALF: i16 = 17734; // 35468 / 2

    /// Inverse DCT using NEON intrinsics.
    /// Input/output: i32[16+] in row-major order.

    #[arcane]
    pub(crate) fn idct4x4_neon(_token: NeonToken, block: &mut [i32; 16]) {
        idct4x4_neon_inner(_token, block);
    }

    #[rite]
    fn idct4x4_neon_inner(_token: NeonToken, block: &mut [i32; 16]) {
        // Load i32[16] and pack to i16x8 pairs
        let (b0, b1, b2, b3) = rows4(block);
        let r0 = simd_mem::vld1q_s32(b0);
        let r1 = simd_mem::vld1q_s32(b1);
        let r2 = simd_mem::vld1q_s32(b2);
        let r3 = simd_mem::vld1q_s32(b3);

        let in01 = vcombine_s16(vmovn_s32(r0), vmovn_s32(r1));
        let in23 = vcombine_s16(vmovn_s32(r2), vmovn_s32(r3));

        // Two passes of TransformPass (vertical then horizontal)
        let (t01, t23) = itransform_pass_neon(_token, in01, in23);
        let (res01, res23) = itransform_pass_neon(_token, t01, t23);

        // Final: (val + 4) >> 3 rounding
        let four = vdupq_n_s16(4);
        let res01_r = vshrq_n_s16::<3>(vaddq_s16(res01, four));
        let res23_r = vshrq_n_s16::<3>(vaddq_s16(res23, four));

        // Widen i16 → i32 and store
        let (b0, b1, b2, b3) = rows4_mut(block);
        simd_mem::vst1q_s32(b0, vmovl_s16(vget_low_s16(res01_r)));
        simd_mem::vst1q_s32(b1, vmovl_s16(vget_high_s16(res01_r)));
        simd_mem::vst1q_s32(b2, vmovl_s16(vget_low_s16(res23_r)));
        simd_mem::vst1q_s32(b3, vmovl_s16(vget_high_s16(res23_r)));
    }

    /// IDCT TransformPass — matches libwebp's TransformPass_NEON.
    /// rows packed as (row0|row1, row2|row3) in i16x8 pairs.

    #[rite]
    fn itransform_pass_neon(
        _token: NeonToken,
        in01: int16x8_t,
        in23: int16x8_t,
    ) -> (int16x8_t, int16x8_t) {
        // B1 = in4 | in12 (high halves)
        let b1 = vcombine_s16(vget_high_s16(in01), vget_high_s16(in23));

        // C0 = kC1 * B1: vqdmulh gives (B1*kC1*2)>>16, then vsraq adds B1>>1
        let c0 = vsraq_n_s16::<1>(b1, vqdmulhq_n_s16(b1, KC1));
        // C1 = kC2 * B1 / 2^16 via vqdmulh with kC2/2
        let c1 = vqdmulhq_n_s16(b1, KC2_HALF);

        // a = in0 + in8, b = in0 - in8
        let a = vqadd_s16(vget_low_s16(in01), vget_low_s16(in23));
        let b = vqsub_s16(vget_low_s16(in01), vget_low_s16(in23));

        // c = kC2*in4 - kC1*in12, d = kC1*in4 + kC2*in12
        let c = vqsub_s16(vget_low_s16(c1), vget_high_s16(c0));
        let d = vqadd_s16(vget_low_s16(c0), vget_high_s16(c1));

        let d0 = vcombine_s16(a, b); // a | b
        let d1 = vcombine_s16(d, c); // d | c

        let e0 = vqaddq_s16(d0, d1); // a+d | b+c
        let e_tmp = vqsubq_s16(d0, d1); // a-d | b-c
        let e1 = vcombine_s16(vget_high_s16(e_tmp), vget_low_s16(e_tmp)); // b-c | a-d

        // Transpose 8x2
        let tmp = vzipq_s16(e0, e1);
        let out = vzipq_s16(tmp.0, tmp.1);
        (out.0, out.1)
    }

    // =============================================================================
    // Fused IDCT + add_residue (decoder hot path, in-place)
    // =============================================================================

    /// Fused IDCT + add residue + clear coefficients.
    /// Performs IDCT on raw DCT coefficients, adds to prediction block in-place,
    /// clamps to [0,255], and zeros the coefficient buffer.

    #[arcane]
    pub(crate) fn idct_add_residue_inplace_neon(
        _token: NeonToken,
        coeffs: &mut [i32; 16],
        block: &mut [u8],
        y0: usize,
        x0: usize,
        stride: usize,
        dc_only: bool,
    ) {
        idct_add_residue_inplace_neon_inner(_token, coeffs, block, y0, x0, stride, dc_only);
    }

    /// `#[rite]` version for inlining into `#[arcane]` prediction+IDCT pipelines.
    #[rite]
    pub(crate) fn idct_add_residue_inplace_neon_inner(
        _token: NeonToken,
        coeffs: &mut [i32; 16],
        block: &mut [u8],
        y0: usize,
        x0: usize,
        stride: usize,
        dc_only: bool,
    ) {
        if dc_only {
            idct_add_residue_dc_neon(_token, coeffs, block, y0, x0, stride);
        } else {
            idct_add_residue_full_neon(_token, coeffs, block, y0, x0, stride);
        }
        coeffs.fill(0);
    }

    /// DC-only fast path: add constant to all 16 pixels.

    #[rite]
    fn idct_add_residue_dc_neon(
        _token: NeonToken,
        coeffs: &[i32; 16],
        block: &mut [u8],
        y0: usize,
        x0: usize,
        stride: usize,
    ) {
        let dc = coeffs[0];
        let dc_adj = ((dc + 4) >> 3) as i16;
        let dc_vec = vdupq_n_s16(dc_adj);

        for row in 0..4 {
            let pos = (y0 + row) * stride + x0;
            let pred_bytes: [u8; 4] = block[pos..pos + 4].try_into().unwrap();
            // Load 4 prediction bytes, zero-extend to i16
            let pred_u32 = u32::from_ne_bytes(pred_bytes);
            let pred_v = vreinterpret_u8_u32(vmov_n_u32(pred_u32));
            let pred_i16 = vreinterpretq_s16_u16(vmovl_u8(pred_v));
            // Add DC
            let sum = vaddq_s16(pred_i16, dc_vec);
            // Saturate to u8
            let packed = vqmovun_s16(sum);
            // Store 4 bytes
            let result_u32 = vget_lane_u32::<0>(vreinterpret_u32_u8(packed));
            block[pos..pos + 4].copy_from_slice(&result_u32.to_ne_bytes());
        }
    }

    /// Full IDCT + add residue.

    #[rite]
    fn idct_add_residue_full_neon(
        _token: NeonToken,
        coeffs: &[i32; 16],
        block: &mut [u8],
        y0: usize,
        x0: usize,
        stride: usize,
    ) {
        // Load and pack coefficients to i16
        let (c0, c1, c2, c3) = rows4(coeffs);
        let r0 = simd_mem::vld1q_s32(c0);
        let r1 = simd_mem::vld1q_s32(c1);
        let r2 = simd_mem::vld1q_s32(c2);
        let r3 = simd_mem::vld1q_s32(c3);

        let in01 = vcombine_s16(vmovn_s32(r0), vmovn_s32(r1));
        let in23 = vcombine_s16(vmovn_s32(r2), vmovn_s32(r3));

        // Two IDCT passes
        let (t01, t23) = itransform_pass_neon(_token, in01, in23);
        let (res01, res23) = itransform_pass_neon(_token, t01, t23);

        // Process each row: load pred, add residual with rounding shift, saturate, store
        // libwebp uses vrsraq_n_s16 for: pred + (residual + rounding) >> 3
        let res_rows: [(int16x8_t, bool); 4] = [
            (res01, false), // row 0 = low half of res01
            (res01, true),  // row 1 = high half of res01
            (res23, false), // row 2 = low half of res23
            (res23, true),  // row 3 = high half of res23
        ];

        for (row_idx, &(res, use_high)) in res_rows.iter().enumerate() {
            let residual = if use_high {
                vcombine_s16(vget_high_s16(res), vget_high_s16(res))
            } else {
                vcombine_s16(vget_low_s16(res), vget_low_s16(res))
            };

            let pos = (y0 + row_idx) * stride + x0;
            let pred_bytes: [u8; 4] = block[pos..pos + 4].try_into().unwrap();
            let pred_u32 = u32::from_ne_bytes(pred_bytes);
            let pred_v = vreinterpret_u8_u32(vmov_n_u32(pred_u32));
            let pred_i16 = vreinterpretq_s16_u16(vmovl_u8(pred_v));

            // vrsraq: pred + (residual + 4) >> 3
            let out = vrsraq_n_s16::<3>(pred_i16, residual);
            let packed = vqmovun_s16(out);
            let result_u32 = vget_lane_u32::<0>(vreinterpret_u32_u8(packed));
            block[pos..pos + 4].copy_from_slice(&result_u32.to_ne_bytes());
        }
    }

    // =============================================================================
    // Fused residual + DCT from u8 arrays (encoder hot path)
    // =============================================================================

    /// Fused residual computation + DCT for a single 4x4 block.
    /// Takes flat u8 source and reference arrays (stride=4), outputs i32 coefficients.

    #[arcane]
    pub(crate) fn ftransform_from_u8_4x4_neon(
        _token: NeonToken,
        src: &[u8; 16],
        ref_: &[u8; 16],
    ) -> [i32; 16] {
        ftransform_from_u8_4x4_neon_inner(_token, src, ref_)
    }

    #[rite]
    fn ftransform_from_u8_4x4_neon_inner(
        _token: NeonToken,
        src: &[u8; 16],
        ref_: &[u8; 16],
    ) -> [i32; 16] {
        // Load src and ref as u8x16
        let s = simd_mem::vld1q_u8(src);
        let r = simd_mem::vld1q_u8(ref_);

        // Compute difference as i16: rows 0-1 and rows 2-3
        let diff_01 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s), vget_low_u8(r)));
        let diff_23 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s), vget_high_u8(r)));

        let d0 = vget_low_s16(diff_01);
        let d1 = vget_high_s16(diff_01);
        let d2 = vget_low_s16(diff_23);
        let d3 = vget_high_s16(diff_23);

        // Transpose for first pass
        let (t0t1, t3t2) = transpose_4x4_s16_neon(_token, d0, d1, d2, d3);

        // First DCT pass (includes internal transpose at end)
        let (p0p1, p3p2) = forward_pass_1_neon(_token, t0t1, t3t2);

        // Second DCT pass — data already transposed by pass 1
        let out = forward_pass_2_neon(_token, p0p1, p3p2);

        let mut result = [0i32; 16];
        let (r0, r1, r2, r3) = rows4_mut(&mut result);
        simd_mem::vst1q_s32(r0, out[0]);
        simd_mem::vst1q_s32(r1, out[1]);
        simd_mem::vst1q_s32(r2, out[2]);
        simd_mem::vst1q_s32(r3, out[3]);
        result
    }

    // =============================================================================
    // ftransform2 — two adjacent 4x4 blocks
    // =============================================================================

    /// Process two 4x4 blocks with forward DCT.

    #[arcane]
    pub(crate) fn ftransform2_neon(
        _token: NeonToken,
        src: &[u8],
        ref_: &[u8],
        src_stride: usize,
        ref_stride: usize,
        out: &mut [[i32; 16]; 2],
    ) {
        for blk in 0..2 {
            let sx = blk * 4;
            let mut s_flat = [0u8; 16];
            let mut r_flat = [0u8; 16];
            for row in 0..4 {
                s_flat[row * 4..row * 4 + 4].copy_from_slice(&src[row * src_stride + sx..][..4]);
                r_flat[row * 4..row * 4 + 4].copy_from_slice(&ref_[row * ref_stride + sx..][..4]);
            }
            out[blk] = ftransform_from_u8_4x4_neon(_token, &s_flat, &r_flat);
        }
    }

    // =============================================================================
    // add_residue — add IDCT residuals to prediction block
    // =============================================================================

    /// Add residuals (i32[16]) to prediction block (u8) with saturation.
    /// Processes 4 rows of 4 pixels each.

    #[arcane]
    pub(crate) fn add_residue_neon(
        _token: NeonToken,
        pblock: &mut [u8],
        rblock: &[i32; 16],
        y0: usize,
        x0: usize,
        stride: usize,
    ) {
        add_residue_neon_inner(_token, pblock, rblock, y0, x0, stride);
    }

    #[rite]
    fn add_residue_neon_inner(
        _token: NeonToken,
        pblock: &mut [u8],
        rblock: &[i32; 16],
        y0: usize,
        x0: usize,
        stride: usize,
    ) {
        // Load residuals as i32x4 and narrow to i16x4
        let (b0, b1, b2, b3) = rows4(rblock);
        let r0_i32 = simd_mem::vld1q_s32(b0);
        let r1_i32 = simd_mem::vld1q_s32(b1);
        let r2_i32 = simd_mem::vld1q_s32(b2);
        let r3_i32 = simd_mem::vld1q_s32(b3);

        let r0_i16 = vmovn_s32(r0_i32); // int16x4
        let r1_i16 = vmovn_s32(r1_i32);
        let r2_i16 = vmovn_s32(r2_i32);
        let r3_i16 = vmovn_s32(r3_i32);

        // Process 4 rows
        for (row, r_i16) in [(0usize, r0_i16), (1, r1_i16), (2, r2_i16), (3, r3_i16)] {
            let offset = (y0 + row) * stride + x0;

            // Load 4 prediction pixels, zero-extend to i16
            let mut pred_bytes = [0u8; 8]; // load 4, pad to 8
            pred_bytes[..4].copy_from_slice(&pblock[offset..offset + 4]);
            let pred_u8 = simd_mem::vld1_u8(&pred_bytes);
            let pred_i16 = vreinterpretq_s16_u16(vmovl_u8(pred_u8));

            // Add residual (only low 4 lanes matter)
            let sum = vaddq_s16(pred_i16, vcombine_s16(r_i16, vdup_n_s16(0)));

            // Saturate to u8
            let result = vqmovun_s16(sum);

            // Store 4 pixels back
            let mut out_bytes = [0u8; 8];
            simd_mem::vst1_u8(&mut out_bytes, result);
            pblock[offset..offset + 4].copy_from_slice(&out_bytes[..4]);
        }
    }
} // mod neon_transform

#[cfg(target_arch = "aarch64")]
pub(crate) use neon_transform::*;

// ============================================================================
// WASM SIMD128 implementations (from transform_wasm.rs)
// ============================================================================
#[cfg(target_arch = "wasm32")]
mod wasm_transform {
    use super::*;

    pub(crate) fn dct4x4_wasm(_token: Wasm128Token, block: &mut [i32; 16]) {
        dct4x4_wasm_impl(_token, block);
    }

    /// Inverse DCT using WASM SIMD128 (entry shim)
    #[cfg(target_arch = "wasm32")]
    #[arcane]
    pub(crate) fn idct4x4_wasm(_token: Wasm128Token, block: &mut [i32; 16]) {
        idct4x4_wasm_impl(_token, block);
    }

    // =============================================================================
    // Helpers
    // =============================================================================

    /// Load 4 i32 values from a block row
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    fn load_row(block: &[i32], row: usize) -> v128 {
        let off = row * 4;
        i32x4(block[off], block[off + 1], block[off + 2], block[off + 3])
    }

    /// Store v128 as 4 i32 values into a block row
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    fn store_row(block: &mut [i32], row: usize, v: v128) {
        let off = row * 4;
        block[off] = i32x4_extract_lane::<0>(v);
        block[off + 1] = i32x4_extract_lane::<1>(v);
        block[off + 2] = i32x4_extract_lane::<2>(v);
        block[off + 3] = i32x4_extract_lane::<3>(v);
    }

    /// Compute (v * constant) >> 16 using 64-bit intermediates.
    /// Matches the scalar `(val * CONST) >> 16` pattern used in VP8 transforms.
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    fn mulhi32x4(v: v128, c: i32) -> v128 {
        let c_vec = i32x4_splat(c);
        let lo_prod = i64x2_extmul_low_i32x4(v, c_vec);
        let hi_prod = i64x2_extmul_high_i32x4(v, c_vec);
        let lo_shifted = i64x2_shr(lo_prod, 16);
        let hi_shifted = i64x2_shr(hi_prod, 16);
        // Narrow back to i32x4 by extracting low 32 bits of each i64
        i32x4(
            i64x2_extract_lane::<0>(lo_shifted) as i32,
            i64x2_extract_lane::<1>(lo_shifted) as i32,
            i64x2_extract_lane::<0>(hi_shifted) as i32,
            i64x2_extract_lane::<1>(hi_shifted) as i32,
        )
    }

    /// Transpose a 4x4 matrix of i32 values stored as 4 row vectors.
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    fn transpose4x4(r0: v128, r1: v128, r2: v128, r3: v128) -> (v128, v128, v128, v128) {
        // Step 1: interleave pairs of 32-bit elements
        let t0 = i32x4_shuffle::<0, 4, 1, 5>(r0, r1); // r0[0], r1[0], r0[1], r1[1]
        let t1 = i32x4_shuffle::<2, 6, 3, 7>(r0, r1); // r0[2], r1[2], r0[3], r1[3]
        let t2 = i32x4_shuffle::<0, 4, 1, 5>(r2, r3); // r2[0], r3[0], r2[1], r3[1]
        let t3 = i32x4_shuffle::<2, 6, 3, 7>(r2, r3); // r2[2], r3[2], r2[3], r3[3]

        // Step 2: interleave 64-bit pairs
        let o0 = i64x2_shuffle::<0, 2>(t0, t2); // col 0: r0[0], r1[0], r2[0], r3[0]
        let o1 = i64x2_shuffle::<1, 3>(t0, t2); // col 1: r0[1], r1[1], r2[1], r3[1]
        let o2 = i64x2_shuffle::<0, 2>(t1, t3); // col 2: r0[2], r1[2], r2[2], r3[2]
        let o3 = i64x2_shuffle::<1, 3>(t1, t3); // col 3: r0[3], r1[3], r2[3], r3[3]

        (o0, o1, o2, o3)
    }

    // =============================================================================
    // Inverse DCT
    // =============================================================================

    // VP8 IDCT constants (WASM version, i32)
    const WASM_CONST1: i32 = 20091; // sqrt(2)*cos(pi/8)*65536 - 65536
    const WASM_CONST2: i32 = 35468; // sqrt(2)*sin(pi/8)*65536

    /// One pass of the IDCT butterfly. Processes 4 elements in parallel.
    /// Matches the scalar: a=in0+in2, b=in0-in2, c=MUL(in1,K2)-MUL(in3,K1), d=MUL(in1,K1)+MUL(in3,K2)
    /// where MUL(x,K) = x + (x*k)>>16 for K1, or just (x*k)>>16 for K2.
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    fn idct_butterfly(in0: v128, in1: v128, in2: v128, in3: v128) -> (v128, v128, v128, v128) {
        let a = i32x4_add(in0, in2);
        let b = i32x4_sub(in0, in2);

        // t1 = (in1 * WASM_CONST2) >> 16
        let t1_c = mulhi32x4(in1, WASM_CONST2);
        // t2 = in3 + (in3 * WASM_CONST1) >> 16  = MUL(in3, K1)
        let t2_c = i32x4_add(in3, mulhi32x4(in3, WASM_CONST1));
        let c = i32x4_sub(t1_c, t2_c);

        // t1 = in1 + (in1 * WASM_CONST1) >> 16  = MUL(in1, K1)
        let t1_d = i32x4_add(in1, mulhi32x4(in1, WASM_CONST1));
        // t2 = (in3 * WASM_CONST2) >> 16
        let t2_d = mulhi32x4(in3, WASM_CONST2);
        let d = i32x4_add(t1_d, t2_d);

        let out0 = i32x4_add(a, d);
        let out1 = i32x4_add(b, c);
        let out2 = i32x4_sub(b, c);
        let out3 = i32x4_sub(a, d);

        (out0, out1, out2, out3)
    }

    /// WASM SIMD128 inverse DCT. Uses i32x4 arithmetic matching the scalar implementation.
    ///
    /// Scalar pass order: column pass (for i in 0..4, reads block[i], block[4+i], ...),
    /// then row pass (for i in 0..4, reads block[4*i], block[4*i+1], ...).
    ///
    /// With row vectors r0..r3, `idct_butterfly(r0,r1,r2,r3)` processes lane j as
    /// butterfly(row0[j], row1[j], row2[j], row3[j]) = column pass (no transpose needed).
    /// For the row pass, we transpose so each vector holds a row's elements as lanes.
    #[cfg(target_arch = "wasm32")]
    #[rite]
    pub(crate) fn idct4x4_wasm_impl(_token: Wasm128Token, block: &mut [i32; 16]) {
        // Load 4 rows
        let r0 = load_row(block, 0);
        let r1 = load_row(block, 1);
        let r2 = load_row(block, 2);
        let r3 = load_row(block, 3);

        // Pass 1: column pass — butterfly on row vectors processes columns in parallel
        let (v0, v1, v2, v3) = idct_butterfly(r0, r1, r2, r3);

        // Pass 2: row pass — transpose so each vector holds one row's elements,
        // butterfly processes rows, then transpose back to row-major
        let (c0, c1, c2, c3) = transpose4x4(v0, v1, v2, v3);
        let (t0, t1, t2, t3) = idct_butterfly(c0, c1, c2, c3);
        let (f0, f1, f2, f3) = transpose4x4(t0, t1, t2, t3);

        // Final rounding: (val + 4) >> 3
        let four = i32x4_splat(4);
        let o0 = i32x4_shr(i32x4_add(f0, four), 3);
        let o1 = i32x4_shr(i32x4_add(f1, four), 3);
        let o2 = i32x4_shr(i32x4_add(f2, four), 3);
        let o3 = i32x4_shr(i32x4_add(f3, four), 3);

        // Store results
        store_row(block, 0, o0);
        store_row(block, 1, o1);
        store_row(block, 2, o2);
        store_row(block, 3, o3);
    }

    // =============================================================================
    // Forward DCT
    // =============================================================================

    /// First pass of the forward DCT butterfly (row pass). Processes 4 elements in parallel.
    ///
    /// Scalar row pass (for each row i):
    ///   a = (block[i*4+0] + block[i*4+3]) * 8
    ///   b = (block[i*4+1] + block[i*4+2]) * 8
    ///   c = (block[i*4+1] - block[i*4+2]) * 8
    ///   d = (block[i*4+0] - block[i*4+3]) * 8
    ///   out[i*4]   = a + b
    ///   out[i*4+1] = (c*2217 + d*5352 + 14500) >> 12
    ///   out[i*4+2] = a - b
    ///   out[i*4+3] = (d*2217 - c*5352 + 7500) >> 12
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    fn dct_butterfly(in0: v128, in1: v128, in2: v128, in3: v128) -> (v128, v128, v128, v128) {
        let eight = i32x4_splat(8);

        let a = i32x4_mul(i32x4_add(in0, in3), eight);
        let b = i32x4_mul(i32x4_add(in1, in2), eight);
        let c = i32x4_mul(i32x4_sub(in1, in2), eight);
        let d = i32x4_mul(i32x4_sub(in0, in3), eight);

        let k2217 = i32x4_splat(2217);
        let k5352 = i32x4_splat(5352);

        let out0 = i32x4_add(a, b);

        // out1 = (c * 2217 + d * 5352 + 14500) >> 12
        let k14500 = i32x4_splat(14500);
        let out1 = i32x4_shr(
            i32x4_add(i32x4_add(i32x4_mul(c, k2217), i32x4_mul(d, k5352)), k14500),
            12,
        );

        // out2 = a - b
        let out2 = i32x4_sub(a, b);

        // out3 = (d * 2217 - c * 5352 + 7500) >> 12
        let k7500 = i32x4_splat(7500);
        let out3 = i32x4_shr(
            i32x4_add(i32x4_sub(i32x4_mul(d, k2217), i32x4_mul(c, k5352)), k7500),
            12,
        );

        (out0, out1, out2, out3)
    }

    /// Second pass of forward DCT (column pass). Same butterfly pairing as pass 1,
    /// but different constants and rounding.
    ///
    /// Scalar column pass (for column i):
    ///   a = block[i] + block[i+12]         (row0 + row3)
    ///   b = block[i+4] + block[i+8]        (row1 + row2)
    ///   c = block[i+4] - block[i+8]        (row1 - row2)
    ///   d = block[i] - block[i+12]         (row0 - row3)
    ///   out_row0 = (a + b + 7) >> 4
    ///   out_row1 = (c*2217 + d*5352 + 12000) >> 16 + (d != 0)
    ///   out_row2 = (a - b + 7) >> 4
    ///   out_row3 = (d*2217 - c*5352 + 51000) >> 16
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    fn dct_butterfly_pass2(in0: v128, in1: v128, in2: v128, in3: v128) -> (v128, v128, v128, v128) {
        let a = i32x4_add(in0, in3);
        let b = i32x4_add(in1, in2);
        let c = i32x4_sub(in1, in2);
        let d = i32x4_sub(in0, in3);

        let k2217 = i32x4_splat(2217);
        let k5352 = i32x4_splat(5352);
        let k7 = i32x4_splat(7);
        let k12000 = i32x4_splat(12000);
        let k51000 = i32x4_splat(51000);

        // out0 = (a + b + 7) >> 4
        let out0 = i32x4_shr(i32x4_add(i32x4_add(a, b), k7), 4);

        // out1 = (c * 2217 + d * 5352 + 12000) >> 16 + (d != 0)
        let out1_raw = i32x4_shr(
            i32x4_add(i32x4_add(i32x4_mul(c, k2217), i32x4_mul(d, k5352)), k12000),
            16,
        );
        // d_ne_0: 0xFFFFFFFF when d != 0, 0 when d == 0
        // We need +1 when d != 0, so negate the mask (-(-1) = 1)
        let d_ne_0 = v128_not(i32x4_eq(d, i32x4_splat(0)));
        let out1 = i32x4_sub(out1_raw, d_ne_0);

        // out2 = (a - b + 7) >> 4
        let out2 = i32x4_shr(i32x4_add(i32x4_sub(a, b), k7), 4);

        // out3 = (d * 2217 - c * 5352 + 51000) >> 16
        let out3 = i32x4_shr(
            i32x4_add(i32x4_sub(i32x4_mul(d, k2217), i32x4_mul(c, k5352)), k51000),
            16,
        );

        (out0, out1, out2, out3)
    }

    /// Fused residual computation + forward DCT for a single 4x4 block.
    /// Takes flat u8 source and reference arrays (stride=4), outputs i32 coefficients.
    ///
    /// Fuses: residual = src - ref, then DCT on residual.
    /// Avoids intermediate i32 storage by computing diff in i16 then widening to i32.
    #[cfg(target_arch = "wasm32")]
    #[arcane]
    pub(crate) fn ftransform_from_u8_4x4_wasm(
        _token: Wasm128Token,
        src: &[u8; 16],
        ref_: &[u8; 16],
    ) -> [i32; 16] {
        ftransform_from_u8_4x4_wasm_impl(_token, src, ref_)
    }

    /// Inner #[rite] implementation of fused residual+DCT for wasm.
    #[cfg(target_arch = "wasm32")]
    #[rite]
    pub(crate) fn ftransform_from_u8_4x4_wasm_impl(
        _token: Wasm128Token,
        src: &[u8; 16],
        ref_: &[u8; 16],
    ) -> [i32; 16] {
        // Load src and ref as u8x16
        let src_vec = u8x16(
            src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], src[9],
            src[10], src[11], src[12], src[13], src[14], src[15],
        );
        let ref_vec = u8x16(
            ref_[0], ref_[1], ref_[2], ref_[3], ref_[4], ref_[5], ref_[6], ref_[7], ref_[8],
            ref_[9], ref_[10], ref_[11], ref_[12], ref_[13], ref_[14], ref_[15],
        );

        // Zero-extend to i16 and compute diff (src - ref)
        let src_lo = u16x8_extend_low_u8x16(src_vec);
        let src_hi = u16x8_extend_high_u8x16(src_vec);
        let ref_lo = u16x8_extend_low_u8x16(ref_vec);
        let ref_hi = u16x8_extend_high_u8x16(ref_vec);
        let diff_lo = i16x8_sub(src_lo, ref_lo);
        let diff_hi = i16x8_sub(src_hi, ref_hi);

        // Widen i16 → i32 for DCT (4 rows of 4 values)
        let r0 = i32x4_extend_low_i16x8(diff_lo);
        let r1 = i32x4_extend_high_i16x8(diff_lo);
        let r2 = i32x4_extend_low_i16x8(diff_hi);
        let r3 = i32x4_extend_high_i16x8(diff_hi);

        // Forward DCT pass 1: row pass (transpose, butterfly, transpose back)
        let (c0, c1, c2, c3) = transpose4x4(r0, r1, r2, r3);
        let (v0, v1, v2, v3) = dct_butterfly(c0, c1, c2, c3);
        let (t0, t1, t2, t3) = transpose4x4(v0, v1, v2, v3);

        // Forward DCT pass 2: column pass
        let (o0, o1, o2, o3) = dct_butterfly_pass2(t0, t1, t2, t3);

        // Store results
        let mut result = [0i32; 16];
        store_row(&mut result, 0, o0);
        store_row(&mut result, 1, o1);
        store_row(&mut result, 2, o2);
        store_row(&mut result, 3, o3);
        result
    }

    /// WASM SIMD128 forward DCT. Uses i32x4 arithmetic matching the scalar implementation.
    ///
    /// Scalar pass order: row pass first (for i in 0..4, reads block[i*4..i*4+3]),
    /// then column pass (for i in 0..4, reads block[i], block[i+4], ...).
    ///
    /// With row vectors, `dct_butterfly(r0,r1,r2,r3)` processes lane j as
    /// butterfly(row0[j], row1[j], row2[j], row3[j]) = column pass.
    /// For the row pass, we transpose so each vector holds one row's elements.
    #[cfg(target_arch = "wasm32")]
    #[rite]
    pub(crate) fn dct4x4_wasm_impl(_token: Wasm128Token, block: &mut [i32; 16]) {
        // Load 4 rows
        let r0 = load_row(block, 0);
        let r1 = load_row(block, 1);
        let r2 = load_row(block, 2);
        let r3 = load_row(block, 3);

        // Pass 1: row pass — transpose so each vector holds one row's elements,
        // butterfly processes rows, then transpose back to row-major
        let (c0, c1, c2, c3) = transpose4x4(r0, r1, r2, r3);
        let (v0, v1, v2, v3) = dct_butterfly(c0, c1, c2, c3);
        let (t0, t1, t2, t3) = transpose4x4(v0, v1, v2, v3);

        // Pass 2: column pass — butterfly on row vectors processes columns in parallel
        let (o0, o1, o2, o3) = dct_butterfly_pass2(t0, t1, t2, t3);

        // Store results
        store_row(block, 0, o0);
        store_row(block, 1, o1);
        store_row(block, 2, o2);
        store_row(block, 3, o3);
    }
} // mod wasm_transform

#[cfg(target_arch = "wasm32")]
pub(crate) use wasm_transform::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_inverse() {
        const BLOCK: [i32; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];

        let mut dct_block = BLOCK;

        dct4x4(&mut dct_block);

        let mut inverse_dct_block = dct_block;
        idct4x4(&mut inverse_dct_block);
        assert_eq!(BLOCK, inverse_dct_block);
    }
}
