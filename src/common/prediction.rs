//! Methods related to vp8 prediction used in both the decoder and the encoder
//!
//! Functions for doing prediction and for setting up buffers for prediction

// Allow dead code when std is disabled - some functions are encoder-only
#![cfg_attr(not(feature = "std"), allow(dead_code))]

use archmage::prelude::*;

/// Luma prediction block stride - 32 bytes for cache alignment (matches libwebp BPS)
/// Layout: 1 border pixel + 16 luma pixels + 4 top-right + padding to 32
pub(crate) const LUMA_STRIDE: usize = 32;
/// Luma prediction block size: 17 rows (1 border + 16) × 32 byte stride
pub(crate) const LUMA_BLOCK_SIZE: usize = LUMA_STRIDE * (1 + 16);

/// Creates a luma block with border used for luma prediction
pub(crate) fn create_border_luma(
    mbx: usize,
    mby: usize,
    mbw: usize,
    top: &[u8],
    left: &[u8],
) -> [u8; LUMA_BLOCK_SIZE] {
    let stride = LUMA_STRIDE;
    let mut ws = [0u8; LUMA_BLOCK_SIZE];

    // A
    {
        let above = &mut ws[1..stride];
        if mby == 0 {
            for above in above.iter_mut() {
                *above = 127;
            }
        } else {
            for (above, &top) in above[..16].iter_mut().zip(&top[mbx * 16..]) {
                *above = top;
            }

            if mbx == mbw - 1 {
                for above in &mut above[16..] {
                    *above = top[mbx * 16 + 15];
                }
            } else {
                for (above, &top) in above[16..].iter_mut().zip(&top[mbx * 16 + 16..]) {
                    *above = top;
                }
            }
        }
    }

    // Copy 4 top-right border pixels to rows 4, 8, 12 (for I4 prediction modes)
    for i in 17usize..21 {
        ws[4 * stride + i] = ws[i];
        ws[8 * stride + i] = ws[i];
        ws[12 * stride + i] = ws[i];
    }

    // L
    if mbx == 0 {
        for i in 0usize..16 {
            ws[(i + 1) * stride] = 129;
        }
    } else {
        for (i, &left) in (0usize..16).zip(&left[1..]) {
            ws[(i + 1) * stride] = left;
        }
    }

    // P
    ws[0] = if mby == 0 {
        127
    } else if mbx == 0 {
        129
    } else {
        left[0]
    };

    ws
}

/// Updates an existing luma border block in-place (avoids re-zeroing 544 bytes per MB).
/// Only the border pixels (top row, left column, corner, top-right) are updated.
/// Interior pixels are overwritten by prediction + add_residue, so their prior values don't matter.
#[inline(always)]
pub(crate) fn update_border_luma(
    ws: &mut [u8; LUMA_BLOCK_SIZE],
    mbx: usize,
    mby: usize,
    mbw: usize,
    top: &[u8],
    left: &[u8],
) {
    let stride = LUMA_STRIDE;

    // Top border row (A) — bytes 1..stride in row 0
    if mby == 0 {
        ws[1..stride].fill(127);
    } else {
        // Pre-slice top border to eliminate repeated mbx*16 bounds checks.
        let top_mb = &top[mbx * 16..];
        ws[1..][..16].copy_from_slice(&top_mb[..16]);

        if mbx == mbw - 1 {
            let last = top_mb[15];
            ws[17..stride].fill(last);
        } else {
            let tr_len = stride - 17; // remaining bytes after 16 luma + 1 border
            let copy_len = tr_len.min(top_mb.len() - 16);
            ws[17..][..copy_len].copy_from_slice(&top_mb[16..][..copy_len]);
        }
    }

    // Copy top-right border pixels to rows 4, 8, 12 (for I4 prediction modes)
    // Use direct array indexing — ws is [u8; LUMA_BLOCK_SIZE] so all offsets
    // are compile-time provable in-bounds (max: 12*32+20 = 404 < 544).
    for i in 17usize..21 {
        ws[4 * stride + i] = ws[i];
        ws[8 * stride + i] = ws[i];
        ws[12 * stride + i] = ws[i];
    }

    // Left border column (L) — column 0 of rows 1..=16
    if mbx == 0 {
        for i in 0usize..16 {
            ws[(i + 1) * stride] = 129;
        }
    } else {
        for (i, &l) in (0usize..16).zip(&left[1..]) {
            ws[(i + 1) * stride] = l;
        }
    }

    // Corner pixel (P) — position [0]
    ws[0] = if mby == 0 {
        127
    } else if mbx == 0 {
        129
    } else {
        left[0]
    };
}

/// Updates an existing chroma border block in-place (avoids re-zeroing 288 bytes per MB).
/// Only 8 bytes of the top border are read by any 8x8 prediction mode.
#[inline(always)]
pub(crate) fn update_border_chroma(
    cb: &mut [u8; CHROMA_BLOCK_SIZE],
    mbx: usize,
    mby: usize,
    top: &[u8],
    left: &[u8],
) {
    let stride = CHROMA_STRIDE;

    // Top border row — only first 8 bytes matter for 8x8 prediction
    if mby == 0 {
        // First row: fill entire top border for safety (zero-init assumption)
        cb[1..stride].fill(127);
    } else {
        // Pre-slice to avoid double bounds check on mbx*8
        cb[1..][..8].copy_from_slice(&top[mbx * 8..mbx * 8 + 8]);
    }

    // Left border column
    if mbx == 0 {
        for y in 0usize..8 {
            cb[(y + 1) * stride] = 129;
        }
    } else {
        for (y, &l) in (0usize..8).zip(&left[1..]) {
            cb[(y + 1) * stride] = l;
        }
    }

    // Corner pixel
    cb[0] = if mby == 0 {
        127
    } else if mbx == 0 {
        129
    } else {
        left[0]
    };
}

/// Chroma prediction block stride - 32 bytes for cache alignment (matches libwebp BPS)
pub(crate) const CHROMA_STRIDE: usize = 32;
/// Chroma prediction block size: 9 rows (1 border + 8) × 32 byte stride
pub(crate) const CHROMA_BLOCK_SIZE: usize = CHROMA_STRIDE * (8 + 1);

/// Total coefficient storage per macroblock: 24 blocks × 16 coefficients each.
/// 16 luma Y blocks + 4 chroma U blocks + 4 chroma V blocks = 24.
pub(crate) const MB_COEFF_SIZE: usize = 24 * 16;

/// Extract a mutable reference to a single 4×4 coefficient block from the
/// macroblock coefficient array. With a `[i32; 384]` source, the compiler
/// can prove `idx < 24` implies in-bounds, eliminating the bounds check.
#[inline(always)]
pub(crate) fn coeff_block(blocks: &mut [i32; MB_COEFF_SIZE], idx: usize) -> &mut [i32; 16] {
    let start = idx * 16;
    (&mut blocks[start..start + 16]).try_into().unwrap()
}

/// Creates a chroma block with border used for chroma prediction
pub(crate) fn create_border_chroma(
    mbx: usize,
    mby: usize,
    top: &[u8],
    left: &[u8],
) -> [u8; CHROMA_BLOCK_SIZE] {
    let stride: usize = CHROMA_STRIDE;
    let mut chroma_block = [0u8; CHROMA_BLOCK_SIZE];

    // above
    {
        let above = &mut chroma_block[1..stride];
        if mby == 0 {
            for above in above.iter_mut() {
                *above = 127;
            }
        } else {
            for (above, &top) in above.iter_mut().zip(&top[mbx * 8..]) {
                *above = top;
            }
        }
    }

    // left
    if mbx == 0 {
        for y in 0usize..8 {
            chroma_block[(y + 1) * stride] = 129;
        }
    } else {
        for (y, &left) in (0usize..8).zip(&left[1..]) {
            chroma_block[(y + 1) * stride] = left;
        }
    }

    chroma_block[0] = if mby == 0 {
        127
    } else if mbx == 0 {
        129
    } else {
        left[0]
    };

    chroma_block
}

// Helper function for adding residue to a 4x4 sub-block
// from a contiguous block of 16
//
// Only 16 elements from rblock are used to add residue, so it is restricted to 16 elements
// to enable SIMD and other optimizations.
//
// Clippy suggests the clamp method, but it seems to optimize worse as of rustc 1.82.0 nightly.
#[allow(clippy::manual_clamp)]
#[inline(always)]
pub(crate) fn add_residue<const N: usize>(
    pblock: &mut [u8; N],
    rblock: &[i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    incant!(
        add_residue_dispatch(pblock.as_mut_slice(), rblock, y0, x0, stride),
        [v3, neon, scalar]
    );
}

#[cfg(target_arch = "x86_64")]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn add_residue_dispatch_v3(
    token: X64V3Token,
    pblock: &mut [u8],
    rblock: &[i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    add_residue_sse2(token, pblock, rblock, y0, x0, stride);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn add_residue_dispatch_neon(
    token: NeonToken,
    pblock: &mut [u8],
    rblock: &[i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    crate::common::transform::add_residue_neon(token, pblock, rblock, y0, x0, stride);
}

#[inline(always)]
fn add_residue_dispatch_scalar(
    _token: ScalarToken,
    pblock: &mut [u8],
    rblock: &[i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    let mut pos = y0 * stride + x0;
    for row in rblock.chunks(4) {
        for (p, &a) in pblock[pos..][..4].iter_mut().zip(row.iter()) {
            *p = (a + i32::from(*p)).clamp(0, 255) as u8;
        }
        pos += stride;
    }
}

/// SIMD implementation of add_residue using SSE2.
/// Processes one row at a time: load 4 u8, zero-extend, add i32, pack back to u8 with saturation.
#[archmage::arcane]
#[inline(always)]
fn add_residue_sse2(
    _token: archmage::X64V3Token,
    pblock: &mut [u8],
    rblock: &[i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    use archmage::intrinsics::x86_64 as simd_mem;
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let zero = _mm_setzero_si128();
    let mut pos = y0 * stride + x0;

    for row_idx in 0..4 {
        // Load 4 residual values (i32)
        let r = simd_mem::_mm_loadu_si128(
            <&[i32; 4]>::try_from(&rblock[row_idx * 4..row_idx * 4 + 4]).unwrap(),
        );

        // Load 4 prediction bytes using array conversion (avoids per-byte indexing)
        let p_arr: [u8; 4] = pblock[pos..pos + 4].try_into().unwrap();
        let p_bytes = i32::from_ne_bytes(p_arr);
        let p4 = _mm_cvtsi32_si128(p_bytes);
        // Unpack bytes to words, then words to dwords (zero extension)
        let p_words = _mm_unpacklo_epi8(p4, zero);
        let p = _mm_unpacklo_epi16(p_words, zero);

        // Add residuals
        let sum = _mm_add_epi32(p, r);

        // Pack back to u8 with saturation: i32 -> i16 -> u8
        let packed_16 = _mm_packs_epi32(sum, sum); // i32 -> i16 (saturate)
        let packed_8 = _mm_packus_epi16(packed_16, packed_16); // i16 -> u8 (saturate)

        // Store 4 bytes using array conversion (avoids per-byte indexing)
        let result = _mm_cvtsi128_si32(packed_8) as u32;
        pblock[pos..pos + 4].copy_from_slice(&result.to_ne_bytes());

        pos += stride;
    }
}

/// Add residue to prediction block AND clear the coefficient block in one operation.
/// This avoids a separate fill(0) call in the decoder hot path.
///
/// For hot loops, prefer `add_residue_and_clear_with_token` to avoid per-call token summoning.
///
/// Note: Currently unused - decoder now uses fused IDCT+add_residue.
/// Kept for potential encoder use or alternative decode paths.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn add_residue_and_clear<const N: usize>(
    pblock: &mut [u8; N],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    incant!(
        add_residue_and_clear_dispatch(pblock.as_mut_slice(), rblock, y0, x0, stride),
        [v3, scalar]
    );
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn add_residue_and_clear_dispatch_v3(
    token: X64V3Token,
    pblock: &mut [u8],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    add_residue_and_clear_sse2(token, pblock, rblock, y0, x0, stride);
}

#[allow(dead_code)]
#[inline(always)]
fn add_residue_and_clear_dispatch_scalar(
    _token: ScalarToken,
    pblock: &mut [u8],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    let mut pos = y0 * stride + x0;
    for row in rblock.chunks_mut(4) {
        for (p, coeff) in pblock[pos..][..4].iter_mut().zip(row.iter_mut()) {
            *p = (*coeff + i32::from(*p)).clamp(0, 255) as u8;
            *coeff = 0;
        }
        pos += stride;
    }
}

/// SIMD-accelerated add_residue_and_clear with a pre-summoned token.
/// Eliminates per-call token summoning overhead (~1.1B instructions saved over a full decode).
///
/// Note: Currently unused - decoder now uses fused IDCT+add_residue.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn add_residue_and_clear_with_token<const N: usize>(
    token: archmage::X64V3Token,
    pblock: &mut [u8; N],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    add_residue_and_clear_sse2(token, pblock.as_mut_slice(), rblock, y0, x0, stride);
}

/// Scalar fallback for add_residue_and_clear (used when no SIMD token available).
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn add_residue_and_clear_with_token<const N: usize>(
    _token: (),
    pblock: &mut [u8; N],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    add_residue_and_clear_scalar(pblock, rblock, y0, x0, stride);
}

/// Type alias for the SIMD token used in the decoder hot path.
#[cfg(target_arch = "x86_64")]
pub(crate) type SimdTokenType = Option<archmage::X64V3Token>;

#[cfg(target_arch = "aarch64")]
pub(crate) type SimdTokenType = Option<archmage::NeonToken>;

#[cfg(target_arch = "wasm32")]
pub(crate) type SimdTokenType = Option<archmage::Wasm128Token>;

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
#[allow(dead_code)]
pub(crate) type SimdTokenType = Option<()>;

#[allow(dead_code)]
#[inline(always)]
fn add_residue_and_clear_scalar<const N: usize>(
    pblock: &mut [u8; N],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    let mut pos = y0 * stride + x0;
    for row in rblock.chunks_mut(4) {
        for (p, coeff) in pblock[pos..][..4].iter_mut().zip(row.iter_mut()) {
            *p = (*coeff + i32::from(*p)).clamp(0, 255) as u8;
            *coeff = 0; // Clear as we go
        }
        pos += stride;
    }
}

#[archmage::arcane]
#[inline(always)]
fn add_residue_and_clear_sse2(
    _token: archmage::X64V3Token,
    pblock: &mut [u8],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    use archmage::intrinsics::x86_64 as simd_mem;
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    // Assert bounds upfront to elide checks in the loop
    let start = y0 * stride + x0;
    let end = start + 3 * stride + 4;
    assert!(end <= pblock.len(), "pblock too small for 4x4 block");

    let zero = _mm_setzero_si128();
    let mut pos = start;

    for row_idx in 0..4 {
        // Load 4 residual values (i32) - rblock is [i32; 16], indexing is always valid
        let rslice = &mut rblock[row_idx * 4..row_idx * 4 + 4];
        let r = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&*rslice).unwrap());

        // Clear coefficients with SIMD store
        simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(rslice).unwrap(), zero);

        // Load 4 prediction bytes - bounds proven by assert above
        let p_arr: [u8; 4] = pblock[pos..pos + 4].try_into().unwrap();
        let p_bytes = i32::from_ne_bytes(p_arr);
        let p4 = _mm_cvtsi32_si128(p_bytes);
        let p_words = _mm_unpacklo_epi8(p4, zero);
        let p = _mm_unpacklo_epi16(p_words, zero);

        // Add and saturate
        let sum = _mm_add_epi32(p, r);
        let packed_16 = _mm_packs_epi32(sum, sum);
        let packed_8 = _mm_packus_epi16(packed_16, packed_16);

        // Store 4 bytes - bounds proven by assert above
        let result = _mm_cvtsi128_si32(packed_8) as u32;
        pblock[pos..pos + 4].copy_from_slice(&result.to_ne_bytes());

        pos += stride;
    }
}

fn avg3(left: u8, this: u8, right: u8) -> u8 {
    let avg = (u16::from(left) + 2 * u16::from(this) + u16::from(right) + 2) >> 2;
    avg as u8
}

fn avg2(this: u8, right: u8) -> u8 {
    let avg = (u16::from(this) + u16::from(right) + 1) >> 1;
    avg as u8
}

pub(crate) fn predict_vpred<const N: usize>(
    a: &mut [u8; N],
    size: usize,
    x0: usize,
    y0: usize,
    stride: usize,
) {
    // Assert max access to enable bounds check elimination.
    assert!((y0 + size - 1) * stride + x0 + size <= N);
    assert!(y0 >= 1); // Need row y0-1 for the "above" row
    // This pass copies the top row to the rows below it.
    let (above, curr) = a.as_mut_slice().split_at_mut(stride * y0);
    let above_slice = &above[x0..][..size];

    for curr_chunk in curr.chunks_exact_mut(stride).take(size) {
        curr_chunk[x0..][..size].copy_from_slice(above_slice);
    }
}

pub(crate) fn predict_hpred<const N: usize>(
    a: &mut [u8; N],
    size: usize,
    x0: usize,
    y0: usize,
    stride: usize,
) {
    assert!((y0 + size - 1) * stride + x0 + size <= N);
    assert!(x0 >= 1); // Need column x0-1 for left pixel
    // This pass copies the first value of a row to the values right of it.
    for chunk in a
        .as_mut_slice()
        .chunks_exact_mut(stride)
        .skip(y0)
        .take(size)
    {
        let left = chunk[x0 - 1];
        chunk[x0..][..size].fill(left);
    }
}

#[inline(always)]
pub(crate) fn predict_dcpred<const N: usize>(
    a: &mut [u8; N],
    size: usize,
    stride: usize,
    above: bool,
    left: bool,
) {
    // Max access: 1 + stride * size + size - 1 = stride*size + size
    assert!(stride * (size + 1) <= N);
    let mut sum = 0u32;
    let mut shf = if size == 8 { 2u32 } else { 3u32 };

    if left {
        for y in 0usize..size {
            sum += u32::from(a[(y + 1) * stride]);
        }
        shf += 1;
    }

    if above {
        for x in 0usize..size {
            sum += u32::from(a[1 + x]);
        }
        shf += 1;
    }

    let dcval = if !left && !above {
        128u8
    } else {
        ((sum + (1 << (shf - 1))) >> shf) as u8
    };

    for y in 0usize..size {
        a[1 + stride * (y + 1)..][..size].fill(dcval);
    }
}

/// DC prediction for 16x16 luma block
/// Takes a bordered block buffer and fills the 16x16 prediction area
/// `x0` and `y0` specify the top-left corner in the buffer (typically 1,1 for bordered blocks)
#[inline]
#[allow(dead_code)] // Available for potential future use
pub(crate) fn predict_dc_16x16<const N: usize>(
    a: &mut [u8; N],
    stride: usize,
    x0: usize,
    y0: usize,
) {
    let above = y0 > 0;
    let left = x0 > 0;

    let mut sum = 0u32;
    let mut shf = 3; // log2(16) - 1 = 3

    if left {
        for y in 0..16 {
            sum += u32::from(a[(y0 + y) * stride + (x0 - 1)]);
        }
        shf += 1;
    }

    if above {
        for x in 0..16 {
            sum += u32::from(a[(y0 - 1) * stride + x0 + x]);
        }
        shf += 1;
    }

    let dcval = if !left && !above {
        128u8
    } else {
        ((sum + (1 << (shf - 1))) >> shf) as u8
    };

    for y in 0..16 {
        for x in 0..16 {
            a[(y0 + y) * stride + x0 + x] = dcval;
        }
    }
}

/// DC prediction for 8x8 chroma block
/// Takes a bordered block buffer and fills the 8x8 prediction area
/// `x0` and `y0` specify the top-left corner in the buffer (typically 1,1 for bordered blocks)
#[inline]
#[allow(dead_code)] // Available for potential future use
pub(crate) fn predict_dc_8x8<const N: usize>(a: &mut [u8; N], stride: usize, x0: usize, y0: usize) {
    let above = y0 > 0;
    let left = x0 > 0;

    let mut sum = 0u32;
    let mut shf = 2; // log2(8) - 1 = 2

    if left {
        for y in 0..8 {
            sum += u32::from(a[(y0 + y) * stride + (x0 - 1)]);
        }
        shf += 1;
    }

    if above {
        for x in 0..8 {
            sum += u32::from(a[(y0 - 1) * stride + x0 + x]);
        }
        shf += 1;
    }

    let dcval = if !left && !above {
        128u8
    } else {
        ((sum + (1 << (shf - 1))) >> shf) as u8
    };

    for y in 0..8 {
        for x in 0..8 {
            a[(y0 + y) * stride + x0 + x] = dcval;
        }
    }
}

// Clippy suggests the clamp method, but it seems to optimize worse as of rustc 1.82.0 nightly.
#[allow(clippy::manual_clamp)]
#[inline(always)]
pub(crate) fn predict_tmpred<const N: usize>(
    a: &mut [u8; N],
    size: usize,
    x0: usize,
    y0: usize,
    stride: usize,
) {
    // The formula for tmpred is:
    // X_ij = L_i + A_j - P (i, j=0, 1, 2, 3)
    //
    // |-----|-----|-----|-----|-----|
    // | P   | A0  | A1  | A2  | A3  |
    // |-----|-----|-----|-----|-----|
    // | L0  | X00 | X01 | X02 | X03 |
    // |-----|-----|-----|-----|-----|
    // | L1  | X10 | X11 | X12 | X13 |
    // |-----|-----|-----|-----|-----|
    // | L2  | X20 | X21 | X22 | X23 |
    // |-----|-----|-----|-----|-----|
    // | L3  | X30 | X31 | X32 | X33 |
    // |-----|-----|-----|-----|-----|
    // Diagram from p. 52 of RFC 6386

    // Assert max access for bounds check elimination.
    assert!((y0 + size - 1) * stride + x0 + size <= N);
    assert!(y0 >= 1 && x0 >= 1);

    // Split at L0
    let (above, x_block) = a.as_mut_slice().split_at_mut(y0 * stride + (x0 - 1));
    let p = i32::from(above[(y0 - 1) * stride + x0 - 1]);
    let above_slice = &above[(y0 - 1) * stride + x0..];

    for y in 0usize..size {
        let left_minus_p = i32::from(x_block[y * stride]) - p;

        // Add 1 to skip over L0 byte
        x_block[y * stride + 1..][..size]
            .iter_mut()
            .zip(above_slice)
            .for_each(|(cur, &abv)| *cur = (left_minus_p + i32::from(abv)).max(0).min(255) as u8);
    }
}

pub(crate) fn predict_bdcpred(a: &mut [u8; LUMA_BLOCK_SIZE], x0: usize, y0: usize, stride: usize) {
    // Assert max index to eliminate all interior bounds checks.
    // Max read: (y0+3)*stride + x0 + 3 for the 4x4 block,
    // plus (y0-1)*stride + x0+3 for the top border.
    assert!((y0 + 3) * stride + x0 + 3 < LUMA_BLOCK_SIZE);
    assert!(x0 >= 1); // left border pixel at x0-1
    assert!(y0 >= 1); // top border row at y0-1

    let mut v = 4;

    a[(y0 - 1) * stride + x0..][..4]
        .iter()
        .for_each(|&a| v += u32::from(a));

    for i in 0usize..4 {
        v += u32::from(a[(y0 + i) * stride + x0 - 1]);
    }

    v >>= 3;
    for chunk in a.as_mut_slice().chunks_exact_mut(stride).skip(y0).take(4) {
        for ch in &mut chunk[x0..][..4] {
            *ch = v as u8;
        }
    }
}

fn topleft_pixel(a: &[u8; LUMA_BLOCK_SIZE], x0: usize, y0: usize, stride: usize) -> u8 {
    a[(y0 - 1) * stride + x0 - 1]
}

#[inline(always)]
fn top_pixels(
    a: &[u8; LUMA_BLOCK_SIZE],
    x0: usize,
    y0: usize,
    stride: usize,
) -> (u8, u8, u8, u8, u8, u8, u8, u8) {
    let pos = (y0 - 1) * stride + x0;
    // Single bounds assertion: max access is pos+7, which must be < LUMA_BLOCK_SIZE.
    assert!(pos + 7 < LUMA_BLOCK_SIZE);
    let a0 = a[pos];
    let a1 = a[pos + 1];
    let a2 = a[pos + 2];
    let a3 = a[pos + 3];
    let a4 = a[pos + 4];
    let a5 = a[pos + 5];
    let a6 = a[pos + 6];
    let a7 = a[pos + 7];

    (a0, a1, a2, a3, a4, a5, a6, a7)
}

#[inline(always)]
fn left_pixels(a: &[u8; LUMA_BLOCK_SIZE], x0: usize, y0: usize, stride: usize) -> (u8, u8, u8, u8) {
    // Max access: (y0+3)*stride + x0 - 1
    assert!((y0 + 3) * stride + x0 > 0);
    assert!((y0 + 3) * stride + x0 - 1 < LUMA_BLOCK_SIZE);
    let l0 = a[y0 * stride + x0 - 1];
    let l1 = a[(y0 + 1) * stride + x0 - 1];
    let l2 = a[(y0 + 2) * stride + x0 - 1];
    let l3 = a[(y0 + 3) * stride + x0 - 1];

    (l0, l1, l2, l3)
}

#[inline(always)]
fn edge_pixels(
    a: &[u8; LUMA_BLOCK_SIZE],
    x0: usize,
    y0: usize,
    stride: usize,
) -> (u8, u8, u8, u8, u8, u8, u8, u8, u8) {
    let pos = (y0 - 1) * stride + x0 - 1;
    // Max access: pos + 4*stride (for e0) and pos + 4 (for e8).
    // Both must be < LUMA_BLOCK_SIZE.
    assert!(pos + 4 * stride < LUMA_BLOCK_SIZE);
    let e0 = a[pos + 4 * stride];
    let e1 = a[pos + 3 * stride];
    let e2 = a[pos + 2 * stride];
    let e3 = a[pos + stride];
    let e4 = a[pos];
    let e5 = a[pos + 1];
    let e6 = a[pos + 2];
    let e7 = a[pos + 3];
    let e8 = a[pos + 4];

    (e0, e1, e2, e3, e4, e5, e6, e7, e8)
}

pub(crate) fn predict_bvepred(a: &mut [u8; LUMA_BLOCK_SIZE], x0: usize, y0: usize, stride: usize) {
    // Assert max write: (y0+3)*stride + x0 + 3
    assert!((y0 + 3) * stride + x0 + 3 < LUMA_BLOCK_SIZE);
    let p = topleft_pixel(a, x0, y0, stride);
    let (a0, a1, a2, a3, a4, ..) = top_pixels(a, x0, y0, stride);
    let avg_1 = avg3(p, a0, a1);
    let avg_2 = avg3(a0, a1, a2);
    let avg_3 = avg3(a1, a2, a3);
    let avg_4 = avg3(a2, a3, a4);

    let avg = [avg_1, avg_2, avg_3, avg_4];

    let mut pos = y0 * stride + x0;
    for _ in 0..4 {
        a[pos..=pos + 3].copy_from_slice(&avg);
        pos += stride;
    }
}

pub(crate) fn predict_bhepred(a: &mut [u8; LUMA_BLOCK_SIZE], x0: usize, y0: usize, stride: usize) {
    assert!((y0 + 3) * stride + x0 + 3 < LUMA_BLOCK_SIZE);
    let p = topleft_pixel(a, x0, y0, stride);
    let (l0, l1, l2, l3) = left_pixels(a, x0, y0, stride);

    let avgs = [
        avg3(p, l0, l1),
        avg3(l0, l1, l2),
        avg3(l1, l2, l3),
        avg3(l2, l3, l3),
    ];

    let mut pos = y0 * stride + x0;
    for avg in avgs {
        for a_p in &mut a[pos..=pos + 3] {
            *a_p = avg;
        }
        pos += stride;
    }
}

pub(crate) fn predict_bldpred(a: &mut [u8; LUMA_BLOCK_SIZE], x0: usize, y0: usize, stride: usize) {
    assert!((y0 + 3) * stride + x0 + 3 < LUMA_BLOCK_SIZE);
    let (a0, a1, a2, a3, a4, a5, a6, a7) = top_pixels(a, x0, y0, stride);

    let avgs = [
        avg3(a0, a1, a2),
        avg3(a1, a2, a3),
        avg3(a2, a3, a4),
        avg3(a3, a4, a5),
        avg3(a4, a5, a6),
        avg3(a5, a6, a7),
        avg3(a6, a7, a7),
    ];

    let mut pos = y0 * stride + x0;

    for i in 0..4 {
        a[pos..=pos + 3].copy_from_slice(&avgs[i..=i + 3]);
        pos += stride;
    }
}

pub(crate) fn predict_brdpred(a: &mut [u8; LUMA_BLOCK_SIZE], x0: usize, y0: usize, stride: usize) {
    assert!((y0 + 3) * stride + x0 + 3 < LUMA_BLOCK_SIZE);
    let (e0, e1, e2, e3, e4, e5, e6, e7, e8) = edge_pixels(a, x0, y0, stride);

    let avgs = [
        avg3(e0, e1, e2),
        avg3(e1, e2, e3),
        avg3(e2, e3, e4),
        avg3(e3, e4, e5),
        avg3(e4, e5, e6),
        avg3(e5, e6, e7),
        avg3(e6, e7, e8),
    ];
    let mut pos = y0 * stride + x0;

    for i in 0..4 {
        a[pos..=pos + 3].copy_from_slice(&avgs[3 - i..7 - i]);
        pos += stride;
    }
}

pub(crate) fn predict_bvrpred(a: &mut [u8; LUMA_BLOCK_SIZE], x0: usize, y0: usize, stride: usize) {
    assert!((y0 + 3) * stride + x0 + 3 < LUMA_BLOCK_SIZE);
    let (_, e1, e2, e3, e4, e5, e6, e7, e8) = edge_pixels(a, x0, y0, stride);

    a[(y0 + 3) * stride + x0] = avg3(e1, e2, e3);
    a[(y0 + 2) * stride + x0] = avg3(e2, e3, e4);
    a[(y0 + 3) * stride + x0 + 1] = avg3(e3, e4, e5);
    a[(y0 + 1) * stride + x0] = avg3(e3, e4, e5);
    a[(y0 + 2) * stride + x0 + 1] = avg2(e4, e5);
    a[y0 * stride + x0] = avg2(e4, e5);
    a[(y0 + 3) * stride + x0 + 2] = avg3(e4, e5, e6);
    a[(y0 + 1) * stride + x0 + 1] = avg3(e4, e5, e6);
    a[(y0 + 2) * stride + x0 + 2] = avg2(e5, e6);
    a[y0 * stride + x0 + 1] = avg2(e5, e6);
    a[(y0 + 3) * stride + x0 + 3] = avg3(e5, e6, e7);
    a[(y0 + 1) * stride + x0 + 2] = avg3(e5, e6, e7);
    a[(y0 + 2) * stride + x0 + 3] = avg2(e6, e7);
    a[y0 * stride + x0 + 2] = avg2(e6, e7);
    a[(y0 + 1) * stride + x0 + 3] = avg3(e6, e7, e8);
    a[y0 * stride + x0 + 3] = avg2(e7, e8);
}

pub(crate) fn predict_bvlpred(a: &mut [u8; LUMA_BLOCK_SIZE], x0: usize, y0: usize, stride: usize) {
    assert!((y0 + 3) * stride + x0 + 3 < LUMA_BLOCK_SIZE);
    let (a0, a1, a2, a3, a4, a5, a6, a7) = top_pixels(a, x0, y0, stride);

    a[y0 * stride + x0] = avg2(a0, a1);
    a[(y0 + 1) * stride + x0] = avg3(a0, a1, a2);
    a[(y0 + 2) * stride + x0] = avg2(a1, a2);
    a[y0 * stride + x0 + 1] = avg2(a1, a2);
    a[(y0 + 1) * stride + x0 + 1] = avg3(a1, a2, a3);
    a[(y0 + 3) * stride + x0] = avg3(a1, a2, a3);
    a[(y0 + 2) * stride + x0 + 1] = avg2(a2, a3);
    a[y0 * stride + x0 + 2] = avg2(a2, a3);
    a[(y0 + 3) * stride + x0 + 1] = avg3(a2, a3, a4);
    a[(y0 + 1) * stride + x0 + 2] = avg3(a2, a3, a4);
    a[(y0 + 2) * stride + x0 + 2] = avg2(a3, a4);
    a[y0 * stride + x0 + 3] = avg2(a3, a4);
    a[(y0 + 3) * stride + x0 + 2] = avg3(a3, a4, a5);
    a[(y0 + 1) * stride + x0 + 3] = avg3(a3, a4, a5);
    a[(y0 + 2) * stride + x0 + 3] = avg3(a4, a5, a6);
    a[(y0 + 3) * stride + x0 + 3] = avg3(a5, a6, a7);
}

pub(crate) fn predict_bhdpred(a: &mut [u8; LUMA_BLOCK_SIZE], x0: usize, y0: usize, stride: usize) {
    assert!((y0 + 3) * stride + x0 + 3 < LUMA_BLOCK_SIZE);
    let (e0, e1, e2, e3, e4, e5, e6, e7, _) = edge_pixels(a, x0, y0, stride);

    a[(y0 + 3) * stride + x0] = avg2(e0, e1);
    a[(y0 + 3) * stride + x0 + 1] = avg3(e0, e1, e2);
    a[(y0 + 2) * stride + x0] = avg2(e1, e2);
    a[(y0 + 3) * stride + x0 + 2] = avg2(e1, e2);
    a[(y0 + 2) * stride + x0 + 1] = avg3(e1, e2, e3);
    a[(y0 + 3) * stride + x0 + 3] = avg3(e1, e2, e3);
    a[(y0 + 2) * stride + x0 + 2] = avg2(e2, e3);
    a[(y0 + 1) * stride + x0] = avg2(e2, e3);
    a[(y0 + 2) * stride + x0 + 3] = avg3(e2, e3, e4);
    a[(y0 + 1) * stride + x0 + 1] = avg3(e2, e3, e4);
    a[(y0 + 1) * stride + x0 + 2] = avg2(e3, e4);
    a[y0 * stride + x0] = avg2(e3, e4);
    a[(y0 + 1) * stride + x0 + 3] = avg3(e3, e4, e5);
    a[y0 * stride + x0 + 1] = avg3(e3, e4, e5);
    a[y0 * stride + x0 + 2] = avg3(e4, e5, e6);
    a[y0 * stride + x0 + 3] = avg3(e5, e6, e7);
}

pub(crate) fn predict_bhupred(a: &mut [u8; LUMA_BLOCK_SIZE], x0: usize, y0: usize, stride: usize) {
    assert!((y0 + 3) * stride + x0 + 3 < LUMA_BLOCK_SIZE);
    let (l0, l1, l2, l3) = left_pixels(a, x0, y0, stride);

    a[y0 * stride + x0] = avg2(l0, l1);
    a[y0 * stride + x0 + 1] = avg3(l0, l1, l2);
    a[y0 * stride + x0 + 2] = avg2(l1, l2);
    a[(y0 + 1) * stride + x0] = avg2(l1, l2);
    a[y0 * stride + x0 + 3] = avg3(l1, l2, l3);
    a[(y0 + 1) * stride + x0 + 1] = avg3(l1, l2, l3);
    a[(y0 + 1) * stride + x0 + 2] = avg2(l2, l3);
    a[(y0 + 2) * stride + x0] = avg2(l2, l3);
    a[(y0 + 1) * stride + x0 + 3] = avg3(l2, l3, l3);
    a[(y0 + 2) * stride + x0 + 1] = avg3(l2, l3, l3);
    a[(y0 + 2) * stride + x0 + 2] = l3;
    a[(y0 + 2) * stride + x0 + 3] = l3;
    a[(y0 + 3) * stride + x0] = l3;
    a[(y0 + 3) * stride + x0 + 1] = l3;
    a[(y0 + 3) * stride + x0 + 2] = l3;
    a[(y0 + 3) * stride + x0 + 3] = l3;
}

/// Pre-computed I4 predictions for all 10 modes
/// Each mode is stored as 16 bytes in row-major order
/// Order: DC, TM, VE, HE, LD, RD, VR, VL, HD, HU (matches IntraMode enum)
#[derive(Clone, Copy)]
pub(crate) struct I4Predictions {
    pub data: [[u8; 16]; 10],
}

impl I4Predictions {
    /// Pre-compute all 10 I4 prediction modes from the source block with borders
    /// `src` is the source block with borders, `x0` and `y0` are the prediction start coordinates
    #[inline]
    pub fn compute(src: &[u8; LUMA_BLOCK_SIZE], x0: usize, y0: usize, stride: usize) -> Self {
        let mut data = [[0u8; 16]; 10];

        // Read all edge pixels once
        let p = src[(y0 - 1) * stride + x0 - 1]; // top-left corner

        // Top row (8 pixels for LD mode)
        let top_pos = (y0 - 1) * stride + x0;
        let a0 = src[top_pos];
        let a1 = src[top_pos + 1];
        let a2 = src[top_pos + 2];
        let a3 = src[top_pos + 3];
        let a4 = src[top_pos + 4];
        let a5 = src[top_pos + 5];
        let a6 = src[top_pos + 6];
        let a7 = src[top_pos + 7];

        // Left column (4 pixels)
        let l0 = src[y0 * stride + x0 - 1];
        let l1 = src[(y0 + 1) * stride + x0 - 1];
        let l2 = src[(y0 + 2) * stride + x0 - 1];
        let l3 = src[(y0 + 3) * stride + x0 - 1];

        // Edge pixels for RD, VR, HD modes (diagonal)
        let e0 = src[(y0 + 3) * stride + x0 - 1]; // same as l3
        let e1 = l2;
        let e2 = l1;
        let e3 = l0;
        let e4 = p;
        let e5 = a0;
        let e6 = a1;
        let e7 = a2;
        let e8 = a3;

        // Mode 0: DC prediction
        {
            let mut v = 4u32;
            v += u32::from(a0) + u32::from(a1) + u32::from(a2) + u32::from(a3);
            v += u32::from(l0) + u32::from(l1) + u32::from(l2) + u32::from(l3);
            let dc = (v >> 3) as u8;
            data[0] = [dc; 16];
        }

        // Mode 1: TM prediction (fully unrolled for performance)
        {
            let p_i32 = i32::from(p);
            let l0_p = i32::from(l0) - p_i32;
            let l1_p = i32::from(l1) - p_i32;
            let l2_p = i32::from(l2) - p_i32;
            let l3_p = i32::from(l3) - p_i32;
            let a0_i = i32::from(a0);
            let a1_i = i32::from(a1);
            let a2_i = i32::from(a2);
            let a3_i = i32::from(a3);
            // Row 0: left = l0
            data[1][0] = (l0_p + a0_i).clamp(0, 255) as u8;
            data[1][1] = (l0_p + a1_i).clamp(0, 255) as u8;
            data[1][2] = (l0_p + a2_i).clamp(0, 255) as u8;
            data[1][3] = (l0_p + a3_i).clamp(0, 255) as u8;
            // Row 1: left = l1
            data[1][4] = (l1_p + a0_i).clamp(0, 255) as u8;
            data[1][5] = (l1_p + a1_i).clamp(0, 255) as u8;
            data[1][6] = (l1_p + a2_i).clamp(0, 255) as u8;
            data[1][7] = (l1_p + a3_i).clamp(0, 255) as u8;
            // Row 2: left = l2
            data[1][8] = (l2_p + a0_i).clamp(0, 255) as u8;
            data[1][9] = (l2_p + a1_i).clamp(0, 255) as u8;
            data[1][10] = (l2_p + a2_i).clamp(0, 255) as u8;
            data[1][11] = (l2_p + a3_i).clamp(0, 255) as u8;
            // Row 3: left = l3
            data[1][12] = (l3_p + a0_i).clamp(0, 255) as u8;
            data[1][13] = (l3_p + a1_i).clamp(0, 255) as u8;
            data[1][14] = (l3_p + a2_i).clamp(0, 255) as u8;
            data[1][15] = (l3_p + a3_i).clamp(0, 255) as u8;
        }

        // Mode 2: VE prediction (vertical edge)
        {
            let avg = [
                avg3(p, a0, a1),
                avg3(a0, a1, a2),
                avg3(a1, a2, a3),
                avg3(a2, a3, a4),
            ];
            for y in 0..4 {
                data[2][y * 4..y * 4 + 4].copy_from_slice(&avg);
            }
        }

        // Mode 3: HE prediction (horizontal edge)
        {
            let avgs = [
                avg3(p, l0, l1),
                avg3(l0, l1, l2),
                avg3(l1, l2, l3),
                avg3(l2, l3, l3),
            ];
            for y in 0..4 {
                data[3][y * 4..y * 4 + 4].fill(avgs[y]);
            }
        }

        // Mode 4: LD prediction (left-down diagonal)
        {
            let avgs = [
                avg3(a0, a1, a2),
                avg3(a1, a2, a3),
                avg3(a2, a3, a4),
                avg3(a3, a4, a5),
                avg3(a4, a5, a6),
                avg3(a5, a6, a7),
                avg3(a6, a7, a7),
            ];
            for y in 0..4 {
                data[4][y * 4..y * 4 + 4].copy_from_slice(&avgs[y..y + 4]);
            }
        }

        // Mode 5: RD prediction (right-down diagonal)
        {
            let avgs = [
                avg3(e0, e1, e2),
                avg3(e1, e2, e3),
                avg3(e2, e3, e4),
                avg3(e3, e4, e5),
                avg3(e4, e5, e6),
                avg3(e5, e6, e7),
                avg3(e6, e7, e8),
            ];
            for y in 0..4 {
                data[5][y * 4..y * 4 + 4].copy_from_slice(&avgs[3 - y..7 - y]);
            }
        }

        // Mode 6: VR prediction (vertical-right)
        {
            data[6][12] = avg3(e1, e2, e3);
            data[6][8] = avg3(e2, e3, e4);
            data[6][13] = avg3(e3, e4, e5);
            data[6][4] = avg3(e3, e4, e5);
            data[6][9] = avg2(e4, e5);
            data[6][0] = avg2(e4, e5);
            data[6][14] = avg3(e4, e5, e6);
            data[6][5] = avg3(e4, e5, e6);
            data[6][10] = avg2(e5, e6);
            data[6][1] = avg2(e5, e6);
            data[6][15] = avg3(e5, e6, e7);
            data[6][6] = avg3(e5, e6, e7);
            data[6][11] = avg2(e6, e7);
            data[6][2] = avg2(e6, e7);
            data[6][7] = avg3(e6, e7, e8);
            data[6][3] = avg2(e7, e8);
        }

        // Mode 7: VL prediction (vertical-left)
        {
            data[7][0] = avg2(a0, a1);
            data[7][4] = avg3(a0, a1, a2);
            data[7][8] = avg2(a1, a2);
            data[7][1] = avg2(a1, a2);
            data[7][5] = avg3(a1, a2, a3);
            data[7][12] = avg3(a1, a2, a3);
            data[7][9] = avg2(a2, a3);
            data[7][2] = avg2(a2, a3);
            data[7][13] = avg3(a2, a3, a4);
            data[7][6] = avg3(a2, a3, a4);
            data[7][10] = avg2(a3, a4);
            data[7][3] = avg2(a3, a4);
            data[7][14] = avg3(a3, a4, a5);
            data[7][7] = avg3(a3, a4, a5);
            data[7][11] = avg3(a4, a5, a6);
            data[7][15] = avg3(a5, a6, a7);
        }

        // Mode 8: HD prediction (horizontal-down)
        {
            data[8][12] = avg2(e0, e1);
            data[8][13] = avg3(e0, e1, e2);
            data[8][8] = avg2(e1, e2);
            data[8][14] = avg2(e1, e2);
            data[8][9] = avg3(e1, e2, e3);
            data[8][15] = avg3(e1, e2, e3);
            data[8][10] = avg2(e2, e3);
            data[8][4] = avg2(e2, e3);
            data[8][11] = avg3(e2, e3, e4);
            data[8][5] = avg3(e2, e3, e4);
            data[8][6] = avg2(e3, e4);
            data[8][0] = avg2(e3, e4);
            data[8][7] = avg3(e3, e4, e5);
            data[8][1] = avg3(e3, e4, e5);
            data[8][2] = avg3(e4, e5, e6);
            data[8][3] = avg3(e5, e6, e7);
        }

        // Mode 9: HU prediction (horizontal-up)
        {
            data[9][0] = avg2(l0, l1);
            data[9][1] = avg3(l0, l1, l2);
            data[9][2] = avg2(l1, l2);
            data[9][4] = avg2(l1, l2);
            data[9][3] = avg3(l1, l2, l3);
            data[9][5] = avg3(l1, l2, l3);
            data[9][6] = avg2(l2, l3);
            data[9][8] = avg2(l2, l3);
            data[9][7] = avg3(l2, l3, l3);
            data[9][9] = avg3(l2, l3, l3);
            data[9][10] = l3;
            data[9][11] = l3;
            data[9][12] = l3;
            data[9][13] = l3;
            data[9][14] = l3;
            data[9][15] = l3;
        }

        I4Predictions { data }
    }

    /// Get prediction for a specific mode (mode index 0-9)
    #[inline]
    pub fn get(&self, mode_idx: usize) -> &[u8; 16] {
        &self.data[mode_idx]
    }
}

/// Fused IDCT + add residue + clear, avoiding intermediate storage of IDCT output.
/// This is the hot path replacement for separate idct4x4 + add_residue_and_clear.
///
/// Takes raw DCT coefficients (NOT already IDCT'd), performs IDCT, adds to prediction,
/// and clears the coefficient block.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn idct_add_residue_and_clear_with_token<const N: usize>(
    token: archmage::X64V3Token,
    pblock: &mut [u8; N],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    // Check if block has AC coefficients (elements 1-15)
    // A block is DC-only if all AC coefficients are zero
    let dc_only = rblock[1..].iter().all(|&c| c == 0);

    // Use in-place fused IDCT + add residue
    use crate::common::transform;
    transform::idct_add_residue_inplace_with_token(
        token,
        rblock,
        pblock.as_mut_slice(),
        y0,
        x0,
        stride,
        dc_only,
    );
}

/// NEON fused IDCT + add residue + clear.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn idct_add_residue_and_clear_with_token<const N: usize>(
    token: archmage::NeonToken,
    pblock: &mut [u8; N],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    let dc_only = rblock[1..].iter().all(|&c| c == 0);
    crate::common::transform::idct_add_residue_inplace_neon(
        token,
        rblock,
        pblock.as_mut_slice(),
        y0,
        x0,
        stride,
        dc_only,
    );
}

/// WASM SIMD128 fused IDCT + add residue + clear.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(crate) fn idct_add_residue_and_clear_with_token<const N: usize>(
    token: archmage::Wasm128Token,
    pblock: &mut [u8; N],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    use crate::common::transform;

    let dc_only = rblock[1..].iter().all(|&c| c == 0);
    if dc_only {
        transform::idct4x4_dc(rblock);
    } else {
        crate::common::transform::idct4x4_wasm(token, rblock);
    }
    add_residue_and_clear_scalar(pblock, rblock, y0, x0, stride);
}

/// Scalar fallback for fused IDCT + add residue + clear.
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "x86",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
#[inline(always)]
pub(crate) fn idct_add_residue_and_clear_with_token<const N: usize>(
    _token: (),
    pblock: &mut [u8; N],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    use crate::common::transform;

    let dc_only = rblock[1..].iter().all(|&c| c == 0);
    if dc_only {
        transform::idct4x4_dc(rblock);
    } else {
        transform::idct4x4(rblock);
    }
    add_residue_and_clear_scalar(pblock, rblock, y0, x0, stride);
}

/// Fused IDCT + add residue + clear without pre-summoned token.
/// Falls back to separate operations if SIMD unavailable.
#[inline(always)]
pub(crate) fn idct_add_residue_and_clear<const N: usize>(
    pblock: &mut [u8; N],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    incant!(
        idct_add_residue_and_clear_dispatch(pblock.as_mut_slice(), rblock, y0, x0, stride),
        [v3, neon, wasm128, scalar]
    );
}

#[cfg(target_arch = "x86_64")]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn idct_add_residue_and_clear_dispatch_v3(
    token: X64V3Token,
    pblock: &mut [u8],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    use crate::common::transform;
    let dc_only = rblock[1..].iter().all(|&c| c == 0);
    transform::idct_add_residue_inplace_with_token(token, rblock, pblock, y0, x0, stride, dc_only);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn idct_add_residue_and_clear_dispatch_neon(
    token: NeonToken,
    pblock: &mut [u8],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    let dc_only = rblock[1..].iter().all(|&c| c == 0);
    crate::common::transform::idct_add_residue_inplace_neon(
        token, rblock, pblock, y0, x0, stride, dc_only,
    );
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn idct_add_residue_and_clear_dispatch_wasm128(
    _token: Wasm128Token,
    pblock: &mut [u8],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    use crate::common::transform;
    let dc_only = rblock[1..].iter().all(|&c| c == 0);
    if dc_only {
        transform::idct4x4_dc(rblock);
    } else {
        crate::common::transform::idct4x4_wasm(_token, rblock);
    }
    add_residue_and_clear_scalar_inner(pblock, rblock, y0, x0, stride);
}

#[inline(always)]
fn idct_add_residue_and_clear_dispatch_scalar(
    _token: ScalarToken,
    pblock: &mut [u8],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    use crate::common::transform;
    let dc_only = rblock[1..].iter().all(|&c| c == 0);
    if dc_only {
        transform::idct4x4_dc(rblock);
    } else {
        transform::idct4x4(rblock);
    }
    add_residue_and_clear_scalar_inner(pblock, rblock, y0, x0, stride);
}

/// Scalar add_residue_and_clear implementation for slice (non-generic).
#[inline(always)]
fn add_residue_and_clear_scalar_inner(
    pblock: &mut [u8],
    rblock: &mut [i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    let mut pos = y0 * stride + x0;
    for row in rblock.chunks_mut(4) {
        for (p, coeff) in pblock[pos..][..4].iter_mut().zip(row.iter_mut()) {
            *p = (*coeff + i32::from(*p)).clamp(0, 255) as u8;
            *coeff = 0;
        }
        pos += stride;
    }
}

#[cfg(all(test, feature = "_benchmarks"))]
mod benches {
    use super::*;
    use test::{Bencher, black_box};

    fn make_luma_ws() -> [u8; LUMA_BLOCK_SIZE] {
        let mut ws = [0u8; LUMA_BLOCK_SIZE];
        for (i, b) in ws.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        ws
    }

    #[bench]
    fn bench_predict_bvepred(b: &mut Bencher) {
        let mut ws = make_luma_ws();
        b.iter(|| {
            predict_bvepred(black_box(&mut ws), 5, 5, LUMA_STRIDE);
        });
    }

    #[bench]
    fn bench_predict_bldpred(b: &mut Bencher) {
        let mut ws = make_luma_ws();
        b.iter(|| {
            black_box(predict_bldpred(black_box(&mut ws), 5, 5, LUMA_STRIDE));
        });
    }

    #[bench]
    fn bench_predict_brdpred(b: &mut Bencher) {
        let mut ws = make_luma_ws();
        b.iter(|| {
            black_box(predict_brdpred(black_box(&mut ws), 5, 5, LUMA_STRIDE));
        });
    }

    #[bench]
    fn bench_predict_bhepred(b: &mut Bencher) {
        let mut ws = make_luma_ws();
        b.iter(|| {
            black_box(predict_bhepred(black_box(&mut ws), 5, 5, LUMA_STRIDE));
        });
    }

    #[bench]
    fn bench_top_pixels(b: &mut Bencher) {
        let ws = make_luma_ws();
        b.iter(|| {
            black_box(top_pixels(black_box(&ws), 5, 5, LUMA_STRIDE));
        });
    }

    #[bench]
    fn bench_edge_pixels(b: &mut Bencher) {
        let ws = make_luma_ws();
        b.iter(|| {
            black_box(edge_pixels(black_box(&ws), 5, 5, LUMA_STRIDE));
        });
    }
}

#[cfg(test)]
#[allow(clippy::erasing_op, clippy::identity_op)]
mod tests {
    use super::*;

    #[test]
    fn test_avg2() {
        for i in 0u8..=255 {
            for j in 0u8..=255 {
                let ceil_avg = (f32::from(i) + f32::from(j)) / 2.0;
                let ceil_avg = ceil_avg.ceil() as u8;
                assert_eq!(
                    ceil_avg,
                    avg2(i, j),
                    "avg2({}, {}), expected {}, got {}.",
                    i,
                    j,
                    ceil_avg,
                    avg2(i, j)
                );
            }
        }
    }

    #[test]
    fn test_avg2_specific() {
        assert_eq!(
            255,
            avg2(255, 255),
            "avg2(255, 255), expected 255, got {}.",
            avg2(255, 255)
        );
        assert_eq!(1, avg2(1, 1), "avg2(1, 1), expected 1, got {}.", avg2(1, 1));
        assert_eq!(2, avg2(2, 1), "avg2(2, 1), expected 2, got {}.", avg2(2, 1));
    }

    #[test]
    fn test_avg3() {
        for i in 0u8..=255 {
            for j in 0u8..=255 {
                for k in 0u8..=255 {
                    let floor_avg =
                        (2.0f32.mul_add(f32::from(j), f32::from(i)) + { f32::from(k) } + 2.0) / 4.0;
                    let floor_avg = floor_avg.floor() as u8;
                    assert_eq!(
                        floor_avg,
                        avg3(i, j, k),
                        "avg3({}, {}, {}), expected {}, got {}.",
                        i,
                        j,
                        k,
                        floor_avg,
                        avg3(i, j, k)
                    );
                }
            }
        }
    }

    #[test]
    fn test_edge_pixels() {
        // Use LUMA_BLOCK_SIZE array with LUMA_STRIDE
        // Layout: row 0 = top border, rows 1-16 = data, stride = LUMA_STRIDE
        let mut im = [0u8; LUMA_BLOCK_SIZE];
        let stride = LUMA_STRIDE;
        // Set up edge pixels at x0=1, y0=1:
        // top-left corner P at (0, 0) = row 0, col 0
        im[0 * stride + 0] = 5; // P (e4)
        im[0 * stride + 1] = 6; // A0 (e5)
        im[0 * stride + 2] = 7; // A1 (e6)
        im[0 * stride + 3] = 8; // A2 (e7)
        im[0 * stride + 4] = 9; // A3 (e8)
        im[1 * stride + 0] = 4; // L0 (e3)
        im[2 * stride + 0] = 3; // L1 (e2)
        im[3 * stride + 0] = 2; // L2 (e1)
        im[4 * stride + 0] = 1; // L3 (e0)

        let (e0, e1, e2, e3, e4, e5, e6, e7, e8) = edge_pixels(&im, 1, 1, stride);
        assert_eq!(e0, 1);
        assert_eq!(e1, 2);
        assert_eq!(e2, 3);
        assert_eq!(e3, 4);
        assert_eq!(e4, 5);
        assert_eq!(e5, 6);
        assert_eq!(e6, 7);
        assert_eq!(e7, 8);
        assert_eq!(e8, 9);
    }

    #[test]
    fn test_top_pixels() {
        let mut im = [0u8; LUMA_BLOCK_SIZE];
        let stride = LUMA_STRIDE;
        // top_pixels reads from row (y0-1) starting at col x0
        // With x0=1, y0=1: reads row 0, cols 1..=8
        im[0 * stride + 1] = 1;
        im[0 * stride + 2] = 2;
        im[0 * stride + 3] = 3;
        im[0 * stride + 4] = 4;
        im[0 * stride + 5] = 5;
        im[0 * stride + 6] = 6;
        im[0 * stride + 7] = 7;
        im[0 * stride + 8] = 8;

        let (e0, e1, e2, e3, e4, e5, e6, e7) = top_pixels(&im, 1, 1, stride);
        assert_eq!(e0, 1);
        assert_eq!(e1, 2);
        assert_eq!(e2, 3);
        assert_eq!(e3, 4);
        assert_eq!(e4, 5);
        assert_eq!(e5, 6);
        assert_eq!(e6, 7);
        assert_eq!(e7, 8);
    }

    #[test]
    fn test_add_residue() {
        let mut pblock = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let rblock = [
            -1, -2, -3, -4, 250, 249, 248, 250, -10, -18, -192, -17, -3, 15, 18, 9,
        ];
        let expected: [u8; 16] = [0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 10, 29, 33, 25];

        add_residue(&mut pblock, &rblock, 0, 0, 4);

        for (&e, &i) in expected.iter().zip(&pblock) {
            assert_eq!(e, i);
        }
    }

    #[test]
    fn test_predict_bhepred() {
        // Test bhepred using LUMA_BLOCK_SIZE buffer with LUMA_STRIDE
        let stride = LUMA_STRIDE;
        let mut im = [0u8; LUMA_BLOCK_SIZE];
        // Set up: P at (0,0), left column at col 0
        im[0 * stride + 0] = 5; // P
        im[1 * stride + 0] = 4; // L0
        im[2 * stride + 0] = 3; // L1
        im[3 * stride + 0] = 2; // L2
        im[4 * stride + 0] = 1; // L3

        predict_bhepred(&mut im, 1, 1, stride);

        // avg3(P, L0, L1) = avg3(5, 4, 3) = (5 + 8 + 3 + 2)/4 = 4
        assert_eq!(im[1 * stride + 1], 4);
        assert_eq!(im[1 * stride + 2], 4);
        assert_eq!(im[1 * stride + 3], 4);
        assert_eq!(im[1 * stride + 4], 4);
        // avg3(L0, L1, L2) = avg3(4, 3, 2) = 3
        assert_eq!(im[2 * stride + 1], 3);
        // avg3(L1, L2, L3) = avg3(3, 2, 1) = 2
        assert_eq!(im[3 * stride + 1], 2);
        // avg3(L2, L3, L3) = avg3(2, 1, 1) = 1
        assert_eq!(im[4 * stride + 1], 1);
    }

    #[test]
    fn test_predict_brdpred() {
        let stride = LUMA_STRIDE;
        let mut im = [0u8; LUMA_BLOCK_SIZE];
        // Set up edge pixels: e0..e8 = 1..9
        // P at (0,0)=5, top row A0..A3=6,7,8,9, left col L0..L3=4,3,2,1
        im[0 * stride + 0] = 5; // P (e4)
        im[0 * stride + 1] = 6; // A0 (e5)
        im[0 * stride + 2] = 7; // A1 (e6)
        im[0 * stride + 3] = 8; // A2 (e7)
        im[0 * stride + 4] = 9; // A3 (e8)
        im[1 * stride + 0] = 4; // L0 (e3)
        im[2 * stride + 0] = 3; // L1 (e2)
        im[3 * stride + 0] = 2; // L2 (e1)
        im[4 * stride + 0] = 1; // L3 (e0)

        predict_brdpred(&mut im, 1, 1, stride);

        // avgs[i] = avg3(e_i, e_{i+1}, e_{i+2}) = i+2 for sequential values
        // Row 0: avgs[3-0..7-0] = avgs[3..7] = [5, 6, 7, 8]
        assert_eq!(im[1 * stride + 1], 5);
        assert_eq!(im[1 * stride + 2], 6);
        assert_eq!(im[1 * stride + 3], 7);
        assert_eq!(im[1 * stride + 4], 8);
        // Row 1: avgs[2..6] = [4, 5, 6, 7]
        assert_eq!(im[2 * stride + 1], 4);
        assert_eq!(im[2 * stride + 2], 5);
        assert_eq!(im[2 * stride + 3], 6);
        assert_eq!(im[2 * stride + 4], 7);
        // Row 2: avgs[1..5] = [3, 4, 5, 6]
        assert_eq!(im[3 * stride + 1], 3);
        assert_eq!(im[3 * stride + 2], 4);
        assert_eq!(im[3 * stride + 3], 5);
        assert_eq!(im[3 * stride + 4], 6);
        // Row 3: avgs[0..4] = [2, 3, 4, 5]
        assert_eq!(im[4 * stride + 1], 2);
        assert_eq!(im[4 * stride + 2], 3);
        assert_eq!(im[4 * stride + 3], 4);
        assert_eq!(im[4 * stride + 4], 5);
    }

    #[test]
    fn test_predict_bldpred() {
        let stride = LUMA_STRIDE;
        let mut im = [0u8; LUMA_BLOCK_SIZE];
        // top_pixels reads row (y0-1), cols x0..x0+8
        // With x0=1, y0=1: row 0, cols 1..=8
        for i in 0..8 {
            im[0 * stride + 1 + i] = (i + 1) as u8;
        }

        predict_bldpred(&mut im, 1, 1, stride);

        // avg3(1,2,3)=2, avg3(2,3,4)=3, avg3(3,4,5)=4, avg3(4,5,6)=5
        assert_eq!(im[1 * stride + 1], 2);
        assert_eq!(im[1 * stride + 2], 3);
        assert_eq!(im[1 * stride + 3], 4);
        assert_eq!(im[1 * stride + 4], 5);
        // Row 1: shifted by 1
        assert_eq!(im[2 * stride + 1], 3);
        assert_eq!(im[2 * stride + 2], 4);
        assert_eq!(im[2 * stride + 3], 5);
        assert_eq!(im[2 * stride + 4], 6);
        // Row 2
        assert_eq!(im[3 * stride + 1], 4);
        assert_eq!(im[3 * stride + 2], 5);
        assert_eq!(im[3 * stride + 3], 6);
        assert_eq!(im[3 * stride + 4], 7);
        // Row 3
        assert_eq!(im[4 * stride + 1], 5);
        assert_eq!(im[4 * stride + 2], 6);
        assert_eq!(im[4 * stride + 3], 7);
        assert_eq!(im[4 * stride + 4], 8);
    }

    #[test]
    fn test_predict_bvepred() {
        let stride = LUMA_STRIDE;
        let mut im = [0u8; LUMA_BLOCK_SIZE];
        // P at (0,0), top row at (0, 1..=8)
        // avg3(P, A0, A1), avg3(A0, A1, A2), avg3(A1, A2, A3), avg3(A2, A3, A4)
        // With sequential values 1,2,3,4,5,6: avg3(1,2,3)=2, avg3(2,3,4)=3, etc.
        im[0 * stride + 0] = 1; // P
        for i in 0..8 {
            im[0 * stride + 1 + i] = (i + 2) as u8;
        }

        predict_bvepred(&mut im, 1, 1, stride);

        // avg3(1,2,3)=2, avg3(2,3,4)=3, avg3(3,4,5)=4, avg3(4,5,6)=5
        // All 4 rows should be identical
        for row in 1..=4 {
            assert_eq!(im[row * stride + 1], 2, "row {row} col 1");
            assert_eq!(im[row * stride + 2], 3, "row {row} col 2");
            assert_eq!(im[row * stride + 3], 4, "row {row} col 3");
            assert_eq!(im[row * stride + 4], 5, "row {row} col 4");
        }
    }
}
