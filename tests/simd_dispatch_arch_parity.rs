//! Per-function SIMD-vs-scalar parity check, intended to be run on x86
#![allow(clippy::needless_range_loop)]
//! and aarch64 (via QEMU) so we can spot which encoder SIMD entry point
//! diverges across architectures.
//!
//! Each test calls the public dispatcher (which on x86 picks SSE2, on
//! aarch64 picks NEON) AND directly calls the scalar reference, then
//! asserts they agree byte-for-byte. If they disagree on aarch64 but
//! agree on x86, the NEON implementation has a bug. If they disagree
//! on x86 but agree on aarch64, the SSE2 has a bug. If they agree on
//! both, the divergence must be elsewhere (DCT, quant, etc.).
//!
//! Methodology: we don't need to manually compare across architectures —
//! a per-platform "dispatched == scalar" assertion is sufficient. If any
//! arch fails the assertion, that's where the bug lives.

#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use zenwebp::encoder::cost::distortion::{
    is_flat_source_16, is_flat_source_16_scalar, tdisto_4x4, tdisto_8x8, tdisto_16x16,
};

// Re-exported test_helpers for cross-arch parity checks.
use zenwebp::test_helpers::{
    CHROMA_BLOCK_SIZE, LUMA_BLOCK_SIZE, dct4x4_dispatch, dct4x4_scalar,
    dequantize_block_dispatch, dequantize_block_scalar, ftransform_from_u8_4x4_dispatch,
    ftransform_from_u8_4x4_scalar, idct4x4_dispatch, idct4x4_scalar, is_flat_coeffs_dispatch,
    is_flat_coeffs_scalar, make_test_matrix, quantize_block_dispatch, quantize_block_scalar,
    quantize_dequantize_block_dispatch, quantize_dequantize_block_scalar, sse_8x8_chroma_dispatch,
    sse_8x8_chroma_scalar, sse_16x16_luma_dispatch, sse_16x16_luma_scalar, sse4x4_dispatch,
    sse4x4_with_residual_dispatch, sse4x4_with_residual_scalar,
};

const W_FAVOR_HIGH_FREQ: [u16; 16] = [
    16384, 1024, 1024, 256, 1024, 256, 256, 64, 1024, 256, 256, 64, 256, 64, 64, 16,
];

/// Reference scalar t_transform — copied verbatim from
/// src/encoder/cost/distortion.rs:t_transform_scalar so this test
/// doesn't depend on a private symbol.
fn t_transform_scalar(input: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let row = i * stride;
        let a0 = i32::from(input[row]) + i32::from(input[row + 2]);
        let a1 = i32::from(input[row + 1]) + i32::from(input[row + 3]);
        let a2 = i32::from(input[row + 1]) - i32::from(input[row + 3]);
        let a3 = i32::from(input[row]) - i32::from(input[row + 2]);
        tmp[i * 4] = a0 + a1;
        tmp[i * 4 + 1] = a3 + a2;
        tmp[i * 4 + 2] = a3 - a2;
        tmp[i * 4 + 3] = a0 - a1;
    }
    let mut out = [0i32; 16];
    for i in 0..4 {
        let a0 = tmp[i] + tmp[i + 8];
        let a1 = tmp[i + 4] + tmp[i + 12];
        let a2 = tmp[i + 4] - tmp[i + 12];
        let a3 = tmp[i] - tmp[i + 8];
        out[i] = a0 + a1;
        out[i + 4] = a3 + a2;
        out[i + 8] = a3 - a2;
        out[i + 12] = a0 - a1;
    }
    let mut sum = 0i32;
    for i in 0..16 {
        sum += (out[i].unsigned_abs() as i32) * (w[i] as i32);
    }
    sum
}

/// Pure scalar reference for tdisto_4x4. Matches the formula in
/// `src/encoder/cost/distortion.rs:tdisto_4x4_dispatch_scalar`:
///
/// ```text
/// (t_transform(b) - t_transform(a)).abs() >> 5
/// ```
fn tdisto_4x4_scalar_reference(a: &[u8], b: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    let sum1 = t_transform_scalar(a, stride, w);
    let sum2 = t_transform_scalar(b, stride, w);
    (sum2 - sum1).abs() >> 5
}

fn make_blocks_4x4(seed: u8) -> ([u8; 16], [u8; 16]) {
    let mut a = [0u8; 16];
    let mut b = [0u8; 16];
    for i in 0..16 {
        a[i] = seed.wrapping_add((i as u8).wrapping_mul(7));
        b[i] = seed
            .wrapping_add((i as u8).wrapping_mul(11))
            .wrapping_add(3);
    }
    (a, b)
}

#[test]
fn tdisto_4x4_dispatch_matches_scalar_reference() {
    let mut mismatches = Vec::new();
    for seed in 0..=255u8 {
        let (a, b) = make_blocks_4x4(seed);
        let dispatched = tdisto_4x4(&a, &b, 4, &W_FAVOR_HIGH_FREQ);
        let scalar = tdisto_4x4_scalar_reference(&a, &b, 4, &W_FAVOR_HIGH_FREQ);
        if dispatched != scalar {
            mismatches.push((seed, dispatched, scalar));
        }
    }
    assert!(
        mismatches.is_empty(),
        "tdisto_4x4 dispatched != scalar on {} of 256 seeds (showing first 5):\n  {}",
        mismatches.len(),
        mismatches
            .iter()
            .take(5)
            .map(|(s, d, sc)| format!("seed={s} dispatched={d} scalar={sc} diff={}", d - sc))
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

fn make_blocks_8x8(seed: u8) -> ([u8; 128], [u8; 128]) {
    // 8x8 with stride 16 → 128 bytes for 8 rows
    let mut a = [0u8; 128];
    let mut b = [0u8; 128];
    for i in 0..128 {
        a[i] = seed.wrapping_add((i as u8).wrapping_mul(13));
        b[i] = seed
            .wrapping_add((i as u8).wrapping_mul(17))
            .wrapping_add(11);
    }
    (a, b)
}

#[test]
fn tdisto_8x8_dispatch_matches_scalar_reference() {
    let mut mismatches = Vec::new();
    for seed in 0..=255u8 {
        let (a, b) = make_blocks_8x8(seed);
        let dispatched = tdisto_8x8(&a, &b, 16, &W_FAVOR_HIGH_FREQ);
        // Scalar reference: sum of tdisto_4x4 over 4 sub-blocks.
        let mut scalar = 0i32;
        for sub_y in 0..2 {
            for sub_x in 0..2 {
                let off = sub_y * 4 * 16 + sub_x * 4;
                scalar += tdisto_4x4_scalar_reference(&a[off..], &b[off..], 16, &W_FAVOR_HIGH_FREQ);
            }
        }
        if dispatched != scalar {
            mismatches.push((seed, dispatched, scalar));
        }
    }
    assert!(
        mismatches.is_empty(),
        "tdisto_8x8 dispatched != scalar on {} of 256 seeds (showing first 5):\n  {}",
        mismatches.len(),
        mismatches
            .iter()
            .take(5)
            .map(|(s, d, sc)| format!("seed={s} dispatched={d} scalar={sc} diff={}", d - sc))
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

fn make_blocks_16x16(seed: u8) -> ([u8; 256], [u8; 256]) {
    let mut a = [0u8; 256];
    let mut b = [0u8; 256];
    for i in 0..256 {
        a[i] = seed.wrapping_add((i as u8).wrapping_mul(19));
        b[i] = seed
            .wrapping_add((i as u8).wrapping_mul(23))
            .wrapping_add(13);
    }
    (a, b)
}

// ---------------------------------------------------------------------------
// sse4x4 / sse_16x16_luma / sse_8x8_chroma: dispatched vs scalar reference
// ---------------------------------------------------------------------------

fn sse_scalar_4x4(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    let mut sum = 0u32;
    for i in 0..16 {
        let d = (a[i] as i32 - b[i] as i32).unsigned_abs();
        sum += d * d;
    }
    sum
}

#[test]
fn sse4x4_dispatch_matches_scalar_reference() {
    let mut mismatches = Vec::new();
    for seed in 0..=255u8 {
        let mut a = [0u8; 16];
        let mut b = [0u8; 16];
        for i in 0..16 {
            a[i] = seed.wrapping_add((i as u8).wrapping_mul(7));
            b[i] = seed
                .wrapping_add((i as u8).wrapping_mul(11))
                .wrapping_add(3);
        }
        let dispatched = sse4x4_dispatch(&a, &b);
        let scalar = sse_scalar_4x4(&a, &b);
        if dispatched != scalar {
            mismatches.push((seed, dispatched, scalar));
        }
    }
    assert!(
        mismatches.is_empty(),
        "sse4x4 dispatched != scalar on {} of 256 seeds (first: {:?})",
        mismatches.len(),
        mismatches.first()
    );
}

// ---------------------------------------------------------------------------
// ftransform / idct: dispatched vs scalar reference
// ---------------------------------------------------------------------------

#[test]
fn ftransform_from_u8_4x4_dispatch_matches_scalar() {
    let mut mismatches = Vec::new();
    for seed in 0..=255u8 {
        let mut src = [0u8; 16];
        let mut pred = [0u8; 16];
        for i in 0..16 {
            src[i] = seed.wrapping_add((i as u8).wrapping_mul(7));
            pred[i] = seed
                .wrapping_add((i as u8).wrapping_mul(11))
                .wrapping_add(3);
        }
        let dispatched = ftransform_from_u8_4x4_dispatch(&src, &pred);
        let scalar = ftransform_from_u8_4x4_scalar(&src, &pred);
        if dispatched != scalar {
            mismatches.push((seed, dispatched, scalar));
        }
    }
    assert!(
        mismatches.is_empty(),
        "ftransform_from_u8_4x4 dispatched != scalar on {} of 256 seeds (first: seed={} disp={:?} scal={:?})",
        mismatches.len(),
        mismatches[0].0,
        &mismatches[0].1,
        &mismatches[0].2,
    );
}

#[test]
fn idct4x4_dispatch_matches_scalar() {
    let mut mismatches = Vec::new();
    for seed in 0..=255u8 {
        let mut block_disp = [0i32; 16];
        let mut block_scal = [0i32; 16];
        for i in 0..16 {
            let v = seed.wrapping_add((i as u8).wrapping_mul(13));
            block_disp[i] = (v as i32) - 128;
            block_scal[i] = block_disp[i];
        }
        idct4x4_dispatch(&mut block_disp);
        idct4x4_scalar(&mut block_scal);
        if block_disp != block_scal {
            mismatches.push(seed);
        }
    }
    assert!(
        mismatches.is_empty(),
        "idct4x4 dispatched != scalar on {} of 256 seeds",
        mismatches.len()
    );
}

// ---------------------------------------------------------------------------
// dct4x4: dispatched vs scalar
// ---------------------------------------------------------------------------

#[test]
fn dct4x4_dispatch_matches_scalar() {
    let mut mismatches = Vec::new();
    for seed in 0..=255u8 {
        let mut block_disp = [0i32; 16];
        let mut block_scal = [0i32; 16];
        for i in 0..16 {
            let v = seed.wrapping_add((i as u8).wrapping_mul(13));
            block_disp[i] = (v as i32) - 128;
            block_scal[i] = block_disp[i];
        }
        dct4x4_dispatch(&mut block_disp);
        dct4x4_scalar(&mut block_scal);
        if block_disp != block_scal {
            mismatches.push(seed);
        }
    }
    assert!(
        mismatches.is_empty(),
        "dct4x4 dispatched != scalar on {} of 256 seeds",
        mismatches.len()
    );
}

// ---------------------------------------------------------------------------
// quantize_dequantize_block: dispatched vs scalar (use_sharpen=false)
// ---------------------------------------------------------------------------

#[test]
fn quantize_dequantize_block_dispatch_matches_scalar() {
    let matrix = make_test_matrix(8, 12);
    let mut mismatches = Vec::new();
    for seed in 0..=255u8 {
        let mut coeffs = [0i32; 16];
        for i in 0..16 {
            let v = seed.wrapping_add((i as u8).wrapping_mul(13));
            coeffs[i] = (v as i32) - 128;
        }
        let mut q_disp = [0i32; 16];
        let mut dq_disp = [0i32; 16];
        let any_disp =
            quantize_dequantize_block_dispatch(&coeffs, &matrix, false, &mut q_disp, &mut dq_disp);

        let mut q_scal = [0i32; 16];
        let mut dq_scal = [0i32; 16];
        let any_scal =
            quantize_dequantize_block_scalar(&coeffs, &matrix, &mut q_scal, &mut dq_scal);

        if any_disp != any_scal || q_disp != q_scal || dq_disp != dq_scal {
            mismatches.push((seed, q_disp, q_scal, dq_disp, dq_scal));
        }
    }
    assert!(
        mismatches.is_empty(),
        "quantize_dequantize_block dispatched != scalar on {} of 256 seeds (first: seed={} q_disp={:?} q_scal={:?})",
        mismatches.len(),
        mismatches[0].0,
        &mismatches[0].1[..4],
        &mismatches[0].2[..4],
    );
}

// ---------------------------------------------------------------------------
// quantize_block (in-place): dispatched vs scalar (use_sharpen=false)
// ---------------------------------------------------------------------------

#[test]
fn quantize_block_dispatch_matches_scalar() {
    let matrix = make_test_matrix(8, 12);
    let mut mismatches = Vec::new();
    for seed in 0..=255u8 {
        let mut coeffs_disp = [0i32; 16];
        let mut coeffs_scal = [0i32; 16];
        for i in 0..16 {
            let v = seed.wrapping_add((i as u8).wrapping_mul(13));
            coeffs_disp[i] = (v as i32) - 128;
            coeffs_scal[i] = coeffs_disp[i];
        }
        let any_disp = quantize_block_dispatch(&mut coeffs_disp, &matrix, false);
        let any_scal = quantize_block_scalar(&mut coeffs_scal, &matrix);

        if any_disp != any_scal || coeffs_disp != coeffs_scal {
            mismatches.push((seed, coeffs_disp, coeffs_scal));
        }
    }
    assert!(
        mismatches.is_empty(),
        "quantize_block dispatched != scalar on {} of 256 seeds (first: seed={} disp={:?} scal={:?})",
        mismatches.len(),
        mismatches[0].0,
        &mismatches[0].1[..4],
        &mismatches[0].2[..4],
    );
}

// ---------------------------------------------------------------------------
// dequantize_block: dispatched vs scalar
// ---------------------------------------------------------------------------

#[test]
fn dequantize_block_dispatch_matches_scalar() {
    let matrix = make_test_matrix(8, 12);
    let mut mismatches = Vec::new();
    for seed in 0..=255u8 {
        let mut coeffs_disp = [0i32; 16];
        let mut coeffs_scal = [0i32; 16];
        for i in 0..16 {
            let v = seed.wrapping_add((i as u8).wrapping_mul(7));
            coeffs_disp[i] = (v as i32) - 128;
            coeffs_scal[i] = coeffs_disp[i];
        }
        dequantize_block_dispatch(&matrix, &mut coeffs_disp);
        dequantize_block_scalar(&matrix, &mut coeffs_scal);
        if coeffs_disp != coeffs_scal {
            mismatches.push((seed, coeffs_disp, coeffs_scal));
        }
    }
    assert!(
        mismatches.is_empty(),
        "dequantize_block dispatched != scalar on {} of 256 seeds (first: seed={})",
        mismatches.len(),
        mismatches[0].0,
    );
}

// ---------------------------------------------------------------------------
// sse4x4_with_residual: dispatched vs scalar
// ---------------------------------------------------------------------------

#[test]
fn sse4x4_with_residual_dispatch_matches_scalar() {
    let mut mismatches = Vec::new();
    for seed in 0..=255u8 {
        let mut src = [0u8; 16];
        let mut pred = [0u8; 16];
        let mut res = [0i32; 16];
        for i in 0..16 {
            src[i] = seed.wrapping_add((i as u8).wrapping_mul(7));
            pred[i] = seed
                .wrapping_add((i as u8).wrapping_mul(11))
                .wrapping_add(3);
            res[i] = (seed.wrapping_add((i as u8).wrapping_mul(13)) as i32) - 128;
        }
        let dispatched = sse4x4_with_residual_dispatch(&src, &pred, &res);
        let scalar = sse4x4_with_residual_scalar(&src, &pred, &res);
        if dispatched != scalar {
            mismatches.push((seed, dispatched, scalar));
        }
    }
    assert!(
        mismatches.is_empty(),
        "sse4x4_with_residual dispatched != scalar on {} of 256 seeds (first: {:?})",
        mismatches.len(),
        mismatches.first()
    );
}

// ---------------------------------------------------------------------------
// is_flat_coeffs: dispatched vs scalar
// ---------------------------------------------------------------------------

#[test]
fn is_flat_coeffs_dispatch_matches_scalar() {
    let mut mismatches = Vec::new();
    for seed in 0..=255u8 {
        let mut levels = [0i16; 16];
        for i in 0..16 {
            let v = seed.wrapping_add((i as u8).wrapping_mul(13));
            levels[i] = (v as i16) - 128;
        }
        for &thresh in &[0i32, 1, 5, 50, 200] {
            let dispatched = is_flat_coeffs_dispatch(&levels, 1, thresh);
            let scalar = is_flat_coeffs_scalar(&levels, 1, thresh);
            if dispatched != scalar {
                mismatches.push((seed, thresh, dispatched, scalar));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "is_flat_coeffs dispatched != scalar on {} cases (first: {:?})",
        mismatches.len(),
        mismatches.first()
    );
}

// ---------------------------------------------------------------------------
// sse_16x16_luma: dispatched vs scalar (uses bordered prediction buffer)
// ---------------------------------------------------------------------------

fn make_bordered_luma_pred(seed: u8) -> [u8; LUMA_BLOCK_SIZE] {
    let mut pred = [0u8; LUMA_BLOCK_SIZE];
    for i in 0..LUMA_BLOCK_SIZE {
        pred[i] = seed
            .wrapping_add((i as u8).wrapping_mul(11))
            .wrapping_add(3);
    }
    pred
}

#[test]
fn sse_16x16_luma_dispatch_matches_scalar() {
    let mut mismatches = Vec::new();
    for seed in 0..=255u8 {
        let mut src = [0u8; 16 * 16];
        for i in 0..256 {
            src[i] = seed.wrapping_add((i as u8).wrapping_mul(13));
        }
        let pred = make_bordered_luma_pred(seed);
        let dispatched = sse_16x16_luma_dispatch(&src, 16, 0, 0, &pred);
        let scalar = sse_16x16_luma_scalar(&src, 16, 0, 0, &pred);
        if dispatched != scalar {
            mismatches.push((seed, dispatched, scalar));
        }
    }
    assert!(
        mismatches.is_empty(),
        "sse_16x16_luma dispatched != scalar on {} of 256 seeds (first: {:?})",
        mismatches.len(),
        mismatches.first()
    );
}

// ---------------------------------------------------------------------------
// sse_8x8_chroma: dispatched vs scalar (bordered prediction buffer)
// ---------------------------------------------------------------------------

fn make_bordered_chroma_pred(seed: u8) -> [u8; CHROMA_BLOCK_SIZE] {
    let mut pred = [0u8; CHROMA_BLOCK_SIZE];
    for i in 0..CHROMA_BLOCK_SIZE {
        pred[i] = seed.wrapping_add((i as u8).wrapping_mul(7)).wrapping_add(5);
    }
    pred
}

#[test]
fn sse_8x8_chroma_dispatch_matches_scalar() {
    let mut mismatches = Vec::new();
    for seed in 0..=255u8 {
        let mut src = [0u8; 8 * 8];
        for i in 0..64 {
            src[i] = seed.wrapping_add((i as u8).wrapping_mul(13));
        }
        let pred = make_bordered_chroma_pred(seed);
        let dispatched = sse_8x8_chroma_dispatch(&src, 8, 0, 0, &pred);
        let scalar = sse_8x8_chroma_scalar(&src, 8, 0, 0, &pred);
        if dispatched != scalar {
            mismatches.push((seed, dispatched, scalar));
        }
    }
    assert!(
        mismatches.is_empty(),
        "sse_8x8_chroma dispatched != scalar on {} of 256 seeds (first: {:?})",
        mismatches.len(),
        mismatches.first()
    );
}

// ---------------------------------------------------------------------------
// is_flat_source_16: dispatched vs explicit scalar entry
// ---------------------------------------------------------------------------

#[test]
fn is_flat_source_16_dispatch_matches_scalar_reference() {
    let mut mismatches = Vec::new();
    for seed in 0..=255u8 {
        let mut src = vec![0u8; 256];
        for (i, b) in src.iter_mut().enumerate() {
            *b = seed.wrapping_add((i as u8).wrapping_mul(3));
        }
        let dispatched = is_flat_source_16(&src, 16);
        let scalar = is_flat_source_16_scalar(&src, 16);
        if dispatched != scalar {
            mismatches.push((seed, dispatched, scalar));
        }
    }
    // Also try the constant case (definitely flat)
    for seed in 0..=255u8 {
        let src = vec![seed; 256];
        let dispatched = is_flat_source_16(&src, 16);
        let scalar = is_flat_source_16_scalar(&src, 16);
        if dispatched != scalar {
            mismatches.push((seed, dispatched, scalar));
        }
    }
    assert!(
        mismatches.is_empty(),
        "is_flat_source_16 dispatched != scalar on {} cases (first: {:?})",
        mismatches.len(),
        mismatches.first()
    );
}

// ---------------------------------------------------------------------------

#[test]
fn tdisto_16x16_dispatch_matches_scalar_reference() {
    let mut mismatches = Vec::new();
    for seed in 0..=255u8 {
        let (a, b) = make_blocks_16x16(seed);
        let dispatched = tdisto_16x16(&a, &b, 16, &W_FAVOR_HIGH_FREQ);
        let mut scalar = 0i32;
        for sub_y in 0..4 {
            for sub_x in 0..4 {
                let off = sub_y * 4 * 16 + sub_x * 4;
                scalar += tdisto_4x4_scalar_reference(&a[off..], &b[off..], 16, &W_FAVOR_HIGH_FREQ);
            }
        }
        if dispatched != scalar {
            mismatches.push((seed, dispatched, scalar));
        }
    }
    assert!(
        mismatches.is_empty(),
        "tdisto_16x16 dispatched != scalar on {} of 256 seeds (showing first 5):\n  {}",
        mismatches.len(),
        mismatches
            .iter()
            .take(5)
            .map(|(s, d, sc)| format!("seed={s} dispatched={d} scalar={sc} diff={}", d - sc))
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}
