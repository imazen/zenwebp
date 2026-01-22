//! SIMD-optimized DCT/IDCT transforms using the `wide` crate.
//!
//! Uses i32x4 for 4-wide SIMD operations, falling back to scalar on unsupported platforms.
//! The `wide` crate provides cross-platform SIMD with SSE2/NEON/WASM backends.

use wide::{i32x4, i64x4};

/// SIMD forward DCT for two 4x4 blocks simultaneously.
/// This processes both blocks in parallel where possible.
#[allow(dead_code)] // Kept for potential future batch processing optimization
pub(crate) fn dct4x4_two(block1: &mut [i32; 16], block2: &mut [i32; 16]) {
    // Process pass 1 (row transform) for both blocks
    // Each row can use SIMD to load 4 elements, but the butterfly pattern
    // requires specific element access, so we use scalar with i32x4 for adds/subs

    // Pass 1: rows for block 1
    for i in 0..4 {
        let (out0, out1, out2, out3) = dct_row_pass1(
            block1[i * 4],
            block1[i * 4 + 1],
            block1[i * 4 + 2],
            block1[i * 4 + 3],
        );
        block1[i * 4] = out0;
        block1[i * 4 + 1] = out1;
        block1[i * 4 + 2] = out2;
        block1[i * 4 + 3] = out3;
    }

    // Pass 1: rows for block 2
    for i in 0..4 {
        let (out0, out1, out2, out3) = dct_row_pass1(
            block2[i * 4],
            block2[i * 4 + 1],
            block2[i * 4 + 2],
            block2[i * 4 + 3],
        );
        block2[i * 4] = out0;
        block2[i * 4 + 1] = out1;
        block2[i * 4 + 2] = out2;
        block2[i * 4 + 3] = out3;
    }

    // Pass 2: columns - can use SIMD to process both blocks together
    // Column i of block1 and column i of block2 can be processed as i32x8
    // But wide doesn't have i32x8 that works well, so process each block

    // Pass 2: columns for block 1
    for col in 0..4 {
        let v0 = block1[col];
        let v1 = block1[col + 4];
        let v2 = block1[col + 8];
        let v3 = block1[col + 12];

        let (out0, out1, out2, out3) = dct_col_pass2(v0, v1, v2, v3);

        block1[col] = out0;
        block1[col + 4] = out1;
        block1[col + 8] = out2;
        block1[col + 12] = out3;
    }

    // Pass 2: columns for block 2
    for col in 0..4 {
        let v0 = block2[col];
        let v1 = block2[col + 4];
        let v2 = block2[col + 8];
        let v3 = block2[col + 12];

        let (out0, out1, out2, out3) = dct_col_pass2(v0, v1, v2, v3);

        block2[col] = out0;
        block2[col + 4] = out1;
        block2[col + 8] = out2;
        block2[col + 12] = out3;
    }
}

/// SIMD forward DCT for a single 4x4 block using i32x4 operations.
pub(crate) fn dct4x4_simd(block: &mut [i32; 16]) {
    // Pass 1: Transform rows
    for i in 0..4 {
        let (out0, out1, out2, out3) = dct_row_pass1(
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        );
        block[i * 4] = out0;
        block[i * 4 + 1] = out1;
        block[i * 4 + 2] = out2;
        block[i * 4 + 3] = out3;
    }

    // Pass 2: Transform columns using SIMD for 4 columns at once
    // Load all 4 rows as i32x4
    let row0 = i32x4::new([block[0], block[1], block[2], block[3]]);
    let row1 = i32x4::new([block[4], block[5], block[6], block[7]]);
    let row2 = i32x4::new([block[8], block[9], block[10], block[11]]);
    let row3 = i32x4::new([block[12], block[13], block[14], block[15]]);

    // Butterfly operations on columns (all 4 columns in parallel)
    let a = row0 + row3;  // v0 + v3 for all columns
    let b = row1 + row2;  // v1 + v2 for all columns
    let c = row1 - row2;  // v1 - v2 for all columns
    let d = row0 - row3;  // v0 - v3 for all columns

    // out0 = (a + b + 7) >> 4
    let seven = i32x4::splat(7);
    let out0: i32x4 = (a + b + seven) >> 4;

    // out2 = (a - b + 7) >> 4
    let out2: i32x4 = (a - b + seven) >> 4;

    // For out1 and out3, we need 64-bit multiply-accumulate
    // out1 = ((c*2217 + d*5352 + 12000) >> 16) + (d != 0 ? 1 : 0)
    // out3 = (d*2217 - c*5352 + 51000) >> 16

    // Use mul_widen for 64-bit precision
    let c_wide = c.mul_widen(i32x4::splat(2217));
    let d_wide = d.mul_widen(i32x4::splat(5352));
    let sum1 = c_wide + d_wide + i64x4::splat(12000);

    // Shift right by 16 and add d != 0 adjustment
    let out1_base: i64x4 = sum1 >> 16;
    let out1_base_arr = out1_base.to_array();
    let d_arr = d.to_array();
    let out1_arr = [
        out1_base_arr[0] as i32 + if d_arr[0] != 0 { 1 } else { 0 },
        out1_base_arr[1] as i32 + if d_arr[1] != 0 { 1 } else { 0 },
        out1_base_arr[2] as i32 + if d_arr[2] != 0 { 1 } else { 0 },
        out1_base_arr[3] as i32 + if d_arr[3] != 0 { 1 } else { 0 },
    ];
    let out1 = i32x4::new(out1_arr);

    // out3 = (d*2217 - c*5352 + 51000) >> 16
    let d_wide2 = d.mul_widen(i32x4::splat(2217));
    let c_wide2 = c.mul_widen(i32x4::splat(5352));
    let sum3 = d_wide2 - c_wide2 + i64x4::splat(51000);
    let out3_shifted: i64x4 = sum3 >> 16;
    let out3_arr = out3_shifted.to_array();
    let out3 = i32x4::new([
        out3_arr[0] as i32,
        out3_arr[1] as i32,
        out3_arr[2] as i32,
        out3_arr[3] as i32,
    ]);

    // Store results - note output order: row0=out0, row1=out1, row2=out2, row3=out3
    let out0_arr = out0.to_array();
    let out1_arr = out1.to_array();
    let out2_arr = out2.to_array();
    let out3_arr = out3.to_array();

    // Row 0 (coefficients at positions 0-3)
    block[0] = out0_arr[0];
    block[1] = out0_arr[1];
    block[2] = out0_arr[2];
    block[3] = out0_arr[3];

    // Row 1 (coefficients at positions 4-7)
    block[4] = out1_arr[0];
    block[5] = out1_arr[1];
    block[6] = out1_arr[2];
    block[7] = out1_arr[3];

    // Row 2 (coefficients at positions 8-11)
    block[8] = out2_arr[0];
    block[9] = out2_arr[1];
    block[10] = out2_arr[2];
    block[11] = out2_arr[3];

    // Row 3 (coefficients at positions 12-15)
    block[12] = out3_arr[0];
    block[13] = out3_arr[1];
    block[14] = out3_arr[2];
    block[15] = out3_arr[3];
}

/// Row transform (pass 1) - matches scalar version exactly
#[inline(always)]
fn dct_row_pass1(d0: i32, d1: i32, d2: i32, d3: i32) -> (i32, i32, i32, i32) {
    let a = (i64::from(d0) + i64::from(d3)) * 8;
    let b = (i64::from(d1) + i64::from(d2)) * 8;
    let c = (i64::from(d1) - i64::from(d2)) * 8;
    let d = (i64::from(d0) - i64::from(d3)) * 8;

    let out0 = (a + b) as i32;
    let out2 = (a - b) as i32;
    let out1 = ((c * 2217 + d * 5352 + 14500) >> 12) as i32;
    let out3 = ((d * 2217 - c * 5352 + 7500) >> 12) as i32;

    (out0, out1, out2, out3)
}

/// Column transform (pass 2) - matches scalar version exactly
#[inline(always)]
#[allow(dead_code)] // Used by dct4x4_two
fn dct_col_pass2(v0: i32, v1: i32, v2: i32, v3: i32) -> (i32, i32, i32, i32) {
    let a = i64::from(v0) + i64::from(v3);
    let b = i64::from(v1) + i64::from(v2);
    let c = i64::from(v1) - i64::from(v2);
    let d = i64::from(v0) - i64::from(v3);

    let out0 = ((a + b + 7) >> 4) as i32;
    let out2 = ((a - b + 7) >> 4) as i32;
    let out1 = (((c * 2217 + d * 5352 + 12000) >> 16) + if d != 0 { 1 } else { 0 }) as i32;
    let out3 = ((d * 2217 - c * 5352 + 51000) >> 16) as i32;

    (out0, out1, out2, out3)
}

/// SIMD inverse DCT for a single 4x4 block.
pub(crate) fn idct4x4_simd(block: &mut [i32]) {
    debug_assert!(block.len() >= 16);

    const CONST1: i64 = 20091;
    const CONST2: i64 = 35468;

    // Pass 1: columns
    for i in 0..4 {
        let v0 = block[i] as i64;
        let v1 = block[4 + i] as i64;
        let v2 = block[8 + i] as i64;
        let v3 = block[12 + i] as i64;

        let a1 = v0 + v2;
        let b1 = v0 - v2;

        let t1 = (v1 * CONST2) >> 16;
        let t2 = v3 + ((v3 * CONST1) >> 16);
        let c1 = t1 - t2;

        let t1 = v1 + ((v1 * CONST1) >> 16);
        let t2 = (v3 * CONST2) >> 16;
        let d1 = t1 + t2;

        block[i] = (a1 + d1) as i32;
        block[4 + i] = (b1 + c1) as i32;
        block[12 + i] = (a1 - d1) as i32;
        block[8 + i] = (b1 - c1) as i32;
    }

    // Pass 2: rows
    // The butterfly pattern requires accessing specific positions within each row,
    // making it harder to vectorize across rows, so we process each row scalar.
    for i in 0..4 {
        let base = i * 4;
        let v0 = block[base] as i64;
        let v1 = block[base + 1] as i64;
        let v2 = block[base + 2] as i64;
        let v3 = block[base + 3] as i64;

        let a1 = v0 + v2;
        let b1 = v0 - v2;

        let t1 = (v1 * CONST2) >> 16;
        let t2 = v3 + ((v3 * CONST1) >> 16);
        let c1 = t1 - t2;

        let t1 = v1 + ((v1 * CONST1) >> 16);
        let t2 = (v3 * CONST2) >> 16;
        let d1 = t1 + t2;

        block[base] = ((a1 + d1 + 4) >> 3) as i32;
        block[base + 3] = ((a1 - d1 + 4) >> 3) as i32;
        block[base + 1] = ((b1 + c1 + 4) >> 3) as i32;
        block[base + 2] = ((b1 - c1 + 4) >> 3) as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_simd_matches_scalar() {
        let input: [i32; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];

        // Test scalar
        let mut scalar_block = input;
        crate::transform::dct4x4(&mut scalar_block);

        // Test SIMD
        let mut simd_block = input;
        dct4x4_simd(&mut simd_block);

        assert_eq!(scalar_block, simd_block, "SIMD DCT doesn't match scalar");
    }

    #[test]
    fn test_dct_two_blocks() {
        let input1: [i32; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];
        let input2: [i32; 16] = [
            100, 50, 25, 75, 200, 150, 100, 50, 25, 75, 125, 175, 225, 200, 150, 100,
        ];

        // Test scalar
        let mut scalar1 = input1;
        let mut scalar2 = input2;
        crate::transform::dct4x4(&mut scalar1);
        crate::transform::dct4x4(&mut scalar2);

        // Test two-block SIMD
        let mut simd1 = input1;
        let mut simd2 = input2;
        dct4x4_two(&mut simd1, &mut simd2);

        assert_eq!(scalar1, simd1, "Two-block SIMD block1 doesn't match scalar");
        assert_eq!(scalar2, simd2, "Two-block SIMD block2 doesn't match scalar");
    }

    #[test]
    fn test_idct_simd_matches_scalar() {
        // Start with DCT coefficients
        let mut input: [i32; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];
        crate::transform::dct4x4(&mut input);

        // Test scalar IDCT
        let mut scalar_block = input;
        crate::transform::idct4x4(&mut scalar_block);

        // Test SIMD IDCT
        let mut simd_block = input;
        idct4x4_simd(&mut simd_block);

        assert_eq!(scalar_block, simd_block, "SIMD IDCT doesn't match scalar");
    }
}

#[cfg(all(test, feature = "_benchmarks"))]
mod benchmarks {
    use super::*;
    use test::Bencher;

    const TEST_BLOCKS: [[i32; 16]; 4] = [
        [38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96],
        [100, 50, 25, 75, 200, 150, 100, 50, 25, 75, 125, 175, 225, 200, 150, 100],
        [12, 34, 56, 78, 90, 12, 34, 56, 78, 90, 12, 34, 56, 78, 90, 12],
        [255, 0, 128, 64, 192, 32, 224, 16, 240, 8, 248, 4, 252, 2, 254, 1],
    ];

    #[bench]
    fn bench_dct_scalar(b: &mut Bencher) {
        b.iter(|| {
            for input in &TEST_BLOCKS {
                let mut block = *input;
                test::black_box(crate::transform::dct4x4_scalar(&mut block));
            }
        });
    }

    #[bench]
    fn bench_dct_simd(b: &mut Bencher) {
        b.iter(|| {
            for input in &TEST_BLOCKS {
                let mut block = *input;
                test::black_box(dct4x4_simd(&mut block));
            }
        });
    }

    #[bench]
    fn bench_dct_two_blocks(b: &mut Bencher) {
        b.iter(|| {
            let mut block1 = TEST_BLOCKS[0];
            let mut block2 = TEST_BLOCKS[1];
            test::black_box(dct4x4_two(&mut block1, &mut block2));
            let mut block3 = TEST_BLOCKS[2];
            let mut block4 = TEST_BLOCKS[3];
            test::black_box(dct4x4_two(&mut block3, &mut block4));
        });
    }

    #[bench]
    fn bench_idct_scalar(b: &mut Bencher) {
        // Pre-compute DCT blocks
        let mut dct_blocks = TEST_BLOCKS;
        for block in &mut dct_blocks {
            crate::transform::dct4x4_scalar(block);
        }

        b.iter(|| {
            for input in &dct_blocks {
                let mut block = *input;
                test::black_box(crate::transform::idct4x4_scalar(&mut block));
            }
        });
    }

    #[bench]
    fn bench_idct_simd(b: &mut Bencher) {
        // Pre-compute DCT blocks
        let mut dct_blocks = TEST_BLOCKS;
        for block in &mut dct_blocks {
            crate::transform::dct4x4(block);
        }

        b.iter(|| {
            for input in &dct_blocks {
                let mut block = *input;
                test::black_box(idct4x4_simd(&mut block));
            }
        });
    }
}
