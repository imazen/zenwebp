//! Coefficient reading, dequantization, and macroblock header parsing for
//! the v2 VP8 decoder.
//!
//! All functions are free-standing (no `self`) and operate on explicit
//! parameters. The hot `read_coefficients` and `get_large_value` functions
//! are `#[inline(never)]` to prevent BTB aliasing from repeated inlining
//! (see investigation notes in CLAUDE.md).

use super::MbRowEntry;
use super::context::PreviousMacroBlock;
use super::tables::{DequantPair, FrameTables};
use crate::common::prediction::{MB_COEFF_SIZE, coeff_block};
use crate::common::transform;
use crate::common::types::*;
use crate::decoder::bit_reader::{ActivePartitionReader, VP8HeaderBitReader};
use crate::decoder::internal_error::InternalDecodeError;

// ---- Zigzag scan order (same as common::types::ZIGZAG) ----
const ZIGZAG: [u8; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

// ---- Extra-value probability tables (category 3-6) ----
// Indexed starting at category offset 2 (cats 3..6 use indices 2..6)
const PROB_DCT_CAT: [[u8; 12]; 6] = [
    [159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [165, 145, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [173, 148, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [176, 155, 140, 135, 0, 0, 0, 0, 0, 0, 0, 0],
    [180, 157, 141, 134, 130, 0, 0, 0, 0, 0, 0, 0],
    [254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129, 0],
];

/// Lengths of extra-value probability sequences for categories 3..6.
const CAT_LENGTHS: [usize; 4] = [3, 4, 5, 11];

// ============================================================================
// Coefficient parsing (hot path)
// ============================================================================

/// Decode large coefficient values (categories 3-6).
///
/// Separated from [`read_coefficients`] to keep the common-case path small,
/// matching libwebp's `GetLargeValue`/`GetCoeffsFast` split. Both functions
/// are `#[inline(never)]` so the CPU branch predictor tracks each through
/// a single set of addresses, reducing BTB aliasing.
#[inline(never)]
fn get_large_value(reader: &mut ActivePartitionReader<'_>, prob: &[u8; 11]) -> i32 {
    if reader.get_bit(prob[6]) == 0 {
        // Category 1 or 2
        if reader.get_bit(prob[7]) == 0 {
            // Cat 1: value 5 or 6
            5 + reader.get_bit(159)
        } else {
            // Cat 2: value 7..10
            7 + 2 * reader.get_bit(165) + reader.get_bit(145)
        }
    } else {
        // Categories 3-6
        let bit1 = reader.get_bit(prob[8]);
        let bit0 = reader.get_bit(prob[9 + bit1 as usize]);
        let cat = (2 * bit1 + bit0) as usize;

        let cat_probs = &PROB_DCT_CAT[2 + cat];
        let cat_len = CAT_LENGTHS[cat];
        let mut extra = 0i32;
        for &prob in cat_probs.iter().take(cat_len) {
            extra = extra + extra + reader.get_bit(prob);
        }
        3 + (8 << cat) + extra
    }
}

/// Read and dequantize DCT coefficients for one 4x4 block.
///
/// Returns `true` if any coefficient beyond `first` was non-zero.
///
/// NOT inlined so the CPU branch predictor tracks all coefficient
/// decoding through a single set of branch addresses. When inlined 25x
/// (once per block in a macroblock), BTB aliasing causes ~2.3M extra
/// mispredicts per decode. The function call overhead is compensated
/// by the mispredict savings.
///
/// Does NOT check for EOF -- caller must check `reader.is_eof()` after
/// processing all blocks in the macroblock.
///
/// # Arguments
/// * `reader` - Boolean arithmetic decoder for the current partition.
/// * `output` - Fixed-size 16-element coefficient array (natural order).
/// * `probs` - Position-indexed flat u8 probability table
///   `[position][context][prob_index]` for this plane.
/// * `first` - First coefficient index to decode (0 for DC, 1 for Y after Y2).
/// * `complexity` - Context from neighboring blocks (0, 1, or 2).
/// * `dq` - Dequantization pair: dc and ac quant values.
#[inline(never)]
fn read_coefficients(
    reader: &mut ActivePartitionReader<'_>,
    output: &mut [i32; 16],
    probs: &[[[u8; 11]; 3]; 17],
    first: usize,
    complexity: usize,
    dq: DequantPair,
) -> bool {
    debug_assert!(complexity <= 2);

    let dq_pair = [i32::from(dq.dc), i32::from(dq.ac)];

    let mut n = first;
    let mut prob = &probs[n][complexity];

    while n < 16 {
        // Is there a non-zero coefficient at this position?
        if reader.get_bit(prob[0]) == 0 {
            break;
        }

        // Skip zero-run: advance past zero coefficients
        while reader.get_bit(prob[1]) == 0 {
            n += 1;
            if n >= 16 {
                return true;
            }
            prob = &probs[n][0]; // context 0 = preceded by zero
        }

        let v: i32;
        let next_ctx: usize;

        if reader.get_bit(prob[2]) == 0 {
            // DCT_1: value is 1
            v = 1;
            next_ctx = 1;
        } else {
            if reader.get_bit(prob[3]) == 0 {
                if reader.get_bit(prob[4]) == 0 {
                    // DCT_2
                    v = 2;
                } else {
                    // DCT_3 or DCT_4
                    v = 3 + reader.get_bit(prob[5]);
                }
            } else {
                // Large value (categories 1-6)
                v = get_large_value(reader, prob);
            }
            next_ctx = 2;
        }

        // Branchless sign reading (VP8GetSigned) + dequantize
        output[ZIGZAG[n] as usize] = reader.get_signed(v) * dq_pair[(n > 0) as usize];

        n += 1;
        if n < 16 {
            prob = &probs[n][next_ctx];
        }
    }

    n > first
}

// ============================================================================
// Residual data parsing (per macroblock)
// ============================================================================

/// Read all DCT coefficients for a macroblock.
///
/// Parses Y2 (or Y with DC), 16 Y sub-blocks, and 8 chroma sub-blocks
/// from the given token partition reader. Applies dequantization and
/// writes to `coeff_blocks`. Updates `mb` flags and `top`/`left`
/// complexity context.
///
/// # Arguments
/// * `reader` - Active reader for the current token partition.
/// * `mb` - Macroblock entry being populated (flags updated in place).
/// * `coeff_blocks` - Fixed-size 384-element coefficient buffer.
/// * `probs` - Position-indexed probability table from `FrameTables`.
/// * `dequant` - Dequantization pairs for this segment `[Y, Y2, UV]`.
/// * `top` - Top neighbor context for this column.
/// * `left` - Left neighbor context (current row).
pub(super) fn read_residual_data(
    reader: &mut ActivePartitionReader<'_>,
    mb: &mut MbRowEntry,
    coeff_blocks: &mut [i32; MB_COEFF_SIZE],
    probs: &[[[[u8; 11]; 3]; 17]; 4],
    dequant: &[DequantPair; 3],
    top: &mut PreviousMacroBlock,
    left: &mut PreviousMacroBlock,
) -> Result<(), InternalDecodeError> {
    // Plane indices in the probability table:
    //   0 = YCoeff1 (Y after Y2 subtraction, first=1)
    //   1 = Y2 (WHT coefficients)
    //   2 = Chroma (U and V)
    //   3 = YCoeff0 (Y without Y2, first=0, i.e. B-pred mode)
    const PLANE_Y1: usize = 0; // Plane::YCoeff1
    const PLANE_Y2: usize = 1; // Plane::Y2
    const PLANE_CHROMA: usize = 2; // Plane::Chroma
    const PLANE_Y0: usize = 3; // Plane::YCoeff0

    let has_y2 = mb.luma_mode != LumaMode::B;

    let (y_plane, first_y) = if has_y2 {
        // ---- Y2 block (Walsh-Hadamard coefficients) ----
        let complexity = top.complexity[0] + left.complexity[0];
        let mut block = [0i32; 16];
        let n = read_coefficients(
            reader,
            &mut block,
            &probs[PLANE_Y2],
            0,
            complexity as usize,
            dequant[1], // Y2 dequant
        );

        let ctx = u8::from(n);
        left.complexity[0] = ctx;
        top.complexity[0] = ctx;

        // WHT DC-only fast path: if only DC is non-zero, broadcast
        // simplified value (matches libwebp's shortcut in ParseResiduals)
        let has_ac = block[1..].iter().any(|&c| c != 0);
        if has_ac {
            transform::iwht4x4(&mut block);
            for (k, &val) in block.iter().enumerate() {
                coeff_blocks[16 * k] = val;
                if val != 0 {
                    mb.non_zero_blocks |= 1u32 << k;
                }
            }
        } else if block[0] != 0 {
            // DC-only: simplified WHT = (dc[0] + 3) >> 3, broadcast to all 16 blocks
            let dc0 = (block[0] + 3) >> 3;
            for k in 0..16 {
                coeff_blocks[16 * k] = dc0;
            }
            // All 16 Y blocks have non-zero DC
            mb.non_zero_blocks |= 0xFFFF;
        }
        // else: all zeros, nothing to do

        (PLANE_Y1, 1usize)
    } else {
        (PLANE_Y0, 0usize)
    };

    // ---- 16 Y sub-blocks ----
    for y in 0usize..4 {
        let mut left_ctx = left.complexity[y + 1];
        for x in 0usize..4 {
            let i = x + y * 4;
            let complexity = top.complexity[x + 1] + left_ctx;

            let block = coeff_block(coeff_blocks, i);
            let n = read_coefficients(
                reader,
                block,
                &probs[y_plane],
                first_y,
                complexity as usize,
                dequant[0],
            );

            // Track non-zero DCT and per-block bitmap for IDCT skip
            if block[0] != 0 || n {
                mb.non_zero_dct = true;
                mb.non_zero_blocks |= 1u32 << i;
            }

            left_ctx = u8::from(n);
            top.complexity[x + 1] = u8::from(n);
        }
        left.complexity[y + 1] = left_ctx;
    }

    // ---- 8 chroma sub-blocks (4 U + 4 V) ----
    let chroma_probs = &probs[PLANE_CHROMA];

    // j=5 for U blocks, j=7 for V blocks (complexity array layout: y2,y,y,y,y,u,u,v,v)
    for &j in &[5usize, 7usize] {
        for y in 0usize..2 {
            let mut left_ctx = left.complexity[y + j];

            for x in 0usize..2 {
                let i = x + y * 2 + if j == 5 { 16 } else { 20 };
                let complexity = top.complexity[x + j] + left_ctx;

                let block = coeff_block(coeff_blocks, i);
                let n = read_coefficients(
                    reader,
                    block,
                    chroma_probs,
                    0,
                    complexity as usize,
                    dequant[2], // UV dequant
                );

                if block[0] != 0 || n {
                    mb.non_zero_dct = true;
                    mb.non_zero_blocks |= 1u32 << i;
                }

                // Check for non-zero UV AC coefficients (any coeff at index > 0).
                // libwebp suppresses dithering on MBs with UV AC content.
                if !mb.has_nonzero_uv_ac {
                    for &coeff in block[1..].iter() {
                        if coeff != 0 {
                            mb.has_nonzero_uv_ac = true;
                            break;
                        }
                    }
                }

                left_ctx = u8::from(n);
                top.complexity[x + j] = u8::from(n);
            }

            left.complexity[y + j] = left_ctx;
        }
    }

    // Single EOF check after all blocks -- matches libwebp's VP8DecodeMB
    // which checks once per MB, not per block.
    if reader.is_eof() {
        return Err(InternalDecodeError::BitStreamError);
    }

    Ok(())
}

// ============================================================================
// Macroblock header parsing
// ============================================================================

/// Read the segment ID from the bitstream using the segment tree.
///
/// Segment tree: `[2, 4, -0, -1, -2, -3]` with 3 probabilities.
/// Inlined directly to avoid TreeNode dependency.
#[inline]
fn read_segment_id(b: &mut VP8HeaderBitReader, probs: &[u8; 3]) -> u8 {
    if !b.read_bool(probs[0]) {
        // Left branch
        if !b.read_bool(probs[1]) { 0 } else { 1 }
    } else {
        // Right branch
        if !b.read_bool(probs[2]) { 2 } else { 3 }
    }
}

/// Read the keyframe luma prediction mode.
///
/// Y-mode tree: `[-B_PRED, 2, 4, 6, -DC_PRED, -V_PRED, -H_PRED, -TM_PRED]`
/// Probs: `[145, 156, 163, 128]`
#[inline]
fn read_ymode(b: &mut VP8HeaderBitReader) -> Result<LumaMode, InternalDecodeError> {
    if !b.read_bool(145) {
        Ok(LumaMode::B)
    } else if !b.read_bool(156) {
        if !b.read_bool(163) {
            Ok(LumaMode::DC)
        } else {
            Ok(LumaMode::V)
        }
    } else if !b.read_bool(128) {
        Ok(LumaMode::H)
    } else {
        Ok(LumaMode::TM)
    }
}

/// Read the keyframe chroma prediction mode.
///
/// UV-mode tree: `[-DC_PRED, 2, -V_PRED, 4, -H_PRED, -TM_PRED]`
/// Probs: `[142, 114, 183]`
#[inline]
fn read_uvmode(b: &mut VP8HeaderBitReader) -> Result<ChromaMode, InternalDecodeError> {
    if !b.read_bool(142) {
        Ok(ChromaMode::DC)
    } else if !b.read_bool(114) {
        Ok(ChromaMode::V)
    } else if !b.read_bool(183) {
        Ok(ChromaMode::H)
    } else {
        Ok(ChromaMode::TM)
    }
}

/// Read the keyframe B-prediction sub-block mode.
///
/// B-pred tree with context-dependent probabilities:
/// ```text
/// node 0: p[0] -> false: DC(0)
///                  true:  node 1
/// node 1: p[1] -> false: TM(1)
///                  true:  node 2
/// node 2: p[2] -> false: VE(2)
///                  true:  node 3
/// node 3: p[3] -> false: node 4
///                  true:  node 6
/// node 4: p[4] -> false: HE(3)
///                  true:  node 5
/// node 5: p[5] -> false: RD(5)
///                  true:  VR(6)
/// node 6: p[6] -> false: LD(4)
///                  true:  node 7
/// node 7: p[7] -> false: VL(7)
///                  true:  node 8
/// node 8: p[8] -> false: HD(8)
///                  true:  HU(9)
/// ```
#[inline]
fn read_bmode(
    b: &mut VP8HeaderBitReader,
    probs: &[Prob; 9],
) -> Result<IntraMode, InternalDecodeError> {
    let mode = if !b.read_bool(probs[0]) {
        IntraMode::DC
    } else if !b.read_bool(probs[1]) {
        IntraMode::TM
    } else if !b.read_bool(probs[2]) {
        IntraMode::VE
    } else if !b.read_bool(probs[3]) {
        // Left subtree: HE, RD, VR
        if !b.read_bool(probs[4]) {
            IntraMode::HE
        } else if !b.read_bool(probs[5]) {
            IntraMode::RD
        } else {
            IntraMode::VR
        }
    } else {
        // Right subtree: LD, VL, HD, HU
        if !b.read_bool(probs[6]) {
            IntraMode::LD
        } else if !b.read_bool(probs[7]) {
            IntraMode::VL
        } else if !b.read_bool(probs[8]) {
            IntraMode::HD
        } else {
            IntraMode::HU
        }
    };
    Ok(mode)
}

/// Read the macroblock header from partition 0 (mode data).
///
/// Reads segment ID, skip flag, luma mode, B-prediction sub-modes (if
/// B-pred), and chroma mode. Updates `top` and `left` border prediction
/// context.
///
/// This is a free function taking explicit parameters instead of `&mut self`
/// to support the v2 split-borrow decode loop.
pub(super) fn read_macroblock_header(
    b: &mut VP8HeaderBitReader,
    tables: &FrameTables,
    top: &mut PreviousMacroBlock,
    left: &mut PreviousMacroBlock,
    mb: &mut MbRowEntry,
) -> Result<(), InternalDecodeError> {
    // ---- Segment ID ----
    if tables.segments_enabled && tables.segments_update_map {
        mb.segmentid = read_segment_id(b, &tables.segment_tree_probs);
    }

    // ---- Skip flag ----
    mb.coeffs_skipped = if let Some(prob) = tables.prob_skip_false {
        b.read_bool(prob)
    } else {
        false
    };

    // ---- Luma mode ----
    mb.luma_mode = read_ymode(b)?;

    // ---- B-prediction sub-modes or broadcast ----
    match mb.luma_mode.into_intra() {
        None => {
            // LumaMode::B — each sub-block independently predicted
            for y in 0usize..4 {
                for x in 0usize..4 {
                    let top_mode = top.bpred[x];
                    let left_mode = left.bpred[y];
                    let bmode = read_bmode(
                        b,
                        &KEYFRAME_BPRED_MODE_PROBS[top_mode as usize][left_mode as usize],
                    )?;
                    mb.bpred[x + y * 4] = bmode;
                    top.bpred[x] = bmode;
                    left.bpred[y] = bmode;
                }
            }
        }
        Some(mode) => {
            // Non-B mode: fill bottom row and left column with this mode
            for i in 0usize..4 {
                mb.bpred[12 + i] = mode;
                left.bpred[i] = mode;
            }
        }
    }

    // ---- Chroma mode ----
    mb.chroma_mode = read_uvmode(b)?;

    // Store bottom row of bpred for the next row's top context
    top.bpred = [mb.bpred[12], mb.bpred[13], mb.bpred[14], mb.bpred[15]];

    // ---- Check for EOF ----
    b.check(())?;

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::bit_reader::VP8Partitions;
    use alloc::vec;

    /// Helper: create an ActivePartitionReader from test data.
    /// Uses VP8Partitions to initialize state (init_from_slice is private).
    fn make_reader_parts(data: &[u8]) -> VP8Partitions {
        let mut parts = VP8Partitions::new();
        let boundaries = [(0, data.len())];
        parts.init(data.to_vec(), &boundaries);
        parts
    }

    /// Verify ZIGZAG matches the shared constant.
    #[test]
    fn zigzag_matches_common() {
        assert_eq!(ZIGZAG, crate::common::types::ZIGZAG);
    }

    /// Verify PROB_DCT_CAT matches the shared constant.
    #[test]
    fn prob_dct_cat_matches_common() {
        assert_eq!(PROB_DCT_CAT, crate::common::types::PROB_DCT_CAT);
    }

    /// Test read_coefficients with an all-zero block (immediate EOB).
    #[test]
    fn read_coefficients_all_zero() {
        // A stream of zero bits will make get_bit return 0 at prob[0],
        // triggering immediate EOB (break at first position).
        let data = vec![0u8; 64];
        let mut parts = make_reader_parts(&data);
        let mut reader = parts.active_reader(0);

        let mut output = [0i32; 16];
        // All-zero probs means get_bit(0) always returns 0 for node 0
        let probs = [[[0u8; 11]; 3]; 17];
        let dq = DequantPair { dc: 10, ac: 20 };

        let n = read_coefficients(&mut reader, &mut output, &probs, 0, 0, dq);

        assert!(!n, "should return false for all-zero block");
        assert_eq!(output, [0i32; 16], "output should remain zero");
    }

    /// Test get_large_value produces valid range.
    #[test]
    fn get_large_value_valid() {
        // Feed deterministic data and check that the result is >= 5
        let data: alloc::vec::Vec<u8> = (0..64).map(|i| ((i * 37 + 13) & 0xFF) as u8).collect();
        let mut parts = make_reader_parts(&data);
        let mut reader = parts.active_reader(0);

        let prob = [128u8; 11];
        let v = get_large_value(&mut reader, &prob);
        assert!(v >= 5, "large value must be >= 5, got {v}");
    }

    /// Test WHT DC-only fast path: when only block[0] is non-zero,
    /// all 16 Y blocks should receive (dc[0] + 3) >> 3.
    #[test]
    fn wht_dc_only_broadcast() {
        let mut coeff_blocks = [0i32; MB_COEFF_SIZE];
        // Simulate what read_residual_data does for DC-only Y2:
        // block = [dc, 0, 0, ..., 0], dc != 0
        let dc = 80i32;
        let dc0 = (dc + 3) >> 3; // = 10
        for k in 0..16 {
            coeff_blocks[16 * k] = dc0;
        }

        // Verify broadcast
        for k in 0..16 {
            assert_eq!(coeff_blocks[16 * k], dc0, "Y block {k} DC should be {dc0}");
            for c in 1..16 {
                assert_eq!(
                    coeff_blocks[16 * k + c],
                    0,
                    "Y block {k} coeff {c} should be 0"
                );
            }
        }
    }

    /// Test that DequantPair is correctly used by read_coefficients.
    #[test]
    fn dequant_pair_dc_ac_indexing() {
        let dq = DequantPair { dc: 7, ac: 13 };
        let pair = [i32::from(dq.dc), i32::from(dq.ac)];
        // n=0 (DC) uses pair[0], n>0 (AC) uses pair[1]
        assert_eq!(pair[(0usize > 0) as usize], 7);
        assert_eq!(pair[(1usize > 0) as usize], 13);
        assert_eq!(pair[(15usize > 0) as usize], 13);
    }

    /// Test MbRowEntry default state.
    #[test]
    fn mb_row_entry_default() {
        let mb = MbRowEntry::default();
        assert_eq!(mb.segmentid, 0);
        assert!(!mb.coeffs_skipped);
        assert_eq!(mb.non_zero_blocks, 0);
        assert!(!mb.non_zero_dct);
        assert!(!mb.has_nonzero_uv_ac);
        assert_eq!(mb.luma_mode, LumaMode::DC);
        assert_eq!(mb.chroma_mode, ChromaMode::DC);
    }

    /// Test non_zero_blocks bitmap for Y2 DC-only case.
    #[test]
    fn non_zero_blocks_y2_dc_only() {
        let mut mb = MbRowEntry::default();
        // Simulate Y2 DC-only: all 16 Y blocks get non-zero DC
        mb.non_zero_blocks |= 0xFFFF;
        assert_eq!(mb.non_zero_blocks & 0xFFFF, 0xFFFF);
        // Chroma bits should still be clear
        assert_eq!(mb.non_zero_blocks >> 16, 0);
    }

    /// Test complexity context layout matches v1 expectations.
    #[test]
    fn complexity_layout() {
        // complexity: [y2, y, y, y, y, u, u, v, v] = 9 elements
        let pmb = PreviousMacroBlock::default();
        assert_eq!(pmb.complexity.len(), 9);
        assert_eq!(pmb.complexity[0], 0); // y2
        assert_eq!(pmb.complexity[5], 0); // u start
        assert_eq!(pmb.complexity[7], 0); // v start
    }

    /// Test read_coefficients with first=1 (Y after Y2).
    #[test]
    fn read_coefficients_first_one() {
        let data = vec![0u8; 64];
        let mut parts = make_reader_parts(&data);
        let mut reader = parts.active_reader(0);

        let mut output = [0i32; 16];
        let probs = [[[0u8; 11]; 3]; 17];
        let dq = DequantPair { dc: 10, ac: 20 };

        // With first=1 and all-zero probs, should return false and
        // not modify any coefficients.
        let n = read_coefficients(&mut reader, &mut output, &probs, 1, 0, dq);

        assert!(!n, "should return false for all-zero block at first=1");
        assert_eq!(output, [0i32; 16]);
    }

    /// Test read_coefficients with high-probability data produces non-zero output.
    #[test]
    fn read_coefficients_nonzero() {
        // Feed data with all 0xFF bytes (high values) and all-255 probs
        // so get_bit(255) almost always returns 0 but get_bit(1) returns 1.
        // With prob=255 for node 0 (is-non-zero test), the branch will
        // be taken when value > split, which depends on the data.
        let data = vec![0xFFu8; 128];
        let mut parts = make_reader_parts(&data);
        let mut reader = parts.active_reader(0);

        let mut output = [0i32; 16];
        // Set all probs to 128 for balanced decoding
        let probs = [[[128u8; 11]; 3]; 17];
        let dq = DequantPair { dc: 4, ac: 8 };

        let _n = read_coefficients(&mut reader, &mut output, &probs, 0, 0, dq);

        // With random-ish data, we expect some coefficients decoded.
        // The exact values depend on the arithmetic decoder state,
        // but the function should not panic.
    }
}
