//! Coefficient reading and dequantization for VP8 decoder.
//!
//! Contains `read_coefficients`, `get_large_value`, and `read_residual_data`.

use super::*;

/// Decode large coefficient values (categories 3-6).
///
/// Separated from `read_coefficients` to keep the hot path small,
/// matching C's `GetLargeValue`/`GetCoeffsFast` split. Both functions
/// are `#[inline(never)]` so the CPU branch predictor tracks each
/// through a single set of addresses, reducing BTB aliasing.
#[inline(never)]
fn get_large_value(
    reader: &mut ActivePartitionReader<'_>,
    prob: &[TreeNode; NUM_DCT_TOKENS - 1],
) -> i32 {
    if reader.get_bit(prob[6].prob) == 0 {
        if reader.get_bit(prob[7].prob) == 0 {
            5 + reader.get_bit(159)
        } else {
            7 + 2 * reader.get_bit(165) + reader.get_bit(145)
        }
    } else {
        let bit1 = reader.get_bit(prob[8].prob);
        let bit0 = reader.get_bit(prob[9 + bit1 as usize].prob);
        let cat = (2 * bit1 + bit0) as usize;

        let cat_probs = &PROB_DCT_CAT[2 + cat];
        let cat_len = [3usize, 4, 5, 11][cat];
        let mut extra = 0i32;
        for i in 0..cat_len {
            extra = extra + extra + reader.get_bit(cat_probs[i]);
        }
        3 + (8 << cat) + extra
    }
}

/// Read and dequantize DCT coefficients for one 4x4 block.
///
/// Returns true if any coefficient beyond `first` was non-zero.
///
/// NOT inlined so the CPU branch predictor tracks all coefficient
/// decoding through a single set of branch addresses. When inlined 25x
/// (once per block in a macroblock), BTB aliasing causes ~2.3M extra
/// mispredicts per decode (measured via cachegrind). The function call
/// overhead is compensated by the mispredict savings.
///
/// Does NOT check for EOF — caller must check `reader.is_eof()` after
/// processing all blocks in the macroblock (matching libwebp's approach
/// where VP8DecodeMB checks EOF once per MB, not per block).
#[inline(never)]
fn read_coefficients(
    reader: &mut ActivePartitionReader<'_>,
    output: &mut [i32; 16],
    probs: &[[[TreeNode; NUM_DCT_TOKENS - 1]; 3]; 17],
    first: usize,
    complexity: usize,
    dq: [i32; 2], // [dc_quant, ac_quant] — indexed by (n > 0) like libwebp
) -> bool {
    debug_assert!(complexity <= 2);

    let mut n = first;
    let mut prob = &probs[n][complexity];

    while n < 16 {
        if reader.get_bit(prob[0].prob) == 0 {
            break;
        }

        while reader.get_bit(prob[1].prob) == 0 {
            n += 1;
            if n >= 16 {
                return true;
            }
            prob = &probs[n][0];
        }

        let v: i32;
        let next_ctx: usize;

        if reader.get_bit(prob[2].prob) == 0 {
            v = 1;
            next_ctx = 1;
        } else {
            if reader.get_bit(prob[3].prob) == 0 {
                if reader.get_bit(prob[4].prob) == 0 {
                    v = 2;
                } else {
                    v = 3 + reader.get_bit(prob[5].prob);
                }
            } else {
                v = get_large_value(reader, prob);
            }
            next_ctx = 2;
        }

        // Branchless sign reading (VP8GetSigned) + dequantize with array lookup
        output[ZIGZAG[n] as usize] = reader.get_signed(v) * dq[(n > 0) as usize];

        n += 1;
        if n < 16 {
            prob = &probs[n][next_ctx];
        }
    }

    n > first
}

impl<'a> Vp8Decoder<'a> {
    /// Read DCT coefficients using libwebp's inline tree structure.
    /// This mirrors GetCoeffsFast from libwebp for maximum performance.
    /// Writes to self.coeff_blocks at the given block index.
    pub(super) fn read_residual_data(
        &mut self,
        mb: &mut MacroBlock,
        mbx: usize,
        p: usize,
        _simd_token: SimdTokenType, // Kept for API consistency, IDCT deferred to prediction
    ) -> Result<(), InternalDecodeError> {
        // Uses self.coeff_blocks which is maintained as zeros between calls.
        // After each IDCT, the block is left with transformed data for intra_predict to use.
        // intra_predict_* is responsible for clearing blocks after use.
        let sindex = mb.segmentid as usize;

        // Extract quantizers as 2-element arrays [dc, ac] — indexed by (n > 0) like libwebp
        let y2_dq = [
            i32::from(self.segment[sindex].y2dc),
            i32::from(self.segment[sindex].y2ac),
        ];
        let y_dq = [
            i32::from(self.segment[sindex].ydc),
            i32::from(self.segment[sindex].yac),
        ];
        let uv_dq = [
            i32::from(self.segment[sindex].uvdc),
            i32::from(self.segment[sindex].uvac),
        ];

        // Split borrows: create active reader from partition fields directly
        // This avoids the per-block PartitionReader creation overhead (~7.7M instructions/decode)
        let (start, len) = self.partitions.boundaries[p];
        let partition_data = &self.partitions.data[start..start + len];
        let partition_state = &mut self.partitions.states[p];
        let mut reader = ActivePartitionReader::new(partition_data, partition_state);

        // Get references to other fields we need
        let probs = &*self.token_probs_by_pos;
        let coeff_blocks = &mut self.coeff_blocks;
        let top = &mut self.top[mbx];
        let left = &mut self.left;

        let mut plane = if mb.luma_mode == LumaMode::B {
            Plane::YCoeff0
        } else {
            Plane::Y2
        };

        if plane == Plane::Y2 {
            let complexity = top.complexity[0] + left.complexity[0];
            let mut block = [0i32; 16];
            let n = read_coefficients(
                &mut reader,
                &mut block,
                &probs[Plane::Y2 as usize],
                0, // first
                complexity as usize,
                y2_dq,
            );

            left.complexity[0] = if n { 1 } else { 0 };
            top.complexity[0] = if n { 1 } else { 0 };

            // Optimized WHT: if only DC is non-zero, broadcast simplified value
            // (matches libwebp's shortcut in ParseResiduals)
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

            plane = Plane::YCoeff1;
        }

        let first_y = if plane == Plane::YCoeff1 { 1 } else { 0 };

        for y in 0usize..4 {
            let mut left_ctx = left.complexity[y + 1];
            for x in 0usize..4 {
                let i = x + y * 4;
                let complexity = top.complexity[x + 1] + left_ctx;

                let block = coeff_block(coeff_blocks, i);
                let n = read_coefficients(
                    &mut reader,
                    block,
                    &probs[plane as usize],
                    first_y,
                    complexity as usize,
                    y_dq,
                );

                // Track non-zero DCT but defer IDCT to fused function during prediction.
                // Also set per-block bitmap for IDCT skip optimization.
                if block[0] != 0 || n {
                    mb.non_zero_dct = true;
                    mb.non_zero_blocks |= 1u32 << i;
                }

                left_ctx = if n { 1 } else { 0 };
                top.complexity[x + 1] = if n { 1 } else { 0 };
            }

            left.complexity[y + 1] = left_ctx;
        }

        // Chroma
        let chroma_probs = &probs[Plane::Chroma as usize];

        for &j in &[5usize, 7usize] {
            for y in 0usize..2 {
                let mut left_ctx = left.complexity[y + j];

                for x in 0usize..2 {
                    let i = x + y * 2 + if j == 5 { 16 } else { 20 };
                    let complexity = top.complexity[x + j] + left_ctx;

                    let block = coeff_block(coeff_blocks, i);
                    let n = read_coefficients(
                        &mut reader,
                        block,
                        chroma_probs,
                        0, // first
                        complexity as usize,
                        uv_dq,
                    );

                    // Track non-zero DCT but defer IDCT to fused function during prediction.
                    // Also set per-block bitmap.
                    if block[0] != 0 || n {
                        mb.non_zero_dct = true;
                        mb.non_zero_blocks |= 1u32 << i;
                    }

                    // Check for non-zero UV AC coefficients (any coeff at index > 0).
                    // libwebp suppresses dithering on MBs with UV AC content
                    // (non_zero_uv & 0xaaaa test in ParseResiduals).
                    // n is true when at least one coefficient was decoded, but that
                    // includes DC-only. Check the block directly for AC content.
                    if !mb.has_nonzero_uv_ac {
                        for &coeff in block[1..].iter() {
                            if coeff != 0 {
                                mb.has_nonzero_uv_ac = true;
                                break;
                            }
                        }
                    }

                    left_ctx = if n { 1 } else { 0 };
                    top.complexity[x + j] = if n { 1 } else { 0 };
                }

                left.complexity[y + j] = left_ctx;
            }
        }

        // Single EOF check after all blocks in the MB — matches libwebp's
        // VP8DecodeMB which checks once per MB, not per block.
        if reader.is_eof() {
            return Err(InternalDecodeError::BitStreamError);
        }

        Ok(())
    }
}
