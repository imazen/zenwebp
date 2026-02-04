//! Token statistics for adaptive probabilities.
//!
//! Ported from libwebp src/enc/cost_enc.h and src/enc/frame_enc.c
//! This enables two-pass encoding with optimal probability updates.

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use super::vp8_bit_cost;
use super::super::tables::{MAX_VARIABLE_LEVEL, VP8_ENC_BANDS, VP8_ENTROPY_COST};

/// Number of coefficient types (DCT types: 0=i16-DC, 1=i16-AC, 2=chroma, 3=i4)
pub const NUM_TYPES: usize = 4;
/// Number of bands for coefficient encoding
pub const NUM_BANDS: usize = 8;
/// Number of contexts (0=zero, 1=one, 2=more)
pub const NUM_CTX: usize = 3;
/// Number of probabilities per context node
pub const NUM_PROBAS: usize = 11;

/// Token statistics for computing optimal probabilities.
/// Format: upper 16 bits = total count, lower 16 bits = count of 1s.
/// Ported from libwebp's proba_t type.
#[derive(Clone, Default)]
pub struct ProbaStats {
    /// Statistics array: \[type\]\[band\]\[context\]\[proba_node\]
    pub stats: [[[[u32; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
}

impl ProbaStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all statistics to zero
    pub fn reset(&mut self) {
        for t in self.stats.iter_mut() {
            for b in t.iter_mut() {
                for c in b.iter_mut() {
                    for p in c.iter_mut() {
                        *p = 0;
                    }
                }
            }
        }
    }

    /// Record a bit value for a specific probability node.
    /// Ported from libwebp's VP8RecordStats.
    #[inline]
    pub fn record(&mut self, t: usize, b: usize, c: usize, p: usize, bit: bool) {
        let stats = &mut self.stats[t][b][c][p];
        // Check for overflow (at 0xfffe0000)
        if *stats >= 0xfffe_0000 {
            // Divide stats by 2 to prevent overflow
            *stats = ((*stats + 1) >> 1) & 0x7fff_7fff;
        }
        // Record: lower 16 bits = count of 1s, upper 16 bits = total count
        *stats += 0x0001_0000 + if bit { 1 } else { 0 };
    }

    /// Get the optimal probability for a node based on accumulated statistics.
    /// Returns 255 - (nb * 255 / total) where nb = count of 1s.
    pub fn calc_proba(&self, t: usize, b: usize, c: usize, p: usize) -> u8 {
        let stats = self.stats[t][b][c][p];
        let nb = stats & 0xffff; // count of 1s
        let total = stats >> 16; // total count
        if total == 0 {
            return 255;
        }
        let proba = 255 - (nb * 255 / total);
        proba as u8
    }

    /// Calculate the cost of updating vs keeping old probability.
    /// Returns (should_update, new_prob, bit_savings).
    pub fn should_update(
        &self,
        t: usize,
        b: usize,
        c: usize,
        p: usize,
        old_proba: u8,
        update_proba: u8,
    ) -> (bool, u8, i32) {
        let stats = self.stats[t][b][c][p];
        let nb = (stats & 0xffff) as i32; // count of 1s
        let total = (stats >> 16) as i32; // total count

        if total == 0 {
            return (false, old_proba, 0);
        }

        let new_p = self.calc_proba(t, b, c, p);

        // Cost with old probability
        let old_cost = branch_cost(nb, total, old_proba) + vp8_bit_cost(false, update_proba) as i32;

        // Cost with new probability (includes signaling cost)
        let new_cost =
            branch_cost(nb, total, new_p) + vp8_bit_cost(true, update_proba) as i32 + 8 * 256; // 8 bits to signal new probability value

        let savings = old_cost - new_cost;
        (savings > 0, new_p, savings)
    }
}

/// Calculate the branch cost for a given count distribution and probability.
/// Cost = nb * cost(1|p) + (total - nb) * cost(0|p)
#[inline]
fn branch_cost(nb: i32, total: i32, proba: u8) -> i32 {
    let cost_1 = VP8_ENTROPY_COST[255 - proba as usize] as i32;
    let cost_0 = VP8_ENTROPY_COST[proba as usize] as i32;
    nb * cost_1 + (total - nb) * cost_0
}

/// Token type for coefficient encoding
/// Values must match libwebp's type indices:
/// - TYPE_I16_AC = 0 (Y1 AC coefficients, first=1)
/// - TYPE_I16_DC = 1 (Y2 DC coefficients, first=0)
/// - TYPE_CHROMA_A = 2
/// - TYPE_I4_AC = 3
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenType {
    /// I16 AC coefficients (Y1 blocks with DC=0) - must be 0 to match libwebp
    I16AC = 0,
    /// I16 DC coefficients (Y2/WHT) - must be 1 to match libwebp
    I16DC = 1,
    /// Chroma (UV) coefficients
    Chroma = 2,
    /// I4 coefficients
    I4 = 3,
}

/// Record coefficient tokens for probability statistics.
/// Structure matches the encoder's skip_eob pattern exactly:
/// - For each coefficient, check node 0 (EOB) if skip_eob=false
/// - Check node 1 (zero/non-zero) for all coefficients
/// - Set skip_eob=true after zeros (can skip EOB check after a zero)
/// - At end, record EOB at node 0 if there are trailing positions
///
/// # Arguments
/// * `coeffs` - Quantized coefficients in zigzag order (16 values)
/// * `token_type` - Type of coefficients (I16DC, I16AC, Chroma, I4)
/// * `first` - First coefficient to process (0 or 1)
/// * `ctx` - Initial context (0, 1, or 2)
/// * `stats` - Statistics accumulator
pub fn record_coeffs(
    coeffs: &[i32],
    token_type: TokenType,
    first: usize,
    ctx: usize,
    stats: &mut ProbaStats,
) {
    let t = token_type as usize;
    let mut n = first;
    let mut context = ctx;

    // Find last non-zero coefficient (end_of_block_index - 1)
    let last = coeffs
        .iter()
        .rposition(|&c| c != 0)
        .map(|i| i as i32)
        .unwrap_or(-1);

    let end_of_block = if last >= 0 { (last + 1) as usize } else { 0 };

    // If no non-zero coefficients, record EOB immediately
    if end_of_block <= first {
        let band = VP8_ENC_BANDS[first] as usize;
        stats.record(t, band, context, 0, false); // EOB at node 0
        return;
    }

    // Track skip_eob like the encoder does
    let mut skip_eob = false;

    // Process coefficients up to end_of_block (last non-zero + 1)
    while n < end_of_block {
        let band = VP8_ENC_BANDS[n] as usize;
        let v = coeffs[n].unsigned_abs();
        n += 1;

        // Record at node 0 (EOB check) if not skipping
        if !skip_eob {
            stats.record(t, band, context, 0, true); // not EOB
        }

        if v == 0 {
            // Zero coefficient: record 0 at node 1, set skip_eob for next
            stats.record(t, band, context, 1, false);
            skip_eob = true;
            context = 0;
            continue;
        }

        // Non-zero coefficient: record 1 at node 1
        stats.record(t, band, context, 1, true);

        // Record value magnitude bits
        if v == 1 {
            stats.record(t, band, context, 2, false);
            context = 1;
        } else {
            stats.record(t, band, context, 2, true);

            // Clamp v to MAX_VARIABLE_LEVEL for statistics
            let v = v.min(MAX_VARIABLE_LEVEL as u32);

            if v <= 4 {
                stats.record(t, band, context, 3, false);
                if v == 2 {
                    stats.record(t, band, context, 4, false);
                } else {
                    stats.record(t, band, context, 4, true);
                    stats.record(t, band, context, 5, v == 4);
                }
            } else if v <= 10 {
                stats.record(t, band, context, 3, true);
                stats.record(t, band, context, 6, false);
                stats.record(t, band, context, 7, v > 6);
            } else {
                stats.record(t, band, context, 3, true);
                stats.record(t, band, context, 6, true);

                if v < 3 + (8 << 2) {
                    stats.record(t, band, context, 8, false);
                    stats.record(t, band, context, 9, v >= 3 + (8 << 1));
                } else {
                    stats.record(t, band, context, 8, true);
                    stats.record(t, band, context, 10, v >= 3 + (8 << 3));
                }
            }

            context = 2;
        }

        // After non-zero, the encoder does NOT reset skip_eob to false.
        // It leaves it unchanged. So if skip_eob was true (from a previous zero),
        // it stays true even after this non-zero. Do NOT reset it here!
    }

    // Record trailing EOB if we didn't reach position 16
    if n < 16 {
        let band = VP8_ENC_BANDS[n] as usize;
        stats.record(t, band, context, 0, false); // EOB
    }
}
