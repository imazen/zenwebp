//! Token statistics for adaptive probabilities.
//!
//! Ported from libwebp src/enc/cost_enc.h and src/enc/frame_enc.c
//! This enables two-pass encoding with optimal probability updates.

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use super::super::tables::VP8_ENTROPY_COST;
use super::vp8_bit_cost;

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
