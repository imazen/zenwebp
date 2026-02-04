//! Level cost tables for accurate coefficient cost estimation.
//!
//! Ported from libwebp src/enc/cost_enc.c
//!
//! The key insight is that coefficient costs depend on the probability context.
//! VP8CalculateLevelCosts precomputes cost tables indexed by [type][band][ctx][level].
//! Then remapped_costs provides direct access by coefficient position: [type][n][ctx].

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use super::stats::{NUM_BANDS, NUM_CTX, NUM_PROBAS, NUM_TYPES};
use super::vp8_bit_cost;
use crate::common::types::TokenProbTables;

use super::super::tables::{MAX_LEVEL, MAX_VARIABLE_LEVEL, VP8_ENC_BANDS, VP8_LEVEL_CODES};

/// Type alias for level cost array: cost for each level 0..=MAX_VARIABLE_LEVEL
pub type LevelCostArray = [u16; MAX_VARIABLE_LEVEL + 1];

/// Level costs indexed by \[type\]\[band\]\[context\]
/// Each entry is an array of costs for levels 0..=MAX_VARIABLE_LEVEL
pub type LevelCostTables = [[[LevelCostArray; NUM_CTX]; NUM_BANDS]; NUM_TYPES];

/// Remapped costs indexed by \[type\]\[position\]\[context\]
/// Maps coefficient position (0..16) directly to its band's level_cost.
/// This avoids the indirection through VP8_ENC_BANDS during cost calculation.
pub type RemappedCosts = [[usize; NUM_CTX]; 16];

/// Calculate the variable-length cost for encoding a level >= 1.
/// Uses the VP8_LEVEL_CODES table to determine which probability nodes to use.
/// Ported from libwebp's VariableLevelCost.
fn variable_level_cost(level: usize, probas: &[u8; NUM_PROBAS]) -> u16 {
    if level == 0 {
        return 0;
    }
    let idx = level.min(MAX_VARIABLE_LEVEL) - 1;
    let pattern = VP8_LEVEL_CODES[idx][0];
    let bits = VP8_LEVEL_CODES[idx][1];

    let mut cost = 0u16;
    let mut p = pattern;
    let mut b = bits;
    let mut i = 2; // Start at proba index 2

    while p != 0 {
        if (p & 1) != 0 {
            cost += vp8_bit_cost((b & 1) != 0, probas[i]);
        }
        b >>= 1;
        p >>= 1;
        i += 1;
    }
    cost
}

/// Level cost tables holder with precomputed costs and remapping.
/// Ported from libwebp's VP8EncProba (level_cost and remapped_costs fields).
#[derive(Clone)]
pub struct LevelCosts {
    /// Level costs indexed by \[type\]\[band\]\[ctx\]\[level\]
    pub level_cost: LevelCostTables,
    /// Remapped indices: [type][position] -> band index for each type
    /// Usage: level_cost[type][remapped[type][n]][ctx][level]
    remapped: [RemappedCosts; NUM_TYPES],
    /// EOB (end-of-block) costs indexed by [type][band][ctx]
    /// This is the cost of signaling "no more coefficients"
    eob_cost: [[[u16; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
    /// Init (has-coefficients) costs indexed by [type][band][ctx]
    /// This is the cost of signaling "block has coefficients"
    /// Used for initializing trellis at ctx0=0
    init_cost: [[[u16; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
    /// Whether the tables are dirty and need recalculation
    dirty: bool,
}

impl Default for LevelCosts {
    fn default() -> Self {
        Self::new()
    }
}

impl LevelCosts {
    /// Create new level cost tables
    pub fn new() -> Self {
        Self {
            level_cost: [[[[0u16; MAX_VARIABLE_LEVEL + 1]; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
            remapped: [[[0usize; NUM_CTX]; 16]; NUM_TYPES],
            eob_cost: [[[0u16; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
            init_cost: [[[0u16; NUM_CTX]; NUM_BANDS]; NUM_TYPES],
            dirty: true,
        }
    }

    /// Mark tables as dirty (need recalculation)
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Check if tables need recalculation
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Calculate level costs from probability tables.
    /// Ported from libwebp's VP8CalculateLevelCosts.
    #[allow(clippy::needless_range_loop)] // indices used for multiple arrays
    pub fn calculate(&mut self, probs: &TokenProbTables) {
        if !self.dirty {
            return;
        }

        for ctype in 0..NUM_TYPES {
            for band in 0..NUM_BANDS {
                for ctx in 0..NUM_CTX {
                    let p = &probs[ctype][band][ctx];

                    // cost0 is the cost of signaling "no more coefficients" at context > 0
                    // For ctx == 0, this cost is handled separately
                    let cost0 = if ctx > 0 { vp8_bit_cost(true, p[0]) } else { 0 };

                    // cost_base is cost of signaling "coefficient present" + cost0
                    let cost_base = vp8_bit_cost(true, p[1]) + cost0;

                    // Level 0: just signal "no coefficient"
                    self.level_cost[ctype][band][ctx][0] = vp8_bit_cost(false, p[1]) + cost0;

                    // Levels 1..=MAX_VARIABLE_LEVEL
                    for v in 1..=MAX_VARIABLE_LEVEL {
                        self.level_cost[ctype][band][ctx][v] =
                            cost_base + variable_level_cost(v, p);
                    }

                    // EOB cost: signaling "no more coefficients" after this position
                    // This is the cost of taking the EOB branch in the coefficient tree
                    self.eob_cost[ctype][band][ctx] = vp8_bit_cost(false, p[0]);

                    // Init cost: signaling "block has coefficients" at this position
                    // Used for initializing trellis at ctx0=0
                    self.init_cost[ctype][band][ctx] = vp8_bit_cost(true, p[0]);
                }
            }

            // Build remapped indices for direct position-based lookup
            for n in 0..16 {
                let band = VP8_ENC_BANDS[n] as usize;
                for ctx in 0..NUM_CTX {
                    self.remapped[ctype][n][ctx] = band;
                }
            }
        }

        self.dirty = false;
    }

    /// Get level cost for a specific coefficient position.
    /// Combines fixed cost from VP8_LEVEL_FIXED_COSTS and variable cost from tables.
    #[inline]
    pub fn get_level_cost(&self, ctype: usize, position: usize, ctx: usize, level: usize) -> u32 {
        use super::super::tables::VP8_LEVEL_FIXED_COSTS;
        let fixed = VP8_LEVEL_FIXED_COSTS[level.min(MAX_LEVEL)] as u32;
        let band = self.remapped[ctype][position][ctx];
        let variable = self.level_cost[ctype][band][ctx][level.min(MAX_VARIABLE_LEVEL)] as u32;
        fixed + variable
    }

    /// Get the cost table for a specific type, position, and context.
    #[inline]
    pub fn get_cost_table(&self, ctype: usize, position: usize, ctx: usize) -> &LevelCostArray {
        let band = self.remapped[ctype][position][ctx];
        &self.level_cost[ctype][band][ctx]
    }

    /// Get the EOB (end-of-block) cost for terminating after position n.
    /// This is the cost of signaling "no more coefficients" at position n+1.
    /// The context should be based on the level at position n (ctx = min(level, 2)).
    #[inline]
    pub fn get_eob_cost(&self, ctype: usize, position: usize, ctx: usize) -> u16 {
        // EOB is signaled at position n+1, so use band for n+1
        let next_pos = (position + 1).min(15);
        let band = VP8_ENC_BANDS[next_pos] as usize;
        self.eob_cost[ctype][band][ctx]
    }

    /// Get the EOB cost for signaling "no coefficients at all" at position first.
    /// Used for skip (all-zero block) calculation.
    #[inline]
    pub fn get_skip_eob_cost(&self, ctype: usize, first: usize, ctx: usize) -> u16 {
        let band = VP8_ENC_BANDS[first] as usize;
        self.eob_cost[ctype][band][ctx]
    }

    /// Get the init cost for signaling "block has coefficients" at position first.
    /// Used for initializing trellis at ctx0=0.
    #[inline]
    pub fn get_init_cost(&self, ctype: usize, first: usize, ctx: usize) -> u16 {
        let band = VP8_ENC_BANDS[first] as usize;
        self.init_cost[ctype][band][ctx]
    }
}
