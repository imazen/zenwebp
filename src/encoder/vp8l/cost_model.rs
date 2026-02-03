//! Cost model and optimal parsing for VP8L backward references.
//!
//! Implements a Zopfli-like cost-based optimizer (TraceBackwards) that
//! improves greedy LZ77 by recomputing optimal literal/copy decisions
//! using dynamic programming with per-symbol bit costs.

use alloc::vec;
use alloc::vec::Vec;

use super::backward_refs::distance_to_plane_code;
use super::color_cache::ColorCache;
use super::histogram::{distance_code_to_prefix, length_to_code, Histogram};
use super::types::{
    argb_alpha, argb_blue, argb_green, argb_red, BackwardRefs, PixOrCopy, MAX_LENGTH,
    NUM_LENGTH_CODES, NUM_LITERAL_CODES,
};

/// Fixed-point precision for entropy (matches libwebp LOG_2_PRECISION_BITS).
const LOG_2_PRECISION_BITS: u32 = 23;

/// Compute log2(v) in fixed-point (matching libwebp's VP8LFastLog2).
/// Returns 0 for v == 0 or v == 1.
fn fast_log2(v: u32) -> u32 {
    if v <= 1 {
        return 0;
    }
    let v_f64 = v as f64;
    (v_f64.log2() * (1u64 << LOG_2_PRECISION_BITS) as f64) as u32
}

/// Rounding division matching libwebp's DivRound.
#[inline]
fn div_round(a: i64, b: i64) -> i64 {
    if (a < 0) == (b < 0) {
        (a + b / 2) / b
    } else {
        (a - b / 2) / b
    }
}

/// Convert histogram population counts to per-symbol bit cost estimates.
///
/// Each output[i] = VP8LFastLog2(sum) - VP8LFastLog2(counts[i]).
/// Matches libwebp's ConvertPopulationCountTableToBitEstimates.
fn counts_to_bit_estimates(counts: &[u32]) -> Vec<u32> {
    let total: u32 = counts.iter().sum();
    let n = counts.len();

    let nonzeros = counts.iter().filter(|&&c| c > 0).count();
    if nonzeros <= 1 {
        return vec![0u32; n];
    }

    let logsum = fast_log2(total);
    let mut output = vec![0u32; n];
    for (i, &count) in counts.iter().enumerate() {
        output[i] = logsum.saturating_sub(fast_log2(count));
    }
    output
}

/// Per-symbol bit cost model built from histogram statistics.
pub struct CostModel {
    /// Green/literal/length costs (256 + 24 + cache).
    literal: Vec<u32>,
    /// Red channel costs (256).
    red: Vec<u32>,
    /// Blue channel costs (256).
    blue: Vec<u32>,
    /// Alpha channel costs (256).
    alpha: Vec<u32>,
    /// Distance costs (40).
    distance: Vec<u32>,
}

impl CostModel {
    /// Build cost model from histogram of initial backward refs.
    pub fn build(xsize: usize, cache_bits: u8, refs: &BackwardRefs) -> Self {
        // Build histogram with plane-code-aware distances
        let histo = Histogram::from_refs_with_plane_codes(refs, cache_bits, xsize);

        Self {
            literal: counts_to_bit_estimates(&histo.literal),
            red: counts_to_bit_estimates(&histo.red),
            blue: counts_to_bit_estimates(&histo.blue),
            alpha: counts_to_bit_estimates(&histo.alpha),
            distance: counts_to_bit_estimates(&histo.distance),
        }
    }

    /// Cost of encoding a literal ARGB pixel.
    #[inline]
    fn literal_cost(&self, argb: u32) -> i64 {
        let a = argb_alpha(argb) as usize;
        let r = argb_red(argb) as usize;
        let g = argb_green(argb) as usize;
        let b = argb_blue(argb) as usize;
        self.alpha[a] as i64 + self.red[r] as i64 + self.literal[g] as i64 + self.blue[b] as i64
    }

    /// Cost of encoding a color cache index.
    #[inline]
    fn cache_cost(&self, idx: u16) -> i64 {
        let literal_idx = NUM_LITERAL_CODES + NUM_LENGTH_CODES + idx as usize;
        if literal_idx < self.literal.len() {
            self.literal[literal_idx] as i64
        } else {
            i64::MAX / 2
        }
    }

    /// Cost of encoding a copy length.
    #[inline]
    fn length_cost(&self, length: u32) -> i64 {
        let (code, _) = length_to_code(length as u16);
        let extra_bits = if code < 4 {
            0
        } else {
            (code / 2).saturating_sub(1) as u32
        };
        self.literal[NUM_LITERAL_CODES + code as usize] as i64
            + ((extra_bits as i64) << LOG_2_PRECISION_BITS)
    }

    /// Cost of encoding a distance code.
    #[inline]
    fn distance_cost(&self, dist_code: u32) -> i64 {
        let (code, _) = distance_code_to_prefix(dist_code);
        let extra_bits = if code < 4 {
            0
        } else {
            (code / 2).saturating_sub(1) as u32
        };
        self.distance[code as usize] as i64 + ((extra_bits as i64) << LOG_2_PRECISION_BITS)
    }
}

/// Compute cost-based optimal backward references using TraceBackwards.
///
/// Improves the greedy LZ77 result by using dynamic programming to find
/// the globally optimal sequence of literals and copies.
///
/// # DP semantics
/// - `costs[j]` = minimum cost to encode pixels `0..=j`
/// - `dist_array[j]` = step size of last operation ending at pixel j
///   - 1 = literal (single pixel)
///   - >= 2 = copy of that length ending at pixel j
///
/// For a copy of length L starting at position i:
/// - Covers pixels i, i+1, ..., i+L-1
/// - Target: `costs[i+L-1] = costs[i-1] + dist_cost + length_cost(L)`
/// - Step: `dist_array[i+L-1] = L`
///
/// # Arguments
/// - `argb`: transformed pixel data
/// - `xsize`, `ysize`: image dimensions
/// - `cache_bits`: color cache bits (0 = disabled)
/// - `hash_chain`: pre-built hash chain for match finding
/// - `initial_refs`: greedy LZ77 refs to derive cost model from
///
/// Returns improved backward refs (with raw distances, no 2D locality).
pub fn trace_backwards_optimize(
    argb: &[u32],
    xsize: usize,
    _ysize: usize,
    cache_bits: u8,
    hash_chain: &super::hash_chain::HashChain,
    initial_refs: &BackwardRefs,
) -> BackwardRefs {
    let pix_count = argb.len();
    let use_color_cache = cache_bits > 0;

    // Phase 1: Build cost model from initial greedy refs
    let cost_model = CostModel::build(xsize, cache_bits, initial_refs);

    // Pre-compute length costs indexed by copy length (1..=max).
    // Index 0 is unused; length_costs[L] = cost of encoding copy length L.
    let max_copy_len = pix_count.min(MAX_LENGTH);
    let mut length_costs = vec![0i64; max_copy_len + 1];
    for (len, cost) in length_costs.iter_mut().enumerate().skip(1) {
        *cost = cost_model.length_cost(len as u32);
    }

    // Phase 2: Forward DP pass
    let mut costs = vec![i64::MAX; pix_count];
    let mut dist_array = vec![0u16; pix_count];

    // Color cache for literal cost estimation (greedy approximation).
    // Updated for every position regardless of whether the optimal path
    // uses a literal there — same approximation libwebp uses.
    let mut cache = if use_color_cache {
        Some(ColorCache::new(cache_bits))
    } else {
        None
    };

    for i in 0..pix_count {
        // Cost of everything before position i
        let prev_cost = if i == 0 { 0i64 } else { costs[i - 1] };
        if i > 0 && prev_cost == i64::MAX {
            continue; // Unreachable position
        }

        // Option 1: Literal or cache hit at position i
        add_single_literal_cost(
            argb,
            &cost_model,
            &mut cache,
            i,
            prev_cost,
            &mut costs,
            &mut dist_array,
        );

        // Option 2: Copy starting at position i (minimum length 2)
        let (offset, match_len) = hash_chain.find_copy(i);
        if match_len >= 2 && offset > 0 {
            let plane_code = distance_to_plane_code(xsize, offset);
            let dist_cost = cost_model.distance_cost(plane_code);

            let max_len = match_len.min(pix_count - i).min(max_copy_len);
            for (copy_len, &len_cost) in length_costs[2..=max_len].iter().enumerate() {
                let copy_len = copy_len + 2;
                let target = i + copy_len - 1; // last pixel of copy
                let total_cost = prev_cost + dist_cost + len_cost;
                if total_cost < costs[target] {
                    costs[target] = total_cost;
                    dist_array[target] = copy_len as u16;
                }
            }
        }
    }

    // Phase 3: Backward trace — extract optimal path
    let mut path = Vec::new();
    let mut cur = pix_count as i64 - 1;
    while cur >= 0 {
        let step = dist_array[cur as usize] as i64;
        if step == 0 {
            // Safety: shouldn't happen if DP is correct, but handle gracefully
            path.push(1u16);
            cur -= 1;
        } else {
            path.push(step as u16);
            cur -= step;
        }
    }
    path.reverse();

    // Phase 4: Follow chosen path to build final backward refs
    let mut refs = BackwardRefs::with_capacity(path.len());
    let mut emit_cache = if use_color_cache {
        Some(ColorCache::new(cache_bits))
    } else {
        None
    };

    let mut i = 0;
    for &step in &path {
        let step = step as usize;
        if step == 1 {
            // Literal or cache hit
            let argb_val = argb[i];
            if let Some(ref mut c) = emit_cache {
                if let Some(idx) = c.lookup(argb_val) {
                    refs.push(PixOrCopy::cache_idx(idx));
                } else {
                    refs.push(PixOrCopy::literal(argb_val));
                    c.insert(argb_val);
                }
            } else {
                refs.push(PixOrCopy::literal(argb_val));
            }
            i += 1;
        } else {
            // Copy with offset from hash chain
            let offset = hash_chain.offset(i);
            refs.push(PixOrCopy::copy(step as u16, offset as u32));
            if let Some(ref mut c) = emit_cache {
                for k in 0..step {
                    c.insert(argb[i + k]);
                }
            }
            i += step;
        }
    }

    refs
}

/// Try adding a literal (or cache hit) at position idx with cost tracking.
/// Matches libwebp's AddSingleLiteralWithCostModel.
#[inline]
fn add_single_literal_cost(
    argb: &[u32],
    cost_model: &CostModel,
    cache: &mut Option<ColorCache>,
    idx: usize,
    prev_cost: i64,
    costs: &mut [i64],
    dist_array: &mut [u16],
) {
    let color = argb[idx];
    let cost_val;

    if let Some(ref mut c) = cache {
        if let Some(cache_idx) = c.lookup(color) {
            // Cache hit: use cache cost, scaled by 68%
            cost_val = prev_cost + div_round(cost_model.cache_cost(cache_idx) * 68, 100);
        } else {
            // Cache miss: insert and use literal cost, scaled by 82%
            c.insert(color);
            cost_val = prev_cost + div_round(cost_model.literal_cost(color) * 82, 100);
        }
    } else {
        cost_val = prev_cost + div_round(cost_model.literal_cost(color) * 82, 100);
    }

    if cost_val < costs[idx] {
        costs[idx] = cost_val;
        dist_array[idx] = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_log2() {
        assert_eq!(fast_log2(0), 0);
        assert_eq!(fast_log2(1), 0);
        // log2(2) = 1.0
        let result = fast_log2(2) as f64 / (1u64 << LOG_2_PRECISION_BITS) as f64;
        assert!((result - 1.0).abs() < 0.01, "log2(2)={result}");
        // log2(8) = 3.0
        let result = fast_log2(8) as f64 / (1u64 << LOG_2_PRECISION_BITS) as f64;
        assert!((result - 3.0).abs() < 0.01, "log2(8)={result}");
        // log2(256) = 8.0
        let result = fast_log2(256) as f64 / (1u64 << LOG_2_PRECISION_BITS) as f64;
        assert!((result - 8.0).abs() < 0.01, "log2(256)={result}");
    }

    #[test]
    fn test_div_round() {
        assert_eq!(div_round(100, 100), 1);
        assert_eq!(div_round(150, 100), 2); // 1.5 rounds to 2
        assert_eq!(div_round(149, 100), 1); // 1.49 rounds to 1
        assert_eq!(div_round(-150, 100), -2);
    }

    #[test]
    fn test_counts_to_bit_estimates() {
        // Single symbol: all costs zero (trivial code)
        let counts = vec![0, 100, 0, 0];
        let estimates = counts_to_bit_estimates(&counts);
        assert!(estimates.iter().all(|&e| e == 0));

        // Uniform distribution: all non-zero symbols have equal cost
        let counts = vec![100, 100, 100, 100];
        let estimates = counts_to_bit_estimates(&counts);
        assert_eq!(estimates[0], estimates[1]);
        assert_eq!(estimates[1], estimates[2]);
    }
}
