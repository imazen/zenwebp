//! Entropy calculation for VP8L encoding.
//!
//! Matches libwebp's PopulationCost/BitsEntropyRefine approach
//! for accurate histogram cost estimation used in clustering.

#![allow(clippy::too_many_arguments)]

use super::histogram::Histogram;

/// Fixed-point precision for entropy calculations (matching libwebp).
const LOG_2_PRECISION_BITS: u32 = 23;

/// Lookup table for v * log2(v) in fixed-point (for v in 0..256).
/// kSLog2Table[v] = v * log2(v) * (1 << LOG_2_PRECISION_BITS).
fn fast_slog2(v: u32) -> u64 {
    if v == 0 {
        return 0;
    }
    if v < 256 {
        // For small values, compute directly with float
        let vf = v as f64;
        return (vf * vf.log2() * (1u64 << LOG_2_PRECISION_BITS) as f64) as u64;
    }
    // For large values, use libwebp's piecewise approximation
    fast_slog2_slow(v)
}

/// Extended range SLog2 for v >= 256 (matches libwebp's FastSLog2Slow_C).
fn fast_slog2_slow(v: u32) -> u64 {
    // Use float for values >= 256
    let vf = v as f64;
    (vf * vf.log2() * (1u64 << LOG_2_PRECISION_BITS) as f64) as u64
}

/// RLE streak statistics matching libwebp's VP8LStreaks.
#[derive(Debug, Default, Clone)]
struct Streaks {
    /// counts[0] = # of zero streaks > 3, counts[1] = # of nonzero streaks > 3
    counts: [u32; 2],
    /// streaks[is_nonzero][is_long]: total length of streaks
    /// [0][0] = zero streak length ≤3, [0][1] = zero streak length >3
    /// [1][0] = nonzero streak length ≤3, [1][1] = nonzero streak length >3
    streaks: [[u32; 2]; 2],
}

/// Bit entropy result matching libwebp's VP8LBitEntropy.
#[derive(Debug, Default, Clone)]
struct BitEntropy {
    entropy: u64,
    sum: u32,
    nonzeros: u32,
    max_val: u32,
    nonzero_code: u16,
}

/// Process a run-length streak (matches libwebp's GetEntropyUnrefinedHelper).
#[inline]
fn entropy_unrefined_helper(
    val: u32,
    i: usize,
    val_prev: &mut u32,
    i_prev: &mut usize,
    bit_entropy: &mut BitEntropy,
    stats: &mut Streaks,
) {
    let streak = (i - *i_prev) as u32;

    if *val_prev != 0 {
        bit_entropy.sum += *val_prev * streak;
        bit_entropy.nonzeros += streak;
        bit_entropy.nonzero_code = *i_prev as u16;
        bit_entropy.entropy += fast_slog2(*val_prev) * streak as u64;
        if bit_entropy.max_val < *val_prev {
            bit_entropy.max_val = *val_prev;
        }
    }

    let is_nonzero = (*val_prev != 0) as usize;
    let is_long = (streak > 3) as usize;
    stats.counts[is_nonzero] += is_long as u32;
    stats.streaks[is_nonzero][is_long] += streak;

    *val_prev = val;
    *i_prev = i;
}

/// Calculate entropy + streak stats for a single distribution (matches VP8LGetEntropyUnrefined).
fn get_entropy_unrefined(x: &[u32]) -> (BitEntropy, Streaks) {
    let mut bit_entropy = BitEntropy::default();
    let mut stats = Streaks::default();

    if x.is_empty() {
        return (bit_entropy, stats);
    }

    let mut i_prev = 0usize;
    let mut x_prev = x[0];

    for (i, &xv) in x.iter().enumerate().skip(1) {
        if xv != x_prev {
            entropy_unrefined_helper(
                xv,
                i,
                &mut x_prev,
                &mut i_prev,
                &mut bit_entropy,
                &mut stats,
            );
        }
    }
    entropy_unrefined_helper(
        0,
        x.len(),
        &mut x_prev,
        &mut i_prev,
        &mut bit_entropy,
        &mut stats,
    );

    bit_entropy.entropy = fast_slog2(bit_entropy.sum).saturating_sub(bit_entropy.entropy);

    (bit_entropy, stats)
}

/// Calculate combined entropy for TWO distributions without merging them.
/// Matches libwebp's GetCombinedEntropyUnrefined_C.
fn get_combined_entropy_unrefined(x: &[u32], y: &[u32]) -> (BitEntropy, Streaks) {
    debug_assert_eq!(x.len(), y.len());
    let mut bit_entropy = BitEntropy::default();
    let mut stats = Streaks::default();

    if x.is_empty() {
        return (bit_entropy, stats);
    }

    let mut i_prev = 0usize;
    let mut xy_prev = x[0] + y[0];

    for i in 1..x.len() {
        let xy = x[i] + y[i];
        if xy != xy_prev {
            entropy_unrefined_helper(
                xy,
                i,
                &mut xy_prev,
                &mut i_prev,
                &mut bit_entropy,
                &mut stats,
            );
        }
    }
    entropy_unrefined_helper(
        0,
        x.len(),
        &mut xy_prev,
        &mut i_prev,
        &mut bit_entropy,
        &mut stats,
    );

    bit_entropy.entropy = fast_slog2(bit_entropy.sum).saturating_sub(bit_entropy.entropy);

    (bit_entropy, stats)
}

/// Refine entropy using perceptual adjustments (matches BitsEntropyRefine).
fn bits_entropy_refine(entropy: &BitEntropy) -> u64 {
    if entropy.nonzeros < 5 {
        if entropy.nonzeros <= 1 {
            return 0; // Trivial: 0 or 1 symbol
        }
        if entropy.nonzeros == 2 {
            // Two symbols: mix 99% entropy + 1% bias
            return div_round(
                99 * ((entropy.sum as u64) << LOG_2_PRECISION_BITS) + entropy.entropy,
                100,
            );
        }
        let mix = if entropy.nonzeros == 3 {
            950u64
        } else {
            700u64
        };
        let min_limit = (2 * entropy.sum as u64 - entropy.max_val as u64) << LOG_2_PRECISION_BITS;
        let min_limit = div_round(mix * min_limit + (1000 - mix) * entropy.entropy, 1000);
        return if entropy.entropy < min_limit {
            min_limit
        } else {
            entropy.entropy
        };
    }

    // >= 5 symbols: use mix of 627/1000
    let mix = 627u64;
    let min_limit = (2 * entropy.sum as u64 - entropy.max_val as u64) << LOG_2_PRECISION_BITS;
    let min_limit = div_round(mix * min_limit + (1000 - mix) * entropy.entropy, 1000);
    if entropy.entropy < min_limit {
        min_limit
    } else {
        entropy.entropy
    }
}

/// Initial Huffman tree encoding cost (matches libwebp's InitialHuffmanCost).
fn initial_huffman_cost() -> u64 {
    const CODE_LENGTH_CODES: u64 = 19;
    let base = CODE_LENGTH_CODES * 3; // 57 bits for the code length codes
    (base << LOG_2_PRECISION_BITS) - div_round(91u64 << LOG_2_PRECISION_BITS, 10)
}

/// Huffman tree cost from RLE stats (matches libwebp's FinalHuffmanCost).
fn final_huffman_cost(stats: &Streaks) -> u64 {
    let mut retval = initial_huffman_cost();

    let retval_extra: u32 = stats.counts[0] * 1600
        + 240 * stats.streaks[0][1]
        + stats.counts[1] * 2640
        + 720 * stats.streaks[1][1]
        + 1840 * stats.streaks[0][0]
        + 3360 * stats.streaks[1][0];

    retval += (retval_extra as u64) << (LOG_2_PRECISION_BITS - 10);
    retval
}

/// Rounding division matching libwebp's DivRound.
#[inline]
fn div_round(num: u64, den: u64) -> u64 {
    (num + den / 2) / den
}

/// Cost to encode a single population (entropy + Huffman tree cost).
/// Returns (cost, trivial_sym, is_used).
pub fn population_cost(population: &[u32]) -> (u64, Option<u16>, bool) {
    let (bit_entropy, stats) = get_entropy_unrefined(population);

    let trivial_sym = if bit_entropy.nonzeros == 1 {
        Some(bit_entropy.nonzero_code)
    } else {
        None
    };

    let is_used = stats.streaks[1][0] != 0 || stats.streaks[1][1] != 0;
    let cost = bits_entropy_refine(&bit_entropy) + final_huffman_cost(&stats);

    (cost, trivial_sym, is_used)
}

/// Cost to encode two populations combined (without modifying them).
fn combined_entropy(x: &[u32], y: &[u32]) -> u64 {
    let (bit_entropy, stats) = get_combined_entropy_unrefined(x, y);
    bits_entropy_refine(&bit_entropy) + final_huffman_cost(&stats)
}

/// Per-type cost info cached in histogram.
#[derive(Debug, Clone, Default)]
pub struct HistogramCosts {
    /// Total bit cost across all 5 types.
    pub total: u64,
    /// Per-type costs [literal, red, blue, alpha, distance].
    pub per_type: [u64; 5],
    /// Trivial symbol per type (Some(sym) if only one symbol).
    pub trivial_sym: [Option<u16>; 5],
    /// Whether each type has any nonzero entries.
    pub is_used: [bool; 5],
}

/// Compute full histogram cost (all 5 types).
pub fn compute_histogram_cost(h: &Histogram) -> HistogramCosts {
    let (lit_cost, lit_triv, lit_used) = population_cost(&h.literal);
    let (red_cost, red_triv, red_used) = population_cost(&h.red);
    let (blue_cost, blue_triv, blue_used) = population_cost(&h.blue);
    let (alpha_cost, alpha_triv, alpha_used) = population_cost(&h.alpha);
    let (dist_cost, dist_triv, dist_used) = population_cost(&h.distance);

    HistogramCosts {
        total: lit_cost + red_cost + blue_cost + alpha_cost + dist_cost,
        per_type: [lit_cost, red_cost, blue_cost, alpha_cost, dist_cost],
        trivial_sym: [lit_triv, red_triv, blue_triv, alpha_triv, dist_triv],
        is_used: [lit_used, red_used, blue_used, alpha_used, dist_used],
    }
}

/// Get combined entropy cost of two histograms for a single type index.
/// Fast path: handles trivial/unused cases without computation.
fn get_combined_cost_for_type(
    h1: &Histogram,
    h1_costs: &HistogramCosts,
    h2: &Histogram,
    h2_costs: &HistogramCosts,
    type_idx: usize,
) -> u64 {
    let h1_used = h1_costs.is_used[type_idx];
    let h2_used = h2_costs.is_used[type_idx];

    // Fast path: trivial symbol match
    let is_trivial = h1_costs.trivial_sym[type_idx].is_some()
        && h1_costs.trivial_sym[type_idx] == h2_costs.trivial_sym[type_idx];

    if is_trivial || !h1_used || !h2_used {
        return if h1_used {
            h1_costs.per_type[type_idx]
        } else {
            h2_costs.per_type[type_idx]
        };
    }

    // Full calculation: combine populations
    let (x, y) = get_populations_for_type(h1, h2, type_idx);
    combined_entropy(x, y)
}

/// Get population arrays for a given type index.
fn get_populations_for_type<'a>(
    h1: &'a Histogram,
    h2: &'a Histogram,
    type_idx: usize,
) -> (&'a [u32], &'a [u32]) {
    match type_idx {
        0 => (h1.literal.as_slice(), h2.literal.as_slice()),
        1 => (&h1.red, &h2.red),
        2 => (&h1.blue, &h2.blue),
        3 => (&h1.alpha, &h2.alpha),
        4 => (&h1.distance, &h2.distance),
        _ => unreachable!(),
    }
}

/// Evaluate merging two histograms with cost threshold (early bail-out).
/// Returns Some(combined_cost) if under threshold, None if exceeds.
pub fn get_combined_histogram_cost(
    h1: &Histogram,
    h1_costs: &HistogramCosts,
    h2: &Histogram,
    h2_costs: &HistogramCosts,
    cost_threshold: u64,
) -> Option<u64> {
    if cost_threshold == 0 {
        return None;
    }

    let mut cost = 0u64;
    for i in 0..5 {
        cost += get_combined_cost_for_type(h1, h1_costs, h2, h2_costs, i);
        if cost >= cost_threshold {
            return None; // Early bail-out
        }
    }

    Some(cost)
}

/// Estimate bit cost for a histogram using entropy (simple version for backward compat).
pub fn estimate_histogram_bits(h: &Histogram) -> u64 {
    let costs = compute_histogram_cost(h);
    costs.total >> LOG_2_PRECISION_BITS
}

/// Estimate combined bit cost for merging two histograms.
pub fn estimate_combined_bits(h1: &Histogram, h2: &Histogram) -> u64 {
    let mut combined = h1.clone();
    combined.add(h2);
    estimate_histogram_bits(&combined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::vp8l::types::make_argb;

    #[test]
    fn test_fast_slog2() {
        // slog2(1) = 1 * log2(1) = 0
        assert_eq!(fast_slog2(0), 0);
        assert_eq!(fast_slog2(1), 0);
        // slog2(2) = 2 * log2(2) = 2 * 1 = 2 (scaled)
        let s2 = fast_slog2(2);
        let expected = 2.0 * (1u64 << LOG_2_PRECISION_BITS) as f64;
        assert!((s2 as f64 - expected).abs() < expected * 0.01);
    }

    #[test]
    fn test_population_cost_trivial() {
        let mut counts = [0u32; 256];
        counts[42] = 100;
        let (cost, trivial, _) = population_cost(&counts);
        assert_eq!(trivial, Some(42));
        // Single symbol → entropy is 0, cost is only Huffman tree overhead
        assert!(cost > 0, "Tree overhead should be nonzero");
    }

    #[test]
    fn test_population_cost_two_symbols() {
        let mut counts = [0u32; 256];
        counts[0] = 50;
        counts[128] = 50;
        let (cost, trivial, is_used) = population_cost(&counts);
        assert!(trivial.is_none());
        assert!(is_used);
        assert!(cost > 0);
    }

    #[test]
    fn test_compute_histogram_cost() {
        let mut h = Histogram::new(0);
        h.add_literal(make_argb(255, 128, 64, 32));
        h.add_literal(make_argb(255, 128, 64, 32));
        let costs = compute_histogram_cost(&h);
        assert!(costs.total > 0);
        // All same pixel → each type should be trivial
        assert!(costs.trivial_sym[0].is_some()); // literal (green)
        assert!(costs.trivial_sym[1].is_some()); // red
        assert!(costs.trivial_sym[2].is_some()); // blue
        assert!(costs.trivial_sym[3].is_some()); // alpha
    }

    #[test]
    fn test_combined_cost_threshold() {
        let mut h1 = Histogram::new(0);
        let mut h2 = Histogram::new(0);
        for i in 0..100u32 {
            h1.add_literal(make_argb(255, (i % 16) as u8, (i % 8) as u8, (i % 4) as u8));
            h2.add_literal(make_argb(
                255,
                ((i + 50) % 16) as u8,
                ((i + 50) % 8) as u8,
                ((i + 50) % 4) as u8,
            ));
        }
        let c1 = compute_histogram_cost(&h1);
        let c2 = compute_histogram_cost(&h2);

        // With very high threshold, should succeed
        let result = get_combined_histogram_cost(&h1, &c1, &h2, &c2, u64::MAX);
        assert!(result.is_some());

        // With 0 threshold, should fail
        let result = get_combined_histogram_cost(&h1, &c1, &h2, &c2, 0);
        assert!(result.is_none());
    }
}
