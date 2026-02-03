//! Entropy calculation for VP8L encoding.
//!
//! Estimates bit costs for Huffman-coded symbol sequences.

use super::histogram::Histogram;

/// Fixed-point precision for entropy calculations.
const LOG_2_PRECISION_BITS: u32 = 26;
const LOG_2_PRECISION: f64 = (1u64 << LOG_2_PRECISION_BITS) as f64;

/// Calculate Shannon entropy for a symbol distribution.
/// Returns entropy in fixed-point format (multiply by total count for total bits).
pub fn bits_entropy(counts: &[u32]) -> u64 {
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return 0;
    }

    let mut entropy = 0.0f64;
    for &count in counts {
        if count > 0 {
            let p = count as f64 / total as f64;
            entropy -= count as f64 * p.log2();
        }
    }

    (entropy * LOG_2_PRECISION) as u64
}

/// Estimate bit cost for a histogram using entropy.
pub fn estimate_histogram_bits(h: &Histogram) -> u64 {
    let literal_bits = bits_entropy(&h.literal);
    let red_bits = bits_entropy(&h.red);
    let blue_bits = bits_entropy(&h.blue);
    let alpha_bits = bits_entropy(&h.alpha);
    let distance_bits = bits_entropy(&h.distance);

    // Convert from fixed-point to approximate bit count
    (literal_bits + red_bits + blue_bits + alpha_bits + distance_bits) >> LOG_2_PRECISION_BITS
}

/// Estimate combined bit cost for merging two histograms.
pub fn estimate_combined_bits(h1: &Histogram, h2: &Histogram) -> u64 {
    // Temporarily merge and estimate
    let mut combined = h1.clone();
    combined.add(h2);
    estimate_histogram_bits(&combined)
}

/// Calculate bit cost savings from using a single symbol (trivial code).
/// A trivial symbol costs 0 bits per occurrence (just 1 bit to signal single-symbol tree).
pub fn trivial_symbol_savings(counts: &[u32]) -> u64 {
    let _total: u64 = counts.iter().map(|&c| c as u64).sum();
    let mut non_zero = 0u32;
    let mut single_symbol_count = 0u64;

    for &c in counts {
        if c > 0 {
            non_zero += 1;
            single_symbol_count = c as u64;
        }
    }

    if non_zero == 1 {
        // Single symbol: saves all entropy bits
        (single_symbol_count as f64 * LOG_2_PRECISION) as u64 >> LOG_2_PRECISION_BITS
    } else {
        0
    }
}

/// Fast log2 approximation for entropy estimation.
/// Uses integer-based approximation for no_std compatibility.
#[inline]
pub fn fast_log2(v: u32) -> f64 {
    if v == 0 {
        return 0.0;
    }
    // Use integer log2 plus linear interpolation for the fraction
    let floor = 31 - v.leading_zeros();
    let frac = (v as f64) / ((1u32 << floor) as f64) - 1.0;
    floor as f64 + frac * core::f64::consts::LN_2 // ln(2) correction for linear approx
}

/// Estimate bit cost of encoding a length code (prefix + extra bits).
#[inline]
pub fn length_cost(len: u16) -> u32 {
    let (code, _) = super::histogram::length_to_code(len);
    let extra_bits = super::histogram::length_code_extra_bits(code);
    // Approximate: prefix code cost + extra bits
    // Prefix codes are typically 2-5 bits depending on frequency
    3 + extra_bits as u32
}

/// Estimate bit cost of encoding a distance code (prefix + extra bits).
#[inline]
pub fn distance_cost(dist_code: u32) -> u32 {
    let (code, _) = super::histogram::distance_code_to_prefix(dist_code);
    let extra_bits = super::histogram::distance_code_extra_bits(code);
    // Approximate: prefix code cost + extra bits
    4 + extra_bits as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits_entropy_uniform() {
        // Uniform distribution: 4 symbols, each appearing 100 times
        let counts = [100u32; 4];
        let entropy = bits_entropy(&counts);
        // log2(4) = 2 bits per symbol, 400 symbols total = 800 bits
        // With fixed-point scaling
        let expected = (400.0 * 2.0 * LOG_2_PRECISION) as u64;
        assert!((entropy as i64 - expected as i64).unsigned_abs() < expected / 100);
    }

    #[test]
    fn test_bits_entropy_single() {
        // Single symbol: 0 entropy
        let mut counts = [0u32; 256];
        counts[42] = 100;
        let entropy = bits_entropy(&counts);
        assert_eq!(entropy, 0);
    }

    #[test]
    fn test_fast_log2() {
        assert!((fast_log2(1) - 0.0).abs() < 0.5);
        assert!((fast_log2(2) - 1.0).abs() < 0.5);
        assert!((fast_log2(4) - 2.0).abs() < 0.5);
        assert!((fast_log2(8) - 3.0).abs() < 0.5);
    }

    #[test]
    fn test_length_cost() {
        // Short lengths should have low cost
        assert!(length_cost(1) <= 5);
        assert!(length_cost(4) <= 5);
        // Longer lengths have higher cost due to extra bits
        assert!(length_cost(100) > length_cost(4));
        assert!(length_cost(1000) > length_cost(100));
    }
}
