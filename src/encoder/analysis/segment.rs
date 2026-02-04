//! Segment assignment using k-means clustering.
//!
//! VP8 supports up to 4 segments with different quantization levels.
//! This module clusters macroblocks by their alpha (compressibility) values
//! and computes appropriate quantization for each segment.
//!
//! Ported from libwebp src/enc/analysis_enc.c

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

extern crate alloc;
use alloc::vec;

use super::{MAX_ALPHA, NUM_SEGMENTS};

/// Number of k-means iterations for segment assignment
const MAX_ITERS_K_MEANS: usize = 6;

/// Assign macroblocks to segments using k-means clustering on alpha values.
///
/// # Arguments
/// * `alphas` - Alpha histogram (count of macroblocks with each alpha value)
/// * `num_segments` - Number of segments to use (1-4)
///
/// # Returns
/// (centers, map, weighted_average) where:
/// - centers\[i\] = alpha center for segment i
/// - map\[alpha\] = segment index for that alpha value
/// - weighted_average = weighted average of centers (for SetSegmentAlphas)
pub fn assign_segments_kmeans(
    alphas: &[u32; 256],
    num_segments: usize,
) -> ([u8; NUM_SEGMENTS], [u8; 256], i32) {
    let num_segments = num_segments.min(NUM_SEGMENTS);
    let mut centers = [0u8; NUM_SEGMENTS];
    let mut map = [0u8; 256];

    // Find min and max alpha with non-zero count
    let mut min_a = 0usize;
    let mut max_a = MAX_ALPHA as usize;

    for (n, &count) in alphas.iter().enumerate() {
        if count > 0 {
            min_a = n;
            break;
        }
    }
    for n in (min_a..=MAX_ALPHA as usize).rev() {
        if alphas[n] > 0 {
            max_a = n;
            break;
        }
    }

    let range_a = max_a.saturating_sub(min_a);

    // Initialize centers evenly spread across the range
    for (k, center) in centers.iter_mut().enumerate().take(num_segments) {
        let n = 1 + 2 * k;
        *center = (min_a + (n * range_a) / (2 * num_segments)) as u8;
    }

    // K-means iterations
    let mut accum = [0u32; NUM_SEGMENTS];
    let mut dist_accum = [0u32; NUM_SEGMENTS];
    let mut weighted_average = 0i32;
    let mut total_weight = 0u32;

    for _ in 0..MAX_ITERS_K_MEANS {
        // Reset accumulators
        for i in 0..num_segments {
            accum[i] = 0;
            dist_accum[i] = 0;
        }

        // Assign each alpha value to nearest center
        let mut current_center = 0usize;
        for a in min_a..=max_a {
            if alphas[a] > 0 {
                // Find nearest center
                while current_center + 1 < num_segments {
                    let d_curr = (a as i32 - centers[current_center] as i32).abs();
                    let d_next = (a as i32 - centers[current_center + 1] as i32).abs();
                    if d_next < d_curr {
                        current_center += 1;
                    } else {
                        break;
                    }
                }
                map[a] = current_center as u8;
                dist_accum[current_center] += a as u32 * alphas[a];
                accum[current_center] += alphas[a];
            }
        }

        // Move centers to center of their clouds
        // Also compute weighted_average from final centers (as libwebp does)
        let mut displaced = 0i32;
        weighted_average = 0;
        total_weight = 0;
        for n in 0..num_segments {
            if accum[n] > 0 {
                let new_center = ((dist_accum[n] + accum[n] / 2) / accum[n]) as u8;
                displaced += (centers[n] as i32 - new_center as i32).abs();
                centers[n] = new_center;
                // libwebp computes weighted_average from final centers
                weighted_average += new_center as i32 * accum[n] as i32;
                total_weight += accum[n];
            }
        }

        // Early exit if centers have converged
        if displaced < 5 {
            break;
        }
    }

    // Finalize weighted_average with rounding (matching libwebp)
    if total_weight > 0 {
        weighted_average = (weighted_average + total_weight as i32 / 2) / total_weight as i32;
    } else {
        weighted_average = 128;
    }

    // Fill unused segments with last valid center
    for i in num_segments..NUM_SEGMENTS {
        centers[i] = centers[num_segments - 1];
    }

    (centers, map, weighted_average)
}

/// Convert user-facing quality (0-100) to compression factor.
/// Emulates jpeg-like behaviour where Q75 is "good quality".
/// Ported from libwebp's QualityToCompression().
fn quality_to_compression(quality: u8) -> f64 {
    let c = f64::from(quality) / 100.0;
    // Piecewise linear mapping to get jpeg-like behavior at Q75
    let linear_c = if c < 0.75 {
        c * (2.0 / 3.0)
    } else {
        2.0 * c - 1.0
    };
    // File size roughly scales as pow(quantizer, 3), so we use inverse
    crate::encoder::fast_math::cbrt(linear_c)
}

/// Compute per-segment quantization using libwebp's formula.
///
/// This matches VP8SetSegmentParams in libwebp/src/enc/quant_enc.c
///
/// # Arguments
/// * `quality` - User-facing quality (0-100), used to compute base compression factor
/// * `segment_alpha` - Transformed alpha for this segment, in range [-127, 127].
///   Computed as: 255 * (center - mid) / (max - min).
///   Positive = easier to compress, negative = harder.
/// * `sns_strength` - SNS strength (0-100), higher = more segment differentiation
///
/// # Returns
/// Adjusted quantizer index for this segment
pub fn compute_segment_quant(quality: u8, segment_alpha: i32, sns_strength: u8) -> u8 {
    // libwebp constant: scaling between SNS strength and quantizer modulation
    const SNS_TO_DQ: f64 = 0.9;

    // Amplitude of quantization modulation
    // amp = SNS_TO_DQ * sns_strength / 100 / 128
    let amp = SNS_TO_DQ * (sns_strength as f64) / 100.0 / 128.0;

    // Exponent for power-law modulation
    // segment_alpha is in [-127, 127] range
    // Positive alpha (easy) -> expn < 1 -> higher compression -> higher quant
    // Negative alpha (hard) -> expn > 1 -> lower compression -> lower quant
    let expn = 1.0 - amp * (segment_alpha as f64);

    // Ensure expn is positive (as asserted in libwebp)
    if expn <= 0.0 {
        // Fallback: compute base quant from quality
        let c = quality_to_compression(quality);
        let q = crate::encoder::fast_math::round(127.0 * (1.0 - c)) as i32;
        return q.clamp(0, 127) as u8;
    }

    // Compression factor computed directly from quality (matches libwebp's QualityToCompression)
    // BUG FIX: Previously we computed c_base from base_quant, which double-applied segment alpha
    let c_base = quality_to_compression(quality);

    // Apply power-law modulation
    let c = crate::encoder::fast_math::pow(c_base, expn);

    // Convert back to quantizer index
    let q = (127.0 * (1.0 - c)) as i32;
    q.clamp(0, 127) as u8
}

/// Smooth the segment map by replacing isolated blocks with the majority of neighbors.
///
/// Uses a 3x3 majority filter: if 5 or more of the 8 neighbors share a segment ID,
/// the center block is reassigned to that segment. Border blocks are not modified.
///
/// This reduces noisy segment boundaries which can cause visual artifacts and
/// increase encoded size due to frequent segment ID changes.
///
/// Ported from libwebp's SmoothSegmentMap.
pub fn smooth_segment_map(segment_map: &mut [u8], mb_w: usize, mb_h: usize) {
    if mb_w < 3 || mb_h < 3 {
        return; // Too small to smooth
    }

    const MAJORITY_THRESHOLD: u8 = 5; // 5 out of 8 neighbors must agree

    // Create temporary buffer for smoothed values
    let mut tmp = vec![0u8; mb_w * mb_h];

    // Copy original values (borders won't be modified)
    tmp.copy_from_slice(segment_map);

    // Process interior blocks only (skip borders)
    for y in 1..mb_h - 1 {
        for x in 1..mb_w - 1 {
            let idx = x + y * mb_w;
            let mut counts = [0u8; NUM_SEGMENTS];
            let current_seg = segment_map[idx];

            // Count 8 neighbors
            counts[segment_map[idx - mb_w - 1] as usize] += 1; // top-left
            counts[segment_map[idx - mb_w] as usize] += 1; // top
            counts[segment_map[idx - mb_w + 1] as usize] += 1; // top-right
            counts[segment_map[idx - 1] as usize] += 1; // left
            counts[segment_map[idx + 1] as usize] += 1; // right
            counts[segment_map[idx + mb_w - 1] as usize] += 1; // bottom-left
            counts[segment_map[idx + mb_w] as usize] += 1; // bottom
            counts[segment_map[idx + mb_w + 1] as usize] += 1; // bottom-right

            // Find majority segment (if any)
            let mut majority_seg = current_seg;
            for (seg, &count) in counts.iter().enumerate() {
                if count >= MAJORITY_THRESHOLD {
                    majority_seg = seg as u8;
                    break;
                }
            }
            tmp[idx] = majority_seg;
        }
    }

    // Copy smoothed values back
    segment_map.copy_from_slice(&tmp);
}
