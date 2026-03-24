//! Chroma dithering for lossy VP8 decoding.
//!
//! Adds small random noise to U/V planes after loop filtering to hide
//! banding artifacts from coarse chroma quantization at low quality.
//!
//! Matches libwebp's default dithering behavior (strength 50).

use crate::common::types::MAX_SEGMENTS;

const VP8_RANDOM_TABLE_SIZE: usize = 55;
const VP8_RANDOM_DITHER_FIX: u32 = 8;
const VP8_DITHER_AMP_BITS: u32 = 7;
const VP8_DITHER_AMP_CENTER: i32 = 1 << VP8_DITHER_AMP_BITS;
const VP8_DITHER_DESCALE: u32 = 4;
const VP8_DITHER_DESCALE_ROUNDER: i32 = 1 << (VP8_DITHER_DESCALE - 1);
const MIN_DITHER_AMP: i32 = 4;

/// Maps UV AC quantizer index to dither amplitude.
/// Lower quantizer indices (higher quality) get less dithering.
const DITHER_AMP_TAB: [u8; 12] = [8, 7, 6, 4, 4, 2, 2, 2, 1, 1, 1, 1];

/// libwebp's precomputed random table from `random_utils.c` (31-bit range values).
/// Knuth difference-based PRNG seeded with these constants.
const RANDOM_TABLE: [u32; VP8_RANDOM_TABLE_SIZE] = [
    0x0de15230, 0x03b31886, 0x775faccb, 0x1c88626a, 0x68385c55, 0x14b3b828, 0x4a85fef8, 0x49ddb84b,
    0x64fcf397, 0x5c550289, 0x4a290000, 0x0d7ec1da, 0x5940b7ab, 0x5492577d, 0x4e19ca72, 0x38d38c69,
    0x0c01ee65, 0x32a1755f, 0x5437f652, 0x5abb2c32, 0x0faa57b1, 0x73f533e7, 0x685feeda, 0x7563cce2,
    0x6e990e83, 0x4730a7ed, 0x4fc0d9c6, 0x496b153c, 0x4f1403fa, 0x541afb0c, 0x73990b32, 0x26d7cb1c,
    0x6fcc3706, 0x2cbb77d8, 0x75762f2a, 0x6425ccdd, 0x24b35461, 0x0a7d8715, 0x220414a8, 0x141ebf67,
    0x56b41583, 0x73e502e3, 0x44cab16f, 0x28264d42, 0x73baaefb, 0x0a50ebed, 0x1d6ab6fb, 0x0d3ad40b,
    0x35db3b68, 0x2b081e83, 0x77ce6b95, 0x5181e5f0, 0x78853bbc, 0x009f9494, 0x27e5ed3c,
];

/// Knuth difference-based pseudo-random number generator.
/// Matches libwebp's `VP8Random` from `random_utils.h`.
pub(crate) struct VP8Random {
    index1: usize,
    index2: usize,
    tab: [u32; VP8_RANDOM_TABLE_SIZE],
}

impl VP8Random {
    /// Create a new PRNG with the standard libwebp seed table.
    /// Equivalent to `VP8InitRandom(rg, 1.0f)`.
    pub(crate) fn new() -> Self {
        Self {
            index1: 0,
            index2: 31,
            tab: RANDOM_TABLE,
        }
    }

    /// Generate a centered pseudo-random value with `num_bits` amplitude,
    /// scaled by `amp` (in `VP8_RANDOM_DITHER_FIX` fixed-point precision).
    ///
    /// Matches libwebp's `VP8RandomBits2`.
    fn random_bits2(&mut self, num_bits: u32, amp: i32) -> i32 {
        let diff = self.tab[self.index1].wrapping_sub(self.tab[self.index2]);
        // Wrap to 31-bit range (matching libwebp: `if (diff < 0) diff += (1u << 31)`)
        let diff = diff & 0x7FFF_FFFF;
        self.tab[self.index1] = diff;
        self.index1 += 1;
        if self.index1 == VP8_RANDOM_TABLE_SIZE {
            self.index1 = 0;
        }
        self.index2 += 1;
        if self.index2 == VP8_RANDOM_TABLE_SIZE {
            self.index2 = 0;
        }
        // sign-extend, 0-center
        let diff = ((diff << 1) as i32) >> (32 - num_bits as i32);
        // restrict range
        let diff = (diff * amp) >> VP8_RANDOM_DITHER_FIX;
        // shift back to 0.5-center
        diff + (1 << (num_bits - 1))
    }
}

/// Compute per-segment dither amplitudes from UV AC quantizer indices
/// and user-specified dithering strength (0-100).
///
/// Returns `(any_enabled, per_segment_amplitudes)`.
pub(crate) fn init_dither_amplitudes(
    uv_quant_indices: &[i32; MAX_SEGMENTS],
    strength: u8,
) -> (bool, [i32; MAX_SEGMENTS]) {
    let max_amp = (1i32 << VP8_RANDOM_DITHER_FIX) - 1;
    let f = if strength == 0 {
        0
    } else {
        (i32::from(strength.min(100)) * max_amp) / 100
    };

    let mut amps = [0i32; MAX_SEGMENTS];
    let mut any_nonzero = false;

    if f > 0 {
        for (s, &uv_quant) in uv_quant_indices.iter().enumerate() {
            let idx = uv_quant.clamp(0, DITHER_AMP_TAB.len() as i32 - 1) as usize;
            if uv_quant < DITHER_AMP_TAB.len() as i32 {
                amps[s] = (f * i32::from(DITHER_AMP_TAB[idx])) >> 3;
            }
            if amps[s] >= MIN_DITHER_AMP {
                any_nonzero = true;
            } else {
                amps[s] = 0; // below threshold, skip
            }
        }
    }

    (any_nonzero, amps)
}

/// Apply dithering to an 8x8 chroma block in-place.
///
/// Generates random deltas centered around zero and adds them to pixel values,
/// clamping to [0, 255]. Matches libwebp's `DitherCombine8x8_C`.
fn dither_8x8(rg: &mut VP8Random, dst: &mut [u8], stride: usize, amp: i32) {
    for j in 0..8 {
        let row_start = j * stride;
        for i in 0..8 {
            let dither_val = rg.random_bits2(VP8_DITHER_AMP_BITS + 1, amp);
            let delta0 = dither_val - VP8_DITHER_AMP_CENTER;
            let delta1 = (delta0 + VP8_DITHER_DESCALE_ROUNDER) >> VP8_DITHER_DESCALE;
            let val = i32::from(dst[row_start + i]) + delta1;
            dst[row_start + i] = val.clamp(0, 255) as u8;
        }
    }
}

/// Parameters for dithering a row of macroblocks.
pub(crate) struct DitherRowParams<'a> {
    pub cache_u: &'a mut [u8],
    pub cache_v: &'a mut [u8],
    pub cache_uv_stride: usize,
    pub extra_uv_rows: usize,
    /// Pre-computed per-MB dither amplitudes. Zero means skip that MB
    /// (either below threshold, coefficients skipped, or has UV AC content).
    pub mb_dither_amps: &'a [i32],
}

/// Dither all U/V macroblocks in a decoded row.
///
/// Called after loop filtering, before output. Each macroblock's chroma
/// gets dithered based on its pre-computed amplitude.
///
/// Matching libwebp: only MBs with all-zero UV AC coefficients are dithered.
/// MBs with actual chroma detail or skipped coefficients are left untouched.
pub(crate) fn dither_row(rg: &mut VP8Random, params: DitherRowParams<'_>) {
    let cache_offset = params.extra_uv_rows * params.cache_uv_stride;
    for (mbx, &amp) in params.mb_dither_amps.iter().enumerate() {
        if amp >= MIN_DITHER_AMP {
            let offset = cache_offset + mbx * 8;
            dither_8x8(
                rg,
                &mut params.cache_u[offset..],
                params.cache_uv_stride,
                amp,
            );
            dither_8x8(
                rg,
                &mut params.cache_v[offset..],
                params.cache_uv_stride,
                amp,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_deterministic() {
        let mut rg = VP8Random::new();
        // Generate some values and check they're deterministic
        let v1 = rg.random_bits2(VP8_DITHER_AMP_BITS + 1, 100);
        let v2 = rg.random_bits2(VP8_DITHER_AMP_BITS + 1, 100);
        assert_ne!(v1, v2, "consecutive values should differ");

        // Re-create and verify same sequence
        let mut rg2 = VP8Random::new();
        let v1b = rg2.random_bits2(VP8_DITHER_AMP_BITS + 1, 100);
        let v2b = rg2.random_bits2(VP8_DITHER_AMP_BITS + 1, 100);
        assert_eq!(v1, v1b, "same seed should produce same sequence");
        assert_eq!(v2, v2b);
    }

    #[test]
    fn test_random_range() {
        let mut rg = VP8Random::new();
        for _ in 0..1000 {
            let val = rg.random_bits2(VP8_DITHER_AMP_BITS + 1, 255);
            // Values should be roughly centered around VP8_DITHER_AMP_CENTER (128)
            // After subtracting center and descaling, deltas should be small
            let delta0 = val - VP8_DITHER_AMP_CENTER;
            let delta1 = (delta0 + VP8_DITHER_DESCALE_ROUNDER) >> VP8_DITHER_DESCALE;
            assert!(
                delta1.abs() <= 16,
                "delta {delta1} out of expected range for amp=255"
            );
        }
    }

    #[test]
    fn test_dither_amplitudes_zero_strength() {
        let indices = [0; MAX_SEGMENTS];
        let (enabled, amps) = init_dither_amplitudes(&indices, 0);
        assert!(!enabled);
        assert_eq!(amps, [0; MAX_SEGMENTS]);
    }

    #[test]
    fn test_dither_amplitudes_low_quant() {
        // Low quantizer index = high quality = significant dither amplitude
        let indices = [0, 1, 2, 3];
        let (enabled, amps) = init_dither_amplitudes(&indices, 50);
        assert!(enabled);
        // All should have non-zero amplitudes for low quant indices at strength 50
        for &amp in &amps {
            assert!(amp > 0, "low quant index should produce non-zero amp");
        }
    }

    #[test]
    fn test_dither_amplitudes_high_quant() {
        // High quantizer index = low quality = less or no dithering
        let indices = [100, 100, 100, 100];
        let (enabled, amps) = init_dither_amplitudes(&indices, 50);
        assert!(!enabled);
        assert_eq!(amps, [0; MAX_SEGMENTS]);
    }

    #[test]
    fn test_dither_8x8_modifies_pixels() {
        let mut rg = VP8Random::new();
        let stride = 16;
        let mut buf = vec![128u8; stride * 8];
        let original = buf.clone();
        dither_8x8(&mut rg, &mut buf, stride, 100);
        // At least some pixels should be modified
        assert_ne!(buf, original, "dithering should modify pixel values");
    }

    #[test]
    fn test_dither_8x8_stays_in_bounds() {
        let mut rg = VP8Random::new();
        let stride = 8;
        // Test with extreme values
        let mut buf_low = vec![0u8; stride * 8];
        dither_8x8(&mut rg, &mut buf_low, stride, 255);
        // All values should still be valid u8

        let mut rg2 = VP8Random::new();
        let mut buf_high = vec![255u8; stride * 8];
        dither_8x8(&mut rg2, &mut buf_high, stride, 255);
        // All values should still be valid u8 (clamped)
    }
}
