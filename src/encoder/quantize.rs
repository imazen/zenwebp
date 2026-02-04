//! Quantization matrix and coefficient quantization.
//!
//! Contains VP8Matrix for quantize/dequantize operations on 4x4 DCT blocks.

// Many loops in this file match libwebp's C patterns for clarity when comparing
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use super::tables::{MAX_LEVEL, VP8_FREQ_SHARPENING};

//------------------------------------------------------------------------------
// Quantization constants

/// Fixed-point precision for quantization
pub const QFIX: u32 = 17;

/// Bias calculation macro equivalent
#[inline]
pub const fn quantization_bias(b: u32) -> u32 {
    (((b) << (QFIX)) + 128) >> 8
}

/// Quantization division: (coeff * iq + bias) >> QFIX
#[inline]
pub fn quantdiv(coeff: u32, iq: u32, bias: u32) -> i32 {
    ((coeff as u64 * iq as u64 + bias as u64) >> QFIX) as i32
}

//------------------------------------------------------------------------------
// Quantization matrix

/// Quantization matrix for a coefficient type (Y1, Y2, UV)
#[derive(Clone, Debug)]
pub struct VP8Matrix {
    /// Quantizer steps for each coefficient position
    pub q: [u16; 16],
    /// Reciprocals (1 << QFIX) / q, for fast division
    pub iq: [u32; 16],
    /// Rounding bias for quantization
    pub bias: [u32; 16],
    /// Zero threshold: coefficients below this are quantized to 0
    pub zthresh: [u32; 16],
    /// Sharpening boost for high-frequency coefficients
    pub sharpen: [u16; 16],
}

impl VP8Matrix {
    /// Create a new quantization matrix from DC and AC quantizer values
    pub fn new(q_dc: u16, q_ac: u16, matrix_type: MatrixType) -> Self {
        let bias_values = match matrix_type {
            MatrixType::Y1 => (96, 110),  // luma-ac
            MatrixType::Y2 => (96, 108),  // luma-dc
            MatrixType::UV => (110, 115), // chroma
        };

        let mut m = Self {
            q: [0; 16],
            iq: [0; 16],
            bias: [0; 16],
            zthresh: [0; 16],
            sharpen: [0; 16],
        };

        // Set DC (index 0) and AC (index 1+) values
        m.q[0] = q_dc;
        m.q[1] = q_ac;

        // Calculate reciprocals, bias, and zero thresholds for DC and AC
        for i in 0..2 {
            let is_ac = i > 0;
            let bias = if is_ac { bias_values.1 } else { bias_values.0 };
            m.iq[i] = ((1u64 << QFIX) / m.q[i] as u64) as u32;
            m.bias[i] = quantization_bias(bias);
            // zthresh: value such that quantdiv(coeff, iq, bias) is 0 if coeff <= zthresh
            m.zthresh[i] = ((1 << QFIX) - 1 - m.bias[i]) / m.iq[i];
        }

        // Replicate AC values for positions 2-15
        for i in 2..16 {
            m.q[i] = m.q[1];
            m.iq[i] = m.iq[1];
            m.bias[i] = m.bias[1];
            m.zthresh[i] = m.zthresh[1];
        }

        // Apply sharpening for Y1 matrix (luma AC)
        if matches!(matrix_type, MatrixType::Y1) {
            const SHARPEN_BITS: u32 = 11;
            for (i, &freq_sharpen) in VP8_FREQ_SHARPENING.iter().enumerate() {
                m.sharpen[i] = ((freq_sharpen as u32 * m.q[i] as u32) >> SHARPEN_BITS) as u16;
            }
        }

        m
    }

    /// Get the average quantizer value (for lambda calculations)
    pub fn average_q(&self) -> u32 {
        let sum: u32 = self.q.iter().map(|&x| x as u32).sum();
        (sum + 8) >> 4
    }

    /// Quantize a single coefficient (matches libwebp's QuantizeBlock_C)
    ///
    /// Adds sharpen boost to absolute coefficient value, then checks against
    /// zthresh to skip coefficients guaranteed to quantize to zero.
    #[inline]
    pub fn quantize_coeff(&self, coeff: i32, pos: usize) -> i32 {
        let sign = coeff < 0;
        let abs_coeff = (if sign { -coeff } else { coeff } as u32) + self.sharpen[pos] as u32;
        if abs_coeff <= self.zthresh[pos] {
            return 0;
        }
        let level = quantdiv(abs_coeff, self.iq[pos], self.bias[pos]).min(MAX_LEVEL as i32);
        if sign {
            -level
        } else {
            level
        }
    }

    /// Quantize a single coefficient with neutral bias (for trellis)
    #[inline]
    pub fn quantize_neutral(&self, coeff: i32, pos: usize) -> i32 {
        let sign = coeff < 0;
        let abs_coeff = if sign { -coeff } else { coeff } as u32;
        let neutral_bias = quantization_bias(0x00); // neutral
        let level = quantdiv(abs_coeff, self.iq[pos], neutral_bias);
        if sign {
            -level
        } else {
            level
        }
    }

    /// Dequantize a coefficient
    #[inline]
    pub fn dequantize(&self, level: i32, pos: usize) -> i32 {
        level * self.q[pos] as i32
    }

    /// Quantize an entire 4x4 block of coefficients in place
    ///
    /// Includes sharpen boost and zthresh check matching libwebp's QuantizeBlock_C.
    pub fn quantize(&self, coeffs: &mut [i32; 16]) {
        for (pos, coeff) in coeffs.iter_mut().enumerate() {
            let sign = *coeff < 0;
            let abs_coeff = (if sign { -*coeff } else { *coeff } as u32) + self.sharpen[pos] as u32;
            if abs_coeff <= self.zthresh[pos] {
                *coeff = 0;
                continue;
            }
            let level = quantdiv(abs_coeff, self.iq[pos], self.bias[pos]).min(MAX_LEVEL as i32);
            *coeff = if sign { -level } else { level };
        }
    }

    /// Quantize only AC coefficients (positions 1-15) in place, leaving DC unchanged
    /// This is used for Y1 blocks where the DC goes to the Y2 block
    ///
    /// Includes sharpen boost and zthresh check matching libwebp's QuantizeBlock_C.
    #[allow(clippy::needless_range_loop)] // pos indexes both coeffs and self.iq/self.bias
    pub fn quantize_ac_only(&self, coeffs: &mut [i32; 16]) {
        for pos in 1..16 {
            let sign = coeffs[pos] < 0;
            let abs_coeff =
                (if sign { -coeffs[pos] } else { coeffs[pos] } as u32) + self.sharpen[pos] as u32;
            if abs_coeff <= self.zthresh[pos] {
                coeffs[pos] = 0;
                continue;
            }
            let level = quantdiv(abs_coeff, self.iq[pos], self.bias[pos]).min(MAX_LEVEL as i32);
            coeffs[pos] = if sign { -level } else { level };
        }
    }

    /// Dequantize an entire 4x4 block of coefficients in place
    pub fn dequantize_block(&self, coeffs: &mut [i32; 16]) {
        for (pos, coeff) in coeffs.iter_mut().enumerate() {
            *coeff *= self.q[pos] as i32;
        }
    }

    /// Dequantize only AC coefficients (positions 1-15) in place
    #[allow(clippy::needless_range_loop)] // pos indexes both coeffs and self.q
    pub fn dequantize_ac_only(&self, coeffs: &mut [i32; 16]) {
        for pos in 1..16 {
            coeffs[pos] *= self.q[pos] as i32;
        }
    }
}

/// Matrix type for bias selection
#[derive(Clone, Copy, Debug)]
pub enum MatrixType {
    /// Luma AC coefficients
    Y1,
    /// Luma DC (WHT) coefficients
    Y2,
    /// Chroma coefficients
    UV,
}
