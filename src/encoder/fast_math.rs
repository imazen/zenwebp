//! Fast math approximations for no_std.
//!
//! Provides round and pow without libm dependency.

/// Round f32 to nearest integer (ties to +inf, like libm::roundf).
/// Only handles non-negative values correctly (sufficient for our use).
#[inline]
pub(crate) fn roundf(x: f32) -> f32 {
    (x + 0.5) as i32 as f32
}

/// Round f64 to nearest integer (ties to +inf, like libm::round).
/// Only handles non-negative values correctly (sufficient for our use).
#[inline]
pub(crate) fn round(x: f64) -> f64 {
    (x + 0.5) as i64 as f64
}

/// Fast cube root approximation using Newton-Raphson.
/// Used for quality_to_compression: pow(x, 1/3).
#[inline]
pub(crate) fn cbrt(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }

    // Initial guess using bit manipulation
    // For x^(1/3), we can approximate by dividing the exponent by 3
    let bits = x.to_bits();
    let exp_bias = 1023_u64;
    let approx_bits = (bits / 3) + (exp_bias * 2 / 3) * (1 << 52);
    let mut y = f64::from_bits(approx_bits);

    // Newton-Raphson iterations: y = y - (y^3 - x) / (3*y^2)
    //                          = y * (2 + x/(y^3)) / 3
    //                          = (2*y + x/(y*y)) / 3
    for _ in 0..4 {
        let y2 = y * y;
        y = (2.0 * y + x / y2) / 3.0;
    }

    y
}

/// Fast power approximation using exp2(n * log2(x)).
/// For x in (0, 1] and n close to 1.0 (typical for our trellis quantization).
#[inline]
pub(crate) fn pow(x: f64, n: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x == 1.0 || n == 0.0 {
        return 1.0;
    }
    if n == 1.0 {
        return x;
    }

    // x^n = 2^(n * log2(x))
    exp2(n * log2(x))
}

/// Fast log2 approximation using (x-1)/(x+1) transform.
/// This gives better accuracy than direct polynomial on (x-1).
#[inline]
#[allow(clippy::excessive_precision)]
fn log2(x: f64) -> f64 {
    // Extract exponent and mantissa
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7FF) as i64 - 1023;
    let mantissa_bits = (bits & 0x000F_FFFF_FFFF_FFFF) | 0x3FF0_0000_0000_0000;
    let m = f64::from_bits(mantissa_bits); // m in [1, 2)

    // Use (m-1)/(m+1) transform for better convergence
    // log2(m) = 2/ln(2) * artanh((m-1)/(m+1))
    //         = 2/ln(2) * y * (1 + y^2/3 + y^4/5 + y^6/7 + ...)
    // where y = (m-1)/(m+1)
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;

    // Coefficients: 2/ln(2) * (1, 1/3, 1/5, 1/7, 1/9)
    const C0: f64 = 2.885_390_081_777_926_8; // 2/ln(2)
    const C1: f64 = 0.961_796_693_925_975_6; // 2/(3*ln(2))
    const C2: f64 = 0.577_078_016_355_585_4; // 2/(5*ln(2))
    const C3: f64 = 0.412_198_583_111_132_4; // 2/(7*ln(2))
    const C4: f64 = 0.320_598_898_753_103_0; // 2/(9*ln(2))

    let poly = C0 + y2 * (C1 + y2 * (C2 + y2 * (C3 + y2 * C4)));
    let log2_m = y * poly;

    exp as f64 + log2_m
}

/// Fast exp2 (2^x) approximation with degree-5 polynomial.
#[inline]
fn exp2(x: f64) -> f64 {
    // Clamp to prevent overflow/underflow
    let x = x.clamp(-1022.0, 1023.0);

    // Split into integer and fractional parts
    // Use floor for correct handling of negative values
    let xi = if x >= 0.0 { x as i64 } else { x as i64 - 1 };
    let xf = x - xi as f64;

    // Minimax polynomial for 2^xf where xf in [0, 1)
    // Coefficients from Taylor series of 2^x = e^(x*ln2)
    const LN2: f64 = core::f64::consts::LN_2;
    const C0: f64 = 1.0;
    const C1: f64 = LN2;
    const C2: f64 = LN2 * LN2 / 2.0;
    const C3: f64 = LN2 * LN2 * LN2 / 6.0;
    const C4: f64 = LN2 * LN2 * LN2 * LN2 / 24.0;
    const C5: f64 = LN2 * LN2 * LN2 * LN2 * LN2 / 120.0;

    let poly = C0 + xf * (C1 + xf * (C2 + xf * (C3 + xf * (C4 + xf * C5))));

    // Construct 2^xi by setting exponent bits
    let exp_bits = ((xi + 1023) as u64) << 52;
    let scale = f64::from_bits(exp_bits);

    poly * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundf() {
        assert_eq!(roundf(0.0), 0.0);
        assert_eq!(roundf(0.4), 0.0);
        assert_eq!(roundf(0.5), 1.0);
        assert_eq!(roundf(0.6), 1.0);
        assert_eq!(roundf(1.5), 2.0);
        assert_eq!(roundf(75.4), 75.0);
        assert_eq!(roundf(75.5), 76.0);
    }

    #[test]
    fn test_round() {
        assert_eq!(round(0.0), 0.0);
        assert_eq!(round(0.4), 0.0);
        assert_eq!(round(0.5), 1.0);
        assert_eq!(round(127.0 * 0.5), 64.0);
    }

    #[test]
    fn test_cbrt() {
        // Test cube root
        let test_cases = [
            (0.0, 0.0),
            (1.0, 1.0),
            (8.0, 2.0),
            (27.0, 3.0),
            (0.125, 0.5),
            (0.001, 0.1),
        ];

        for (x, expected) in test_cases {
            let result = cbrt(x);
            assert!(
                (result - expected).abs() < 1e-10,
                "cbrt({}) = {}, expected {}",
                x,
                result,
                expected
            );
        }
    }

    #[test]
    fn test_pow_basic() {
        // Test basic cases
        assert_eq!(pow(0.0, 2.0), 0.0);
        assert_eq!(pow(1.0, 5.0), 1.0);
        assert_eq!(pow(2.0, 0.0), 1.0);
        assert_eq!(pow(2.0, 1.0), 2.0);
    }

    #[test]
    #[allow(clippy::approx_constant)] // Intentional approximate values for test cases
    fn test_pow_typical_values() {
        // Test values typical for our trellis quantization
        // c_base in [0, 1], expn close to 1.0
        let test_cases = [
            (0.5, 1.0, 0.5),
            (0.5, 2.0, 0.25),
            (0.5, 0.5, 0.707_106_781),
            (0.8, 0.9, 0.821_871_788),
            (0.3, 1.1, 0.268_269_580),
        ];

        for (x, n, expected) in test_cases {
            let result = pow(x, n);
            let rel_err = (result - expected).abs() / expected.abs().max(1e-10);
            assert!(
                rel_err < 0.01, // 1% tolerance - quantizer is integer 0-127
                "pow({}, {}) = {}, expected {}, rel_err = {}",
                x,
                n,
                result,
                expected,
                rel_err
            );
        }
    }
}
