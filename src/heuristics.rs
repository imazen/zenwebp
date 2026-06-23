//! Resource estimation heuristics for encoding and decoding operations.
//!
//! Provides approximate estimates for memory consumption and relative time
//! costs. Use these for pre-allocating buffers, sizing thread pools, memory
//! budgeting, and progress estimation.
//!
//! # Accuracy
//!
//! Estimates are based on profiling zenwebp's pure-Rust implementation using
//! `examples/profile_memory.rs`. Throughput measurements are empirically derived
//! from 580+ encode and 20+ decode operations across various sizes and configurations.
//!
//! Actual usage varies with content complexity:
//! - Simple content (solid colors, gradients): lower bound
//! - Typical photos (gradients, natural scenes): middle estimate
//! - High-entropy content (noise, detailed textures): upper bound
//!
//! To regenerate estimates:
//! ```bash
//! cargo build --release --example profile_memory
//! target/release/examples/profile_memory encode > encode_profile.csv
//! target/release/examples/profile_memory decode > decode_profile.csv
//! cargo run --release --example analyze_profiles encode_profile.csv decode_profile.csv
//! ```
//!
//! # Example
//!
//! ```rust
//! use zenwebp::heuristics::{estimate_encode, estimate_decode};
//! use zenwebp::EncoderConfig;
//!
//! let est = estimate_encode(1920, 1080, 4, &EncoderConfig::new_lossy());
//! println!("Peak memory: ~{:.1} MB", est.peak_memory_bytes as f64 / 1_000_000.0);
//!
//! let dec = estimate_decode(1920, 1080, 4);
//! println!("Decode memory: ~{:.1} MB", dec.peak_memory_bytes as f64 / 1_000_000.0);
//! ```

use crate::encoder::EncoderConfig;

// =============================================================================
// Encoder constants (measured via heaptrack/valgrind on zenwebp, 2026-02)
//
// zenwebp allocates differently from libwebp: pure Rust, Vec-based buffers,
// no C malloc. Measurements are from callgrind + heaptrack runs.
// =============================================================================

// Heaptrack/VmHWM-calibrated 2026-06-23 (benchmarks/zenwebp_encode_mem_2026-06-23.tsv,
// bike.png 256²..4096² × lossy/lossless × method{0,2,4,6} × q{50,85}). Method-INDEPENDENT
// (the method knob changes CPU time, not the working set). Admission control gates on
// `_EST`, so `_EST` is fit as a SAFE UPPER BOUND: it clears every measured cell — including
// the concave 1-4MP working-set hump that a least-squares line under-fits (webp memory vs
// size is sub-linear: lossy 14.7→3.0 B/px over 256²→4096²) — so it never under-predicts the
// gating value. It is intentionally loose at small sizes: the raised intercept that covers
// the mid-range concavity over-predicts tiny images, which is harmless (cheap to over-reserve)
// and per the gating semantics the est need not be tight, only safe. `_MAX` = requested heap
// (heaptrack peak) — a loose ceiling, NOT the gate, so it need not be close.

/// Fixed overhead for lossy encoding (YUV planes, segment structs, token buffers); sized so
/// `_EST` covers the concave 1-4MP hump (measured 2048² ≈ 19.3 MB; the asymptotic slope alone
/// would under-predict it).
const LOSSY_FIXED_OVERHEAD: u64 = 8_500_000;
/// Lossy bytes/pixel — `_EST` = safe-upper-bound slope (gating), `_MAX` = requested-heap ceiling.
/// `_EST` carries ~10-15% margin over measured (the gating value should be conservative).
const LOSSY_BYTES_PER_PIXEL_EST: f64 = 3.4;
const LOSSY_BYTES_PER_PIXEL_MAX: f64 = 8.0;

/// Fixed overhead for lossless encoding (hash chains, histograms, huffman tables); sized so
/// `_EST` covers the concave mid-range (measured 1024²/2048² need a ~15 MB intercept under the
/// asymptotic slope).
const LOSSLESS_FIXED_OVERHEAD: u64 = 18_000_000;
/// Lossless bytes/pixel — `_EST` = safe-upper-bound slope (gating), `_MAX` = requested-heap ceiling.
/// `_EST` carries ~10% margin over measured (the gating value should be conservative).
const LOSSLESS_BYTES_PER_PIXEL_EST: f64 = 24.0;
const LOSSLESS_BYTES_PER_PIXEL_MAX: f64 = 28.0;

// =============================================================================
// Decoder constants
// =============================================================================

/// Fixed overhead for decoding (~100KB for VP8/VP8L decoder state).
const DECODE_FIXED_OVERHEAD: u64 = 100_000;

/// Bytes per pixel for decoding.
/// Includes: row cache, reconstruction buffer, loop filter working space.
/// FILTER_PADDING adds ~57KB. Output buffer is separate.
const DECODE_BYTES_PER_PIXEL: f64 = 12.0;

// =============================================================================
// Throughput constants (measured via examples/profile_memory.rs, 2026-02-06)
// =============================================================================

/// Lossy encode throughput at method 4 in Mpix/s.
const LOSSY_ENCODE_THROUGHPUT_MPIXELS: f64 = 14.5; // Measured average across sizes

/// Lossless encode throughput in Mpix/s (method 6).
const LOSSLESS_ENCODE_THROUGHPUT_MPIXELS: f64 = 221.0; // Measured average

/// Decode throughput in Mpix/s (mixed lossy/lossless).
/// Lossy: ~85 Mpix/s, Lossless: ~352 Mpix/s, using conservative estimate.
const DECODE_THROUGHPUT_MPIXELS: f64 = 150.0;

/// Resource estimation for encoding operations.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct EncodeEstimate {
    /// Minimum expected peak memory (best case: simple content).
    pub peak_memory_bytes_min: u64,
    /// Typical peak memory (average case: natural photos).
    pub peak_memory_bytes: u64,
    /// Maximum expected peak memory (worst case: noise, high-entropy).
    pub peak_memory_bytes_max: u64,
    /// Estimated encode time in milliseconds (typical content).
    pub time_ms: f32,
    /// Estimated output size in bytes.
    pub output_bytes: u64,
}

/// Resource estimation for decoding operations.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct DecodeEstimate {
    /// Typical peak memory in bytes during decoding (excluding output buffer).
    pub peak_memory_bytes: u64,
    /// Output buffer size in bytes.
    pub output_bytes: u64,
    /// Estimated decode time in milliseconds.
    pub time_ms: f32,
}

/// Estimate resources for encoding an image.
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `bpp` - Bytes per pixel of input (3 for RGB, 4 for RGBA)
/// * `config` - Encoder configuration
#[must_use]
pub fn estimate_encode(width: u32, height: u32, bpp: u8, config: &EncoderConfig) -> EncodeEstimate {
    let pixels = (width as u64) * (height as u64);
    let input_bytes = pixels * (bpp as u64);

    let method = config.get_method();
    let is_lossless = config.is_lossless();

    // Method-independent working set; two measured tiers. `method` still drives the
    // time estimate below, just not memory.
    let (fixed, bpp_est, bpp_max) = if is_lossless {
        (
            LOSSLESS_FIXED_OVERHEAD,
            LOSSLESS_BYTES_PER_PIXEL_EST,
            LOSSLESS_BYTES_PER_PIXEL_MAX,
        )
    } else {
        (
            LOSSY_FIXED_OVERHEAD,
            LOSSY_BYTES_PER_PIXEL_EST,
            LOSSY_BYTES_PER_PIXEL_MAX,
        )
    };

    let peak_memory_bytes = fixed + (pixels as f64 * bpp_est) as u64;
    let peak_memory_bytes_max = fixed + (pixels as f64 * bpp_max) as u64;
    let peak_memory_bytes_min = (peak_memory_bytes as f64 * 0.85) as u64;

    // Output estimate
    let output_ratio = if is_lossless {
        0.5 // ~50% of input
    } else {
        let q = config.get_quality() as f64;
        0.02 + (q / 100.0) * 0.18
    };
    let output_bytes = (input_bytes as f64 * output_ratio) as u64;

    // Time estimate (multiplier relative to method 4)
    // Measured: m0=25.7, m2=16.7, m4=14.5, m6=11.1 Mpix/s
    let method_speed = match method {
        0 => 25.7 / 14.5, // 1.77x faster than method 4
        1 => 1.5,         // interpolated
        2 => 16.7 / 14.5, // 1.15x faster than method 4
        3 => 1.08,        // interpolated
        4 => 1.0,         // baseline
        5 => 0.95,        // interpolated
        _ => 11.1 / 14.5, // 0.77x (method 6, slower than 4)
    };

    let throughput = if is_lossless {
        LOSSLESS_ENCODE_THROUGHPUT_MPIXELS * method_speed
    } else {
        LOSSY_ENCODE_THROUGHPUT_MPIXELS * method_speed
    };
    let time_ms = (pixels as f64 / (throughput * 1000.0)) as f32;

    EncodeEstimate {
        peak_memory_bytes_min,
        peak_memory_bytes,
        peak_memory_bytes_max,
        time_ms,
        output_bytes,
    }
}

/// Estimate resources for decoding an image.
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `output_bpp` - Bytes per pixel of output (3 for RGB, 4 for RGBA)
#[must_use]
pub fn estimate_decode(width: u32, height: u32, output_bpp: u8) -> DecodeEstimate {
    let pixels = (width as u64) * (height as u64);
    let output_bytes = pixels * (output_bpp as u64);

    let peak_memory_bytes = DECODE_FIXED_OVERHEAD + (pixels as f64 * DECODE_BYTES_PER_PIXEL) as u64;

    let time_ms = (pixels as f64 / (DECODE_THROUGHPUT_MPIXELS * 1000.0)) as f32;

    DecodeEstimate {
        peak_memory_bytes,
        output_bytes,
        time_ms,
    }
}

/// Estimate resources for encoding an animation.
///
/// Peak memory is approximately one frame's worth plus encoder state
/// and previous-frame canvas for delta compression.
#[must_use]
pub fn estimate_animation_encode(
    width: u32,
    height: u32,
    frame_count: u32,
    config: &EncoderConfig,
) -> EncodeEstimate {
    let single = estimate_encode(width, height, 4, config);
    let canvas_bytes = (width as u64) * (height as u64) * 4;

    // Animation encoder keeps prev_canvas (RGBA) + current encode buffers
    let peak_memory_bytes = single.peak_memory_bytes + canvas_bytes;

    EncodeEstimate {
        peak_memory_bytes_min: single.peak_memory_bytes_min + canvas_bytes,
        peak_memory_bytes,
        peak_memory_bytes_max: single.peak_memory_bytes_max + canvas_bytes,
        time_ms: single.time_ms * frame_count as f32,
        output_bytes: single.output_bytes * frame_count as u64,
    }
}

/// Estimate resources for decoding an animation.
#[must_use]
pub fn estimate_animation_decode(width: u32, height: u32, frame_count: u32) -> DecodeEstimate {
    let single = estimate_decode(width, height, 4);
    let canvas_bytes = (width as u64) * (height as u64) * 4;

    DecodeEstimate {
        // Decoder holds canvas + decode buffers
        peak_memory_bytes: single.peak_memory_bytes + canvas_bytes,
        output_bytes: single.output_bytes * frame_count as u64,
        time_ms: single.time_ms * frame_count as f32,
    }
}
