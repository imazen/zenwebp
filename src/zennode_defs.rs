//! Zennode pipeline node definitions for WebP encoding.
//!
//! Provides two encode nodes with distinct parameter spaces:
//!
//! - [`EncodeWebpLossy`] — VP8 lossy encoding (quality, effort, sharp YUV, etc.)
//! - [`EncodeWebpLossless`] — VP8L lossless encoding (effort, near-lossless, exact)
//!
//! Each node exposes a `to_encoder_config()` method that produces the native
//! [`EncoderConfig`](crate::EncoderConfig) used by zenwebp's encoder.
//!
//! Feature-gated behind `feature = "zennode"`.

use zennode::*;

use crate::encoder::config::{EncoderConfig, LosslessConfig, LossyConfig};

// ── Lossy (VP8) ─────────────────────────────────────────────────────────────

/// WebP lossy (VP8) encode node for pipeline use.
///
/// Maps directly to [`LossyConfig`] via [`to_encoder_config()`](Self::to_encoder_config).
/// Advanced parameters use `-1` as a sentinel meaning "use preset default."
#[derive(Node, Clone, Debug)]
#[node(id = "zenwebp.encode_lossy", group = Encode, role = Encode)]
#[node(tags("webp", "lossy", "encode"))]
pub struct EncodeWebpLossy {
    /// Lossy quality (0 = smallest file, 100 = best quality).
    #[param(range(0.0..=100.0), default = 75.0, step = 1.0)]
    #[param(section = "Main", label = "Quality")]
    #[kv("webp.quality", "webp.q")]
    pub quality: f32,

    /// Speed/quality tradeoff (0 = fastest, 10 = best compression).
    /// Mapped internally to WebP method 0-6.
    #[param(range(0..=10), default = 7)]
    #[param(section = "Main", label = "Effort")]
    #[kv("webp.effort")]
    pub effort: u32,

    /// Use iterative (sharp) YUV conversion for better chroma fidelity.
    #[param(default = true)]
    #[param(section = "Main", label = "Sharp YUV")]
    #[kv("webp.sharp_yuv")]
    pub sharp_yuv: bool,

    /// Alpha channel quality (0 = smallest, 100 = lossless alpha).
    #[param(range(0..=100), default = 100)]
    #[param(section = "Alpha", label = "Alpha Quality")]
    #[kv("webp.alpha_quality", "webp.aq")]
    pub alpha_quality: u32,

    /// Spatial noise shaping strength (0-100). -1 = use preset default.
    #[param(range(-1..=100), default = -1)]
    #[param(section = "Advanced", label = "SNS Strength")]
    #[kv("webp.sns")]
    pub sns_strength: i32,

    /// Loop filter strength (0-100). -1 = use preset default.
    #[param(range(-1..=100), default = -1)]
    #[param(section = "Advanced", label = "Filter Strength")]
    #[kv("webp.filter")]
    pub filter_strength: i32,

    /// Loop filter sharpness (0-7). -1 = use preset default.
    #[param(range(-1..=7), default = -1)]
    #[param(section = "Advanced", label = "Filter Sharpness")]
    #[kv("webp.sharpness")]
    pub filter_sharpness: i32,
}

impl Default for EncodeWebpLossy {
    fn default() -> Self {
        Self {
            quality: 75.0,
            effort: 7,
            sharp_yuv: true,
            alpha_quality: 100,
            sns_strength: -1,
            filter_strength: -1,
            filter_sharpness: -1,
        }
    }
}

impl EncodeWebpLossy {
    /// Convert this node into a native [`EncoderConfig`] for zenwebp.
    ///
    /// Effort 0-10 is mapped to WebP method 0-6.
    /// Parameters at their sentinel value (-1) inherit from the preset defaults.
    #[must_use]
    pub fn to_encoder_config(&self) -> EncoderConfig {
        let method = ((self.effort as u64 * 6) / 10).min(6) as u8;
        let mut cfg = LossyConfig::new()
            .with_quality(self.quality)
            .with_method(method)
            .with_sharp_yuv(self.sharp_yuv)
            .with_alpha_quality(self.alpha_quality.min(100) as u8);

        if self.sns_strength >= 0 {
            cfg = cfg.with_sns_strength(self.sns_strength.min(100) as u8);
        }
        if self.filter_strength >= 0 {
            cfg = cfg.with_filter_strength(self.filter_strength.min(100) as u8);
        }
        if self.filter_sharpness >= 0 {
            cfg = cfg.with_filter_sharpness(self.filter_sharpness.min(7) as u8);
        }

        EncoderConfig::Lossy(cfg)
    }
}

// ── Lossless (VP8L) ─────────────────────────────────────────────────────────

/// WebP lossless (VP8L) encode node for pipeline use.
///
/// Maps directly to [`LosslessConfig`] via [`to_encoder_config()`](Self::to_encoder_config).
///
/// The `effort` parameter (0-100) controls both the internal quality setting
/// and the method level (0-6), derived as `round(effort / 100 * 6)`.
#[derive(Node, Clone, Debug)]
#[node(id = "zenwebp.encode_lossless", group = Encode, role = Encode)]
#[node(tags("webp", "lossless", "encode"))]
pub struct EncodeWebpLossless {
    /// Compression effort (0 = fastest, 100 = best compression).
    /// Controls both the VP8L quality parameter and method level.
    #[param(range(0.0..=100.0), default = 75.0, step = 1.0)]
    #[param(section = "Main", label = "Compression Effort")]
    #[kv("webp.effort")]
    pub effort: f32,

    /// Near-lossless preprocessing (0 = max preprocessing, 100 = fully lossless).
    /// Values below 100 allow slight color changes for better compression.
    #[param(range(0..=100), default = 100)]
    #[param(section = "Main", label = "Near-Lossless")]
    #[kv("webp.near_lossless", "webp.nl")]
    pub near_lossless: u32,

    /// Preserve exact RGB values under fully transparent pixels.
    /// When false, RGB under alpha=0 may be modified for better compression.
    #[param(default = false)]
    #[param(section = "Advanced")]
    #[kv("webp.exact")]
    pub exact: bool,
}

impl Default for EncodeWebpLossless {
    fn default() -> Self {
        Self {
            effort: 75.0,
            near_lossless: 100,
            exact: false,
        }
    }
}

impl EncodeWebpLossless {
    /// Convert this node into a native [`EncoderConfig`] for zenwebp.
    ///
    /// Effort 0-100 maps to:
    /// - VP8L quality = effort (direct passthrough)
    /// - VP8L method = round(effort / 100 * 6), clamped to 0-6
    #[must_use]
    pub fn to_encoder_config(&self) -> EncoderConfig {
        let effort = self.effort.clamp(0.0, 100.0);
        let method = ((effort / 100.0 * 6.0) + 0.5) as u8;
        let method = method.min(6);

        let cfg = LosslessConfig::new()
            .with_quality(effort)
            .with_method(method)
            .with_near_lossless(self.near_lossless.min(100) as u8)
            .with_exact(self.exact);

        EncoderConfig::Lossless(cfg)
    }
}
