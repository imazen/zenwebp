//! Zennode pipeline node definitions for WebP encoding and decoding.
//!
//! Provides three nodes with distinct parameter spaces:
//!
//! - [`EncodeWebpLossy`] — VP8 lossy encoding (quality, effort, sharp YUV, etc.)
//! - [`EncodeWebpLossless`] — VP8L lossless encoding (effort, near-lossless, exact)
//! - [`DecodeWebp`] — WebP decoding (upsampling, dithering)
//!
//! Each encode node exposes a `to_encoder_config()` method that produces the native
//! [`EncoderConfig`](crate::EncoderConfig) used by zenwebp's encoder.
//!
//! When the `zencodec` feature is also enabled, `apply()` and
//! `to_webp_encoder_config()` produce a [`WebpEncoderConfig`](crate::WebpEncoderConfig)
//! that integrates with the zencodec trait system.
//!
//! Feature-gated behind `feature = "zennode"`.

use zennode::*;

use crate::encoder::config::{EncoderConfig, LosslessConfig, LossyConfig};

/// Map a uniform effort value (0.0-100.0) to WebP method (0-6).
///
/// Uses rounding: method = round(effort / 100 * 6), clamped to 0-6.
fn effort_to_method(effort: f32) -> u8 {
    let effort = effort.clamp(0.0, 100.0);
    let method = ((effort / 100.0 * 6.0) + 0.5) as u8;
    method.min(6)
}

/// Parse a preset string (case-insensitive) into a [`Preset`](crate::Preset).
///
/// Returns `None` for unrecognized values.
fn parse_preset(s: &str) -> Option<crate::Preset> {
    match s.to_ascii_lowercase().as_str() {
        "default" => Some(crate::Preset::Default),
        "picture" => Some(crate::Preset::Picture),
        "photo" => Some(crate::Preset::Photo),
        "drawing" => Some(crate::Preset::Drawing),
        "icon" => Some(crate::Preset::Icon),
        "text" => Some(crate::Preset::Text),
        "auto" => Some(crate::Preset::Auto),
        _ => None,
    }
}

// ── Lossy (VP8) ─────────────────────────────────────────────────────────────

/// WebP lossy (VP8) encode node for pipeline use.
///
/// Maps directly to [`LossyConfig`] via [`to_encoder_config()`](Self::to_encoder_config).
/// All parameters are `Option<T>` — `None` means "inherit from base config / use
/// preset default," which is critical for correct overlay semantics.
///
/// ## Effort mapping
///
/// Effort (0-100) is a uniform scale mapped to WebP method (0-6):
/// `method = round(effort / 100 * 6)`.
///
/// | Effort | Method | Description |
/// |--------|--------|-------------|
/// | 0      | 0      | Fastest     |
/// | 17     | 1      | Fast        |
/// | 33     | 2      | Fast+       |
/// | 50     | 3      | Balanced    |
/// | 67     | 4      | Good        |
/// | 83     | 5      | Better      |
/// | 100    | 6      | Best        |
#[derive(Node, Clone, Debug, Default)]
#[node(id = "zenwebp.encode_lossy", group = Encode, role = Encode)]
#[node(tags("webp", "lossy", "encode"))]
pub struct EncodeWebpLossy {
    /// Lossy quality (0 = smallest file, 100 = best quality).
    #[param(range(0.0..=100.0), step = 1.0)]
    #[param(section = "Main", label = "Quality")]
    #[kv("webp.quality", "webp.q")]
    pub quality: Option<f32>,

    /// Speed/quality tradeoff (0 = fastest, 100 = best compression).
    /// Mapped to WebP method 0-6 via `round(effort / 100 * 6)`.
    #[param(range(0.0..=100.0), step = 1.0)]
    #[param(section = "Main", label = "Effort")]
    #[kv("webp.effort")]
    pub effort: Option<f32>,

    /// Content-aware preset: "default", "picture", "photo", "drawing",
    /// "icon", "text", or "auto". Affects SNS, filter, and sharpness defaults.
    #[param(section = "Main", label = "Preset")]
    #[kv("webp.preset")]
    pub preset: Option<String>,

    /// Use iterative (sharp) YUV conversion for better chroma fidelity.
    /// Default: false (matches libwebp).
    #[param(section = "Main", label = "Sharp YUV")]
    #[kv("webp.sharp_yuv")]
    pub sharp_yuv: Option<bool>,

    /// Alpha channel quality (0 = smallest, 100 = lossless alpha).
    #[param(range(0..=100))]
    #[param(section = "Alpha", label = "Alpha Quality")]
    #[kv("webp.alpha_quality", "webp.aq")]
    pub alpha_quality: Option<u32>,

    /// Target file size in bytes. 0 or None = disabled.
    /// When set, the encoder adjusts quality to converge on this size.
    #[param(section = "Target", label = "Target Size")]
    #[kv("webp.target_size")]
    pub target_size: Option<u32>,

    /// Target PSNR in dB. 0.0 or None = disabled.
    /// When set, the encoder adjusts quality to converge on this PSNR.
    #[param(range(0.0..=100.0), step = 0.1)]
    #[param(section = "Target", label = "Target PSNR")]
    #[kv("webp.target_psnr")]
    pub target_psnr: Option<f32>,

    /// Number of segments for adaptive quantization (1-4).
    /// More segments allow finer per-region quantization but are slower.
    #[param(range(1..=4))]
    #[param(section = "Advanced", label = "Segments")]
    #[kv("webp.segments")]
    pub segments: Option<u32>,

    /// Spatial noise shaping strength (0-100). None = use preset default.
    #[param(range(0..=100))]
    #[param(section = "Advanced", label = "SNS Strength")]
    #[kv("webp.sns")]
    pub sns_strength: Option<u32>,

    /// Loop filter strength (0-100). None = use preset default.
    #[param(range(0..=100))]
    #[param(section = "Advanced", label = "Filter Strength")]
    #[kv("webp.filter")]
    pub filter_strength: Option<u32>,

    /// Loop filter sharpness (0-7). None = use preset default.
    #[param(range(0..=7))]
    #[param(section = "Advanced", label = "Filter Sharpness")]
    #[kv("webp.sharpness")]
    pub filter_sharpness: Option<u32>,
}

impl EncodeWebpLossy {
    /// Default quality when building a standalone [`EncoderConfig`].
    const DEFAULT_QUALITY: f32 = 75.0;
    /// Default effort (0-100) when building a standalone [`EncoderConfig`].
    const DEFAULT_EFFORT: f32 = 70.0;
    /// Default sharp-YUV setting when building a standalone [`EncoderConfig`].
    /// Matches libwebp default (false).
    const DEFAULT_SHARP_YUV: bool = false;
    /// Default alpha quality when building a standalone [`EncoderConfig`].
    const DEFAULT_ALPHA_QUALITY: u32 = 100;

    /// Convert this node into a native [`EncoderConfig`] for zenwebp.
    ///
    /// Effort 0-100 is mapped to WebP method 0-6 via `round(effort / 100 * 6)`.
    /// Values are clamped to valid ranges before conversion.
    #[must_use]
    pub fn to_encoder_config(&self) -> EncoderConfig {
        let quality = self
            .quality
            .unwrap_or(Self::DEFAULT_QUALITY)
            .clamp(0.0, 100.0);
        let effort = self
            .effort
            .unwrap_or(Self::DEFAULT_EFFORT)
            .clamp(0.0, 100.0);
        let sharp_yuv = self.sharp_yuv.unwrap_or(Self::DEFAULT_SHARP_YUV);
        let alpha_quality = self
            .alpha_quality
            .unwrap_or(Self::DEFAULT_ALPHA_QUALITY)
            .min(100);

        let method = effort_to_method(effort);
        let mut cfg = LossyConfig::new()
            .with_quality(quality)
            .with_method(method)
            .with_sharp_yuv(sharp_yuv)
            .with_alpha_quality(alpha_quality as u8);

        if let Some(preset) = self.preset.as_deref().and_then(parse_preset) {
            cfg = cfg.with_preset_value(preset);
        }
        if let Some(v) = self.target_size {
            cfg = cfg.with_target_size(v);
        }
        if let Some(v) = self.target_psnr {
            cfg = cfg.with_target_psnr(v.clamp(0.0, 100.0));
        }
        if let Some(v) = self.segments {
            cfg = cfg.with_segments(v.clamp(1, 4) as u8);
        }
        if let Some(v) = self.sns_strength {
            cfg = cfg.with_sns_strength(v.min(100) as u8);
        }
        if let Some(v) = self.filter_strength {
            cfg = cfg.with_filter_strength(v.min(100) as u8);
        }
        if let Some(v) = self.filter_sharpness {
            cfg = cfg.with_filter_sharpness(v.min(7) as u8);
        }

        EncoderConfig::Lossy(cfg)
    }
}

#[cfg(feature = "zencodec")]
impl EncodeWebpLossy {
    /// Apply this node's explicitly-set params on top of an existing
    /// [`WebpEncoderConfig`](crate::WebpEncoderConfig).
    ///
    /// `None` fields are skipped so that the base config's values are preserved.
    pub fn apply(&self, mut config: crate::WebpEncoderConfig) -> crate::WebpEncoderConfig {
        if let Some(q) = self.quality {
            config = config.with_quality(q.clamp(0.0, 100.0));
        }
        if let Some(e) = self.effort {
            let method = effort_to_method(e);
            config = config.with_method(method);
        }
        if let Some(preset) = self.preset.as_deref().and_then(parse_preset) {
            config = config.with_preset_value(preset);
        }
        if let Some(s) = self.sharp_yuv {
            config = config.with_sharp_yuv(s);
        }
        if let Some(aq) = self.alpha_quality {
            config = config.with_alpha_quality_value(aq.min(100) as f32);
        }
        if let Some(v) = self.target_size {
            config = config.with_target_size(v);
        }
        if let Some(v) = self.target_psnr {
            config = config.with_target_psnr(v.clamp(0.0, 100.0));
        }
        if let Some(v) = self.segments {
            config = config.with_segments(v.clamp(1, 4) as u8);
        }
        if let Some(v) = self.sns_strength {
            config = config.with_sns_strength(v.min(100) as u8);
        }
        if let Some(v) = self.filter_strength {
            config = config.with_filter_strength(v.min(100) as u8);
        }
        if let Some(v) = self.filter_sharpness {
            config = config.with_filter_sharpness(v.min(7) as u8);
        }

        config
    }

    /// Build a [`WebpEncoderConfig`](crate::WebpEncoderConfig) from scratch
    /// using this node's params.
    #[must_use]
    pub fn to_webp_encoder_config(&self) -> crate::WebpEncoderConfig {
        self.apply(crate::WebpEncoderConfig::lossy())
    }
}

// ── Lossless (VP8L) ─────────────────────────────────────────────────────────

/// WebP lossless (VP8L) encode node for pipeline use.
///
/// Maps directly to [`LosslessConfig`] via [`to_encoder_config()`](Self::to_encoder_config).
///
/// ## Effort mapping
///
/// Effort (0-100) is a uniform scale mapped to both:
/// - VP8L quality = effort (direct passthrough)
/// - VP8L method = round(effort / 100 * 6), clamped to 0-6
///
/// All parameters are `Option<T>` — `None` means "inherit from base config,"
/// which is critical for correct overlay semantics.
#[derive(Node, Clone, Debug, Default)]
#[node(id = "zenwebp.encode_lossless", group = Encode, role = Encode)]
#[node(tags("webp", "lossless", "encode"))]
pub struct EncodeWebpLossless {
    /// Compression effort (0 = fastest, 100 = best compression).
    /// Controls both the VP8L quality parameter and method level.
    #[param(range(0.0..=100.0), step = 1.0)]
    #[param(section = "Main", label = "Compression Effort")]
    #[kv("webp.effort")]
    pub effort: Option<f32>,

    /// Near-lossless preprocessing (0 = max preprocessing, 100 = fully lossless).
    /// Values below 100 allow slight color changes for better compression.
    #[param(range(0..=100))]
    #[param(section = "Main", label = "Near-Lossless")]
    #[kv("webp.near_lossless", "webp.nl")]
    pub near_lossless: Option<u32>,

    /// Preserve exact RGB values under fully transparent pixels.
    /// When false, RGB under alpha=0 may be modified for better compression.
    #[param(section = "Advanced")]
    #[kv("webp.exact")]
    pub exact: Option<bool>,

    /// Alpha channel quality (0-100, 100 = lossless alpha).
    #[param(range(0..=100))]
    #[param(section = "Alpha", label = "Alpha Quality")]
    #[kv("webp.alpha_quality", "webp.aq")]
    pub alpha_quality: Option<u32>,

    /// Target file size in bytes. 0 or None = disabled.
    #[param(section = "Target", label = "Target Size")]
    #[kv("webp.target_size")]
    pub target_size: Option<u32>,
}

impl EncodeWebpLossless {
    /// Default effort when building a standalone [`EncoderConfig`].
    const DEFAULT_EFFORT: f32 = 75.0;
    /// Default near-lossless when building a standalone [`EncoderConfig`].
    const DEFAULT_NEAR_LOSSLESS: u32 = 100;
    /// Default exact flag when building a standalone [`EncoderConfig`].
    const DEFAULT_EXACT: bool = false;

    /// Convert this node into a native [`EncoderConfig`] for zenwebp.
    ///
    /// Effort 0-100 maps to:
    /// - VP8L quality = effort (direct passthrough)
    /// - VP8L method = round(effort / 100 * 6), clamped to 0-6
    ///
    /// Values are clamped to valid ranges before conversion.
    #[must_use]
    pub fn to_encoder_config(&self) -> EncoderConfig {
        let effort = self
            .effort
            .unwrap_or(Self::DEFAULT_EFFORT)
            .clamp(0.0, 100.0);
        let near_lossless = self
            .near_lossless
            .unwrap_or(Self::DEFAULT_NEAR_LOSSLESS)
            .min(100);
        let exact = self.exact.unwrap_or(Self::DEFAULT_EXACT);

        let method = effort_to_method(effort);

        let mut cfg = LosslessConfig::new()
            .with_quality(effort)
            .with_method(method)
            .with_near_lossless(near_lossless as u8)
            .with_exact(exact);

        if let Some(aq) = self.alpha_quality {
            cfg = cfg.with_alpha_quality(aq.min(100) as u8);
        }
        if let Some(v) = self.target_size {
            cfg = cfg.with_target_size(v);
        }

        EncoderConfig::Lossless(cfg)
    }
}

#[cfg(feature = "zencodec")]
impl EncodeWebpLossless {
    /// Apply this node's explicitly-set params on top of an existing
    /// [`WebpEncoderConfig`](crate::WebpEncoderConfig).
    ///
    /// `None` fields are skipped so that the base config's values are preserved.
    pub fn apply(&self, mut config: crate::WebpEncoderConfig) -> crate::WebpEncoderConfig {
        if let Some(effort) = self.effort {
            let effort = effort.clamp(0.0, 100.0);
            let method = effort_to_method(effort);
            config = config.with_method(method);
            config = config.with_quality(effort);
        }
        if let Some(nl) = self.near_lossless {
            config = config.with_near_lossless(nl.min(100) as u8);
        }
        if let Some(exact) = self.exact {
            config = config.with_exact(exact);
        }
        if let Some(aq) = self.alpha_quality {
            config = config.with_alpha_quality_value(aq.min(100) as f32);
        }
        if let Some(v) = self.target_size {
            config = config.with_target_size(v);
        }

        config
    }

    /// Build a [`WebpEncoderConfig`](crate::WebpEncoderConfig) from scratch
    /// using this node's params.
    #[must_use]
    pub fn to_webp_encoder_config(&self) -> crate::WebpEncoderConfig {
        self.apply(crate::WebpEncoderConfig::lossless())
    }
}

// ── Decode ──────────────────────────────────────────────────────────────────

/// WebP decode node for pipeline use.
///
/// Configures the decoder's chroma upsampling and dithering behavior.
/// Maps to [`DecodeConfig`](crate::DecodeConfig).
///
/// All parameters are `Option<T>` — `None` means "use default."
#[derive(Node, Clone, Debug, Default)]
#[node(id = "zenwebp.decode", group = Decode, role = Decode)]
#[node(tags("webp", "decode"))]
pub struct DecodeWebp {
    /// Chroma upsampling method: "bilinear" (default) or "simple".
    /// Simple is faster but may produce jagged edges.
    #[param(section = "Main", label = "Upsampling")]
    #[kv("webp.upsampling")]
    pub upsampling: Option<String>,

    /// Chroma dithering strength for lossy decoding (0 = off, 100 = max).
    /// Adds noise to chroma planes to hide banding at low quality settings.
    /// Default: 50.
    #[param(range(0..=100))]
    #[param(section = "Main", label = "Dithering Strength")]
    #[kv("webp.dithering", "webp.dither")]
    pub dithering_strength: Option<u32>,
}

impl DecodeWebp {
    /// Default dithering strength.
    const DEFAULT_DITHERING_STRENGTH: u32 = 50;

    /// Convert this node into a native [`DecodeConfig`](crate::DecodeConfig).
    #[must_use]
    pub fn to_decode_config(&self) -> crate::DecodeConfig {
        let mut cfg = crate::DecodeConfig::default();

        if let Some(ref method) = self.upsampling {
            match method.to_ascii_lowercase().as_str() {
                "simple" | "nearest" => {
                    cfg = cfg.no_fancy_upsampling();
                }
                // "bilinear" or any other value keeps the default
                _ => {}
            }
        }

        let dithering = self
            .dithering_strength
            .unwrap_or(Self::DEFAULT_DITHERING_STRENGTH)
            .min(100);
        cfg = cfg.with_dithering_strength(dithering as u8);

        cfg
    }
}

#[cfg(feature = "zencodec")]
impl DecodeWebp {
    /// Apply this node's explicitly-set params on top of an existing
    /// [`WebpDecoderConfig`](crate::WebpDecoderConfig).
    ///
    /// `None` fields are skipped so that the base config's values are preserved.
    pub fn apply(&self, mut config: crate::WebpDecoderConfig) -> crate::WebpDecoderConfig {
        if let Some(ref method) = self.upsampling {
            match method.to_ascii_lowercase().as_str() {
                "simple" | "nearest" => {
                    config =
                        config.with_upsampling(crate::UpsamplingMethod::Simple);
                }
                "bilinear" | "fancy" => {
                    config =
                        config.with_upsampling(crate::UpsamplingMethod::Bilinear);
                }
                _ => {}
            }
        }
        if let Some(d) = self.dithering_strength {
            config = config.with_dithering_strength(d.min(100) as u8);
        }
        config
    }

    /// Build a [`WebpDecoderConfig`](crate::WebpDecoderConfig) from scratch
    /// using this node's params.
    #[must_use]
    pub fn to_webp_decoder_config(&self) -> crate::WebpDecoderConfig {
        self.apply(crate::WebpDecoderConfig::new())
    }
}

// ── Registration ────────────────────────────────────────────────────────────

/// Register all WebP zennode definitions with a registry.
pub fn register(registry: &mut NodeRegistry) {
    registry.register(&ENCODE_WEBP_LOSSY_NODE);
    registry.register(&ENCODE_WEBP_LOSSLESS_NODE);
    registry.register(&DECODE_WEBP_NODE);
}

/// All WebP zennode definitions.
pub static ALL: &[&dyn NodeDef] = &[
    &ENCODE_WEBP_LOSSY_NODE,
    &ENCODE_WEBP_LOSSLESS_NODE,
    &DECODE_WEBP_NODE,
];

#[cfg(test)]
mod tests {
    use super::*;

    // ── effort_to_method unit tests ──

    #[test]
    fn effort_to_method_boundary_values() {
        assert_eq!(effort_to_method(0.0), 0);
        assert_eq!(effort_to_method(8.0), 0); // 8/100*6 = 0.48, rounds to 0
        assert_eq!(effort_to_method(9.0), 1); // 9/100*6 = 0.54, rounds to 1
        assert_eq!(effort_to_method(17.0), 1);
        assert_eq!(effort_to_method(25.0), 2); // 25/100*6 = 1.5, rounds to 2
        assert_eq!(effort_to_method(50.0), 3);
        assert_eq!(effort_to_method(67.0), 4);
        assert_eq!(effort_to_method(75.0), 5); // 75/100*6 = 4.5, rounds to 5
        assert_eq!(effort_to_method(83.0), 5);
        assert_eq!(effort_to_method(92.0), 6); // 92/100*6 = 5.52, rounds to 6
        assert_eq!(effort_to_method(100.0), 6);
    }

    #[test]
    fn effort_to_method_clamping() {
        assert_eq!(effort_to_method(-10.0), 0);
        assert_eq!(effort_to_method(200.0), 6);
    }

    // ── EncodeWebpLossy schema / param tests ──

    #[test]
    fn lossy_schema_metadata() {
        let schema = ENCODE_WEBP_LOSSY_NODE.schema();
        assert_eq!(schema.id, "zenwebp.encode_lossy");
        assert_eq!(schema.group, NodeGroup::Encode);
        assert_eq!(schema.role, NodeRole::Encode);
        assert!(schema.tags.contains(&"webp"));
        assert!(schema.tags.contains(&"lossy"));
        assert!(schema.tags.contains(&"encode"));
    }

    #[test]
    fn lossy_param_names() {
        let schema = ENCODE_WEBP_LOSSY_NODE.schema();
        let names: Vec<&str> = schema.params.iter().map(|p| p.name).collect();
        assert!(names.contains(&"quality"), "missing quality");
        assert!(names.contains(&"effort"), "missing effort");
        assert!(names.contains(&"preset"), "missing preset");
        assert!(names.contains(&"sharp_yuv"), "missing sharp_yuv");
        assert!(names.contains(&"alpha_quality"), "missing alpha_quality");
        assert!(names.contains(&"target_size"), "missing target_size");
        assert!(names.contains(&"target_psnr"), "missing target_psnr");
        assert!(names.contains(&"segments"), "missing segments");
        assert!(names.contains(&"sns_strength"), "missing sns_strength");
        assert!(
            names.contains(&"filter_strength"),
            "missing filter_strength"
        );
        assert!(
            names.contains(&"filter_sharpness"),
            "missing filter_sharpness"
        );
        assert_eq!(names.len(), 11);
    }

    #[test]
    fn lossy_params_are_optional() {
        let schema = ENCODE_WEBP_LOSSY_NODE.schema();
        for param in schema.params {
            assert!(param.optional, "param {} should be optional", param.name);
        }
    }

    #[test]
    fn lossy_defaults() {
        let node = ENCODE_WEBP_LOSSY_NODE.create_default().unwrap();
        assert_eq!(node.get_param("quality"), Some(ParamValue::None));
        assert_eq!(node.get_param("effort"), Some(ParamValue::None));
        assert_eq!(node.get_param("preset"), Some(ParamValue::None));
        assert_eq!(node.get_param("sharp_yuv"), Some(ParamValue::None));
        assert_eq!(node.get_param("alpha_quality"), Some(ParamValue::None));
        assert_eq!(node.get_param("target_size"), Some(ParamValue::None));
        assert_eq!(node.get_param("target_psnr"), Some(ParamValue::None));
        assert_eq!(node.get_param("segments"), Some(ParamValue::None));
        assert_eq!(node.get_param("sns_strength"), Some(ParamValue::None));
        assert_eq!(node.get_param("filter_strength"), Some(ParamValue::None));
        assert_eq!(node.get_param("filter_sharpness"), Some(ParamValue::None));
    }

    #[test]
    fn lossy_from_kv_quality() {
        let mut kv = KvPairs::from_querystring("webp.quality=90&webp.sharp_yuv=false");
        let node = ENCODE_WEBP_LOSSY_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("quality"), Some(ParamValue::F32(90.0)));
        assert_eq!(node.get_param("sharp_yuv"), Some(ParamValue::Bool(false)));
        assert_eq!(kv.unconsumed().count(), 0);
    }

    #[test]
    fn lossy_from_kv_shorthand() {
        let mut kv = KvPairs::from_querystring("webp.q=85");
        let node = ENCODE_WEBP_LOSSY_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("quality"), Some(ParamValue::F32(85.0)));
    }

    #[test]
    fn lossy_from_kv_effort() {
        let mut kv = KvPairs::from_querystring("webp.effort=75");
        let node = ENCODE_WEBP_LOSSY_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("effort"), Some(ParamValue::F32(75.0)));
    }

    #[test]
    fn lossy_from_kv_preset() {
        let mut kv = KvPairs::from_querystring("webp.preset=photo");
        let node = ENCODE_WEBP_LOSSY_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(
            node.get_param("preset"),
            Some(ParamValue::Str("photo".into()))
        );
    }

    #[test]
    fn lossy_from_kv_advanced() {
        let mut kv =
            KvPairs::from_querystring("webp.sns=50&webp.filter=30&webp.sharpness=3");
        let node = ENCODE_WEBP_LOSSY_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("sns_strength"), Some(ParamValue::U32(50)));
        assert_eq!(node.get_param("filter_strength"), Some(ParamValue::U32(30)));
        assert_eq!(node.get_param("filter_sharpness"), Some(ParamValue::U32(3)));
        assert_eq!(kv.unconsumed().count(), 0);
    }

    #[test]
    fn lossy_from_kv_no_match() {
        let mut kv = KvPairs::from_querystring("w=800&h=600");
        let result = ENCODE_WEBP_LOSSY_NODE.from_kv(&mut kv).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn lossy_json_round_trip() {
        let mut params = ParamMap::new();
        params.insert("quality".into(), ParamValue::F32(92.0));
        params.insert("effort".into(), ParamValue::F32(75.0));
        params.insert("sharp_yuv".into(), ParamValue::Bool(false));
        params.insert("alpha_quality".into(), ParamValue::U32(80));
        params.insert("sns_strength".into(), ParamValue::U32(50));
        params.insert("preset".into(), ParamValue::Str("photo".into()));
        params.insert("target_size".into(), ParamValue::U32(50000));
        params.insert("segments".into(), ParamValue::U32(3));

        let node = ENCODE_WEBP_LOSSY_NODE.create(&params).unwrap();
        assert_eq!(node.get_param("quality"), Some(ParamValue::F32(92.0)));
        assert_eq!(node.get_param("effort"), Some(ParamValue::F32(75.0)));
        assert_eq!(node.get_param("sharp_yuv"), Some(ParamValue::Bool(false)));
        assert_eq!(node.get_param("alpha_quality"), Some(ParamValue::U32(80)));
        assert_eq!(node.get_param("sns_strength"), Some(ParamValue::U32(50)));
        assert_eq!(
            node.get_param("preset"),
            Some(ParamValue::Str("photo".into()))
        );
        assert_eq!(
            node.get_param("target_size"),
            Some(ParamValue::U32(50000))
        );
        assert_eq!(node.get_param("segments"), Some(ParamValue::U32(3)));

        // Round-trip
        let exported = node.to_params();
        let node2 = ENCODE_WEBP_LOSSY_NODE.create(&exported).unwrap();
        assert_eq!(node2.get_param("quality"), Some(ParamValue::F32(92.0)));
        assert_eq!(node2.get_param("effort"), Some(ParamValue::F32(75.0)));
    }

    #[test]
    fn lossy_downcast_to_concrete() {
        let node = ENCODE_WEBP_LOSSY_NODE.create_default().unwrap();
        let enc = node.as_any().downcast_ref::<EncodeWebpLossy>().unwrap();
        assert_eq!(enc.quality, None);
        assert_eq!(enc.effort, None);
        assert_eq!(enc.sharp_yuv, None);
        assert_eq!(enc.alpha_quality, None);
        assert_eq!(enc.sns_strength, None);
        assert_eq!(enc.preset, None);
        assert_eq!(enc.target_size, None);
        assert_eq!(enc.segments, None);
    }

    #[test]
    fn lossy_to_encoder_config_defaults() {
        let node = EncodeWebpLossy::default();
        let config = node.to_encoder_config();
        match config {
            EncoderConfig::Lossy(cfg) => {
                assert!(
                    (cfg.quality - EncodeWebpLossy::DEFAULT_QUALITY).abs() < f32::EPSILON
                );
                assert_eq!(cfg.sharp_yuv, EncodeWebpLossy::DEFAULT_SHARP_YUV);
                assert!(!cfg.sharp_yuv, "default sharp_yuv should be false");
                // effort 70 -> method round(70/100*6) = round(4.2) = 4
                assert_eq!(cfg.method, 4);
            }
            _ => panic!("expected Lossy config"),
        }
    }

    #[test]
    fn lossy_to_encoder_config_none_skips() {
        let node = EncodeWebpLossy::default();
        let config = node.to_encoder_config();
        match config {
            EncoderConfig::Lossy(cfg) => {
                assert_eq!(cfg.sns_strength, LossyConfig::new().sns_strength);
                assert_eq!(cfg.preset, None);
                assert_eq!(cfg.target_size, 0);
                assert_eq!(cfg.segments, None);
            }
            _ => panic!("expected Lossy config"),
        }
    }

    #[test]
    fn lossy_to_encoder_config_custom() {
        let node = EncodeWebpLossy {
            quality: Some(90.0),
            effort: Some(100.0),
            preset: Some("photo".into()),
            sharp_yuv: Some(false),
            alpha_quality: Some(80),
            target_size: Some(50000),
            target_psnr: Some(40.0),
            segments: Some(3),
            sns_strength: Some(50),
            filter_strength: Some(30),
            filter_sharpness: Some(3),
        };
        let config = node.to_encoder_config();
        match config {
            EncoderConfig::Lossy(cfg) => {
                assert!((cfg.quality - 90.0).abs() < f32::EPSILON);
                // effort 100 -> method 6
                assert_eq!(cfg.method, 6);
                assert!(!cfg.sharp_yuv);
                assert_eq!(cfg.alpha_quality, 80);
                assert_eq!(cfg.preset, Some(crate::Preset::Photo));
                assert_eq!(cfg.target_size, 50000);
                assert!((cfg.target_psnr - 40.0).abs() < f32::EPSILON);
                assert_eq!(cfg.segments, Some(3));
                assert_eq!(cfg.sns_strength, Some(50));
                assert_eq!(cfg.filter_strength, Some(30));
                assert_eq!(cfg.filter_sharpness, Some(3));
            }
            _ => panic!("expected Lossy config"),
        }
    }

    #[test]
    fn lossy_to_encoder_config_effort_zero() {
        let node = EncodeWebpLossy {
            effort: Some(0.0),
            ..EncodeWebpLossy::default()
        };
        let config = node.to_encoder_config();
        match config {
            EncoderConfig::Lossy(cfg) => {
                assert_eq!(cfg.method, 0);
            }
            _ => panic!("expected Lossy config"),
        }
    }

    #[test]
    fn lossy_to_encoder_config_clamps_out_of_range() {
        let node = EncodeWebpLossy {
            quality: Some(200.0),
            effort: Some(999.0),
            alpha_quality: Some(999),
            segments: Some(99),
            filter_sharpness: Some(99),
            ..EncodeWebpLossy::default()
        };
        let config = node.to_encoder_config();
        match config {
            EncoderConfig::Lossy(cfg) => {
                assert!((cfg.quality - 100.0).abs() < f32::EPSILON);
                assert_eq!(cfg.method, 6);
                assert_eq!(cfg.alpha_quality, 100);
                assert_eq!(cfg.segments, Some(4));
                assert_eq!(cfg.filter_sharpness, Some(7));
            }
            _ => panic!("expected Lossy config"),
        }
    }

    // ── EncodeWebpLossless schema / param tests ──

    #[test]
    fn lossless_schema_metadata() {
        let schema = ENCODE_WEBP_LOSSLESS_NODE.schema();
        assert_eq!(schema.id, "zenwebp.encode_lossless");
        assert_eq!(schema.group, NodeGroup::Encode);
        assert_eq!(schema.role, NodeRole::Encode);
        assert!(schema.tags.contains(&"webp"));
        assert!(schema.tags.contains(&"lossless"));
    }

    #[test]
    fn lossless_param_names() {
        let schema = ENCODE_WEBP_LOSSLESS_NODE.schema();
        let names: Vec<&str> = schema.params.iter().map(|p| p.name).collect();
        assert!(names.contains(&"effort"), "missing effort");
        assert!(names.contains(&"near_lossless"), "missing near_lossless");
        assert!(names.contains(&"exact"), "missing exact");
        assert!(names.contains(&"alpha_quality"), "missing alpha_quality");
        assert!(names.contains(&"target_size"), "missing target_size");
        assert_eq!(names.len(), 5);
    }

    #[test]
    fn lossless_params_are_optional() {
        let schema = ENCODE_WEBP_LOSSLESS_NODE.schema();
        for param in schema.params {
            assert!(param.optional, "param {} should be optional", param.name);
        }
    }

    #[test]
    fn lossless_defaults() {
        let node = ENCODE_WEBP_LOSSLESS_NODE.create_default().unwrap();
        assert_eq!(node.get_param("effort"), Some(ParamValue::None));
        assert_eq!(node.get_param("near_lossless"), Some(ParamValue::None));
        assert_eq!(node.get_param("exact"), Some(ParamValue::None));
        assert_eq!(node.get_param("alpha_quality"), Some(ParamValue::None));
        assert_eq!(node.get_param("target_size"), Some(ParamValue::None));
    }

    #[test]
    fn lossless_from_kv_effort() {
        let mut kv = KvPairs::from_querystring("webp.effort=50");
        let node = ENCODE_WEBP_LOSSLESS_NODE
            .from_kv(&mut kv)
            .unwrap()
            .unwrap();
        assert_eq!(node.get_param("effort"), Some(ParamValue::F32(50.0)));
        assert_eq!(kv.unconsumed().count(), 0);
    }

    #[test]
    fn lossless_from_kv_near_lossless() {
        let mut kv = KvPairs::from_querystring("webp.nl=60");
        let node = ENCODE_WEBP_LOSSLESS_NODE
            .from_kv(&mut kv)
            .unwrap()
            .unwrap();
        assert_eq!(node.get_param("near_lossless"), Some(ParamValue::U32(60)));
    }

    #[test]
    fn lossless_from_kv_no_match() {
        let mut kv = KvPairs::from_querystring("jpeg.quality=85");
        let result = ENCODE_WEBP_LOSSLESS_NODE.from_kv(&mut kv).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn lossless_json_round_trip() {
        let mut params = ParamMap::new();
        params.insert("effort".into(), ParamValue::F32(50.0));
        params.insert("near_lossless".into(), ParamValue::U32(80));
        params.insert("exact".into(), ParamValue::Bool(true));
        params.insert("alpha_quality".into(), ParamValue::U32(90));
        params.insert("target_size".into(), ParamValue::U32(100000));

        let node = ENCODE_WEBP_LOSSLESS_NODE.create(&params).unwrap();
        assert_eq!(node.get_param("effort"), Some(ParamValue::F32(50.0)));
        assert_eq!(node.get_param("near_lossless"), Some(ParamValue::U32(80)));
        assert_eq!(node.get_param("exact"), Some(ParamValue::Bool(true)));
        assert_eq!(
            node.get_param("alpha_quality"),
            Some(ParamValue::U32(90))
        );
        assert_eq!(
            node.get_param("target_size"),
            Some(ParamValue::U32(100000))
        );

        let exported = node.to_params();
        let node2 = ENCODE_WEBP_LOSSLESS_NODE.create(&exported).unwrap();
        assert_eq!(node2.get_param("effort"), Some(ParamValue::F32(50.0)));
    }

    #[test]
    fn lossless_downcast_to_concrete() {
        let node = ENCODE_WEBP_LOSSLESS_NODE.create_default().unwrap();
        let enc = node.as_any().downcast_ref::<EncodeWebpLossless>().unwrap();
        assert_eq!(enc.effort, None);
        assert_eq!(enc.near_lossless, None);
        assert_eq!(enc.exact, None);
        assert_eq!(enc.alpha_quality, None);
        assert_eq!(enc.target_size, None);
    }

    #[test]
    fn lossless_to_encoder_config_defaults() {
        let node = EncodeWebpLossless::default();
        let config = node.to_encoder_config();
        match config {
            EncoderConfig::Lossless(cfg) => {
                // effort 75 -> method round(75/100*6) = round(4.5) = 5
                assert_eq!(cfg.method, 5);
                assert!(
                    (cfg.quality - EncodeWebpLossless::DEFAULT_EFFORT).abs() < f32::EPSILON
                );
                assert_eq!(
                    cfg.near_lossless,
                    EncodeWebpLossless::DEFAULT_NEAR_LOSSLESS as u8
                );
                assert_eq!(cfg.exact, EncodeWebpLossless::DEFAULT_EXACT);
            }
            _ => panic!("expected Lossless config"),
        }
    }

    #[test]
    fn lossless_to_encoder_config_custom() {
        let node = EncodeWebpLossless {
            effort: Some(100.0),
            near_lossless: Some(60),
            exact: Some(true),
            alpha_quality: Some(90),
            target_size: Some(100000),
        };
        let config = node.to_encoder_config();
        match config {
            EncoderConfig::Lossless(cfg) => {
                // effort 100 -> method round(100/100*6) = 6
                assert_eq!(cfg.method, 6);
                assert!((cfg.quality - 100.0).abs() < f32::EPSILON);
                assert_eq!(cfg.near_lossless, 60);
                assert!(cfg.exact);
                assert_eq!(cfg.alpha_quality, 90);
                assert_eq!(cfg.target_size, 100000);
            }
            _ => panic!("expected Lossless config"),
        }
    }

    #[test]
    fn lossless_to_encoder_config_effort_zero() {
        let node = EncodeWebpLossless {
            effort: Some(0.0),
            ..EncodeWebpLossless::default()
        };
        let config = node.to_encoder_config();
        match config {
            EncoderConfig::Lossless(cfg) => {
                assert_eq!(cfg.method, 0);
                assert!((cfg.quality - 0.0).abs() < f32::EPSILON);
            }
            _ => panic!("expected Lossless config"),
        }
    }

    // ── DecodeWebp schema / param tests ──

    #[test]
    fn decode_schema_metadata() {
        let schema = DECODE_WEBP_NODE.schema();
        assert_eq!(schema.id, "zenwebp.decode");
        assert_eq!(schema.group, NodeGroup::Decode);
        assert_eq!(schema.role, NodeRole::Decode);
        assert!(schema.tags.contains(&"webp"));
        assert!(schema.tags.contains(&"decode"));
    }

    #[test]
    fn decode_param_names() {
        let schema = DECODE_WEBP_NODE.schema();
        let names: Vec<&str> = schema.params.iter().map(|p| p.name).collect();
        assert!(names.contains(&"upsampling"), "missing upsampling");
        assert!(
            names.contains(&"dithering_strength"),
            "missing dithering_strength"
        );
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn decode_params_are_optional() {
        let schema = DECODE_WEBP_NODE.schema();
        for param in schema.params {
            assert!(param.optional, "param {} should be optional", param.name);
        }
    }

    #[test]
    fn decode_defaults() {
        let node = DECODE_WEBP_NODE.create_default().unwrap();
        assert_eq!(node.get_param("upsampling"), Some(ParamValue::None));
        assert_eq!(
            node.get_param("dithering_strength"),
            Some(ParamValue::None)
        );
    }

    #[test]
    fn decode_from_kv() {
        let mut kv = KvPairs::from_querystring("webp.upsampling=simple&webp.dithering=0");
        let node = DECODE_WEBP_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(
            node.get_param("upsampling"),
            Some(ParamValue::Str("simple".into()))
        );
        assert_eq!(
            node.get_param("dithering_strength"),
            Some(ParamValue::U32(0))
        );
        assert_eq!(kv.unconsumed().count(), 0);
    }

    #[test]
    fn decode_from_kv_shorthand() {
        let mut kv = KvPairs::from_querystring("webp.dither=25");
        let node = DECODE_WEBP_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(
            node.get_param("dithering_strength"),
            Some(ParamValue::U32(25))
        );
    }

    #[test]
    fn decode_from_kv_no_match() {
        let mut kv = KvPairs::from_querystring("jpeg.quality=85");
        let result = DECODE_WEBP_NODE.from_kv(&mut kv).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn decode_to_config_defaults() {
        let node = DecodeWebp::default();
        let config = node.to_decode_config();
        assert_eq!(
            config.upsampling,
            crate::UpsamplingMethod::Bilinear
        );
        assert_eq!(config.dithering_strength, 50);
    }

    #[test]
    fn decode_to_config_simple() {
        let node = DecodeWebp {
            upsampling: Some("simple".into()),
            dithering_strength: Some(0),
        };
        let config = node.to_decode_config();
        assert_eq!(config.upsampling, crate::UpsamplingMethod::Simple);
        assert_eq!(config.dithering_strength, 0);
    }

    #[test]
    fn decode_to_config_bilinear() {
        let node = DecodeWebp {
            upsampling: Some("bilinear".into()),
            dithering_strength: Some(100),
        };
        let config = node.to_decode_config();
        assert_eq!(
            config.upsampling,
            crate::UpsamplingMethod::Bilinear
        );
        assert_eq!(config.dithering_strength, 100);
    }

    #[test]
    fn decode_downcast_to_concrete() {
        let node = DECODE_WEBP_NODE.create_default().unwrap();
        let dec = node.as_any().downcast_ref::<DecodeWebp>().unwrap();
        assert_eq!(dec.upsampling, None);
        assert_eq!(dec.dithering_strength, None);
    }

    // ── apply() / to_webp_encoder_config() tests (zencodec) ──

    #[cfg(feature = "zencodec")]
    #[test]
    fn lossy_to_webp_encoder_config_defaults() {
        let node = EncodeWebpLossy::default();
        let _config = node.to_webp_encoder_config();
    }

    #[cfg(feature = "zencodec")]
    #[test]
    fn lossy_apply_preserves_existing() {
        let base = crate::WebpEncoderConfig::lossy().with_sharp_yuv(true);
        let node = EncodeWebpLossy::default();
        let config = node.apply(base);
        // sharp_yuv was set on base, node doesn't override -> preserved
        let inner = config.inner();
        if let EncoderConfig::Lossy(cfg) = inner {
            assert!(cfg.sharp_yuv);
        } else {
            panic!("expected Lossy config");
        }
    }

    #[cfg(feature = "zencodec")]
    #[test]
    fn lossy_apply_sharp_yuv_and_effort() {
        let mut node = EncodeWebpLossy::default();
        node.sharp_yuv = Some(false);
        node.effort = Some(100.0);
        let config = node.to_webp_encoder_config();
        let inner = config.inner();
        if let EncoderConfig::Lossy(cfg) = inner {
            assert!(!cfg.sharp_yuv);
            assert_eq!(cfg.method, 6);
        } else {
            panic!("expected Lossy config");
        }
    }

    #[cfg(feature = "zencodec")]
    #[test]
    fn lossy_apply_all_new_params() {
        let mut node = EncodeWebpLossy::default();
        node.target_size = Some(50000);
        node.target_psnr = Some(35.0);
        node.segments = Some(2);
        node.preset = Some("drawing".into());
        let config = node.to_webp_encoder_config();
        let inner = config.inner();
        if let EncoderConfig::Lossy(cfg) = inner {
            assert_eq!(cfg.target_size, 50000);
            assert!((cfg.target_psnr - 35.0).abs() < f32::EPSILON);
            assert_eq!(cfg.segments, Some(2));
            assert_eq!(cfg.preset, Some(crate::Preset::Drawing));
        } else {
            panic!("expected Lossy config");
        }
    }

    #[cfg(feature = "zencodec")]
    #[test]
    fn lossless_to_webp_encoder_config_defaults() {
        let node = EncodeWebpLossless::default();
        let _config = node.to_webp_encoder_config();
    }

    #[cfg(feature = "zencodec")]
    #[test]
    fn lossless_apply_preserves_existing() {
        let base = crate::WebpEncoderConfig::lossless();
        let node = EncodeWebpLossless::default();
        let _config = node.apply(base);
    }

    #[cfg(feature = "zencodec")]
    #[test]
    fn lossless_apply_effort_maps_correctly() {
        let mut node = EncodeWebpLossless::default();
        node.effort = Some(50.0);
        let config = node.to_webp_encoder_config();
        let inner = config.inner();
        if let EncoderConfig::Lossless(cfg) = inner {
            assert_eq!(cfg.method, 3); // round(50/100*6) = round(3.0) = 3
        } else {
            panic!("expected Lossless config");
        }
    }

    #[cfg(feature = "zencodec")]
    #[test]
    fn decode_to_webp_decoder_config() {
        let node = DecodeWebp {
            upsampling: Some("simple".into()),
            dithering_strength: Some(0),
        };
        let config = node.to_webp_decoder_config();
        let inner = config.inner();
        assert_eq!(inner.upsampling, crate::UpsamplingMethod::Simple);
        assert_eq!(inner.dithering_strength, 0);
    }

    #[cfg(feature = "zencodec")]
    #[test]
    fn decode_apply_preserves_existing() {
        let base = crate::WebpDecoderConfig::new();
        let node = DecodeWebp::default();
        let config = node.apply(base);
        // Defaults preserved
        let inner = config.inner();
        assert_eq!(
            inner.upsampling,
            crate::UpsamplingMethod::Bilinear
        );
    }

    // ── Registry integration ──

    #[test]
    fn registry_integration() {
        let mut registry = NodeRegistry::new();
        register(&mut registry);
        assert!(registry.get("zenwebp.encode_lossy").is_some());
        assert!(registry.get("zenwebp.encode_lossless").is_some());
        assert!(registry.get("zenwebp.decode").is_some());

        // webp.quality triggers lossy node
        let result = registry.from_querystring("webp.quality=80&webp.sharp_yuv=false");
        assert_eq!(result.instances.len(), 1);
        assert_eq!(result.instances[0].schema().id, "zenwebp.encode_lossy");

        // shorthand also works
        let result2 = registry.from_querystring("webp.q=85");
        assert_eq!(result2.instances.len(), 1);
        assert_eq!(result2.instances[0].schema().id, "zenwebp.encode_lossy");
    }

    #[test]
    fn registry_lossless_querystring() {
        let mut registry = NodeRegistry::new();
        register(&mut registry);

        let result = registry.from_querystring("webp.effort=50&webp.nl=80");
        // Both lossy and lossless share webp.effort, so both may match.
        // At minimum, lossless should be present.
        let ids: Vec<&str> = result.instances.iter().map(|n| n.schema().id).collect();
        assert!(ids.contains(&"zenwebp.encode_lossless"));
    }

    #[test]
    fn registry_decode_querystring() {
        let mut registry = NodeRegistry::new();
        register(&mut registry);

        let result = registry.from_querystring("webp.dithering=0&webp.upsampling=simple");
        let ids: Vec<&str> = result.instances.iter().map(|n| n.schema().id).collect();
        assert!(ids.contains(&"zenwebp.decode"));
    }

    #[test]
    fn all_contains_all_nodes() {
        assert_eq!(ALL.len(), 3);
        let ids: Vec<&str> = ALL.iter().map(|n| n.schema().id).collect();
        assert!(ids.contains(&"zenwebp.encode_lossy"));
        assert!(ids.contains(&"zenwebp.encode_lossless"));
        assert!(ids.contains(&"zenwebp.decode"));
    }
}
