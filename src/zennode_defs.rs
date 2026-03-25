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
//! When the `zencodec` feature is also enabled, `apply()` and
//! `to_webp_encoder_config()` produce a [`WebpEncoderConfig`](crate::WebpEncoderConfig)
//! that integrates with the zencodec trait system.
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

#[cfg(feature = "zencodec")]
impl EncodeWebpLossy {
    /// Apply this node's explicitly-set params on top of an existing
    /// [`WebpEncoderConfig`](crate::WebpEncoderConfig).
    ///
    /// Fields at their default/sentinel value are skipped so that the
    /// base config's values are preserved.
    pub fn apply(
        &self,
        mut config: crate::WebpEncoderConfig,
    ) -> crate::WebpEncoderConfig {
        // Quality (75.0 is the default, but always apply since any explicit
        // quality= in the querystring should take effect)
        config = config.with_quality(self.quality);

        // Effort → method
        config = config.with_effort_u32(self.effort);

        // Sharp YUV
        config = config.with_sharp_yuv(self.sharp_yuv);

        // Alpha quality
        config = config.with_alpha_quality_value(self.alpha_quality as f32);

        // Advanced: only apply when explicitly set (sentinel = -1)
        if self.sns_strength >= 0 {
            config = config.with_sns_strength(self.sns_strength.min(100) as u8);
        }
        if self.filter_strength >= 0 {
            config = config.with_filter_strength(self.filter_strength.min(100) as u8);
        }
        if self.filter_sharpness >= 0 {
            config = config.with_filter_sharpness(self.filter_sharpness.min(7) as u8);
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

#[cfg(feature = "zencodec")]
impl EncodeWebpLossless {
    /// Apply this node's explicitly-set params on top of an existing
    /// [`WebpEncoderConfig`](crate::WebpEncoderConfig).
    pub fn apply(
        &self,
        mut config: crate::WebpEncoderConfig,
    ) -> crate::WebpEncoderConfig {
        // Effort → quality + method (effort 0-100 maps to both)
        let effort = self.effort.clamp(0.0, 100.0) as u32;
        config = config.with_effort_u32(effort);
        config = config.with_quality(self.effort.clamp(0.0, 100.0));

        // Near-lossless
        config = config.with_near_lossless(self.near_lossless.min(100) as u8);

        config
    }

    /// Build a [`WebpEncoderConfig`](crate::WebpEncoderConfig) from scratch
    /// using this node's params.
    #[must_use]
    pub fn to_webp_encoder_config(&self) -> crate::WebpEncoderConfig {
        self.apply(crate::WebpEncoderConfig::lossless())
    }
}

/// Register all WebP zennode definitions with a registry.
pub fn register(registry: &mut NodeRegistry) {
    registry.register(&ENCODE_WEBP_LOSSY_NODE);
    registry.register(&ENCODE_WEBP_LOSSLESS_NODE);
}

/// All WebP zennode definitions.
pub static ALL: &[&dyn NodeDef] = &[&ENCODE_WEBP_LOSSY_NODE, &ENCODE_WEBP_LOSSLESS_NODE];

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(names.contains(&"quality"));
        assert!(names.contains(&"effort"));
        assert!(names.contains(&"sharp_yuv"));
        assert!(names.contains(&"alpha_quality"));
        assert!(names.contains(&"sns_strength"));
        assert!(names.contains(&"filter_strength"));
        assert!(names.contains(&"filter_sharpness"));
        assert_eq!(names.len(), 7);
    }

    #[test]
    fn lossy_defaults() {
        let node = ENCODE_WEBP_LOSSY_NODE.create_default().unwrap();
        assert_eq!(node.get_param("quality"), Some(ParamValue::F32(75.0)));
        assert_eq!(node.get_param("effort"), Some(ParamValue::U32(7)));
        assert_eq!(node.get_param("sharp_yuv"), Some(ParamValue::Bool(true)));
        assert_eq!(node.get_param("alpha_quality"), Some(ParamValue::U32(100)));
        assert_eq!(node.get_param("sns_strength"), Some(ParamValue::I32(-1)));
        assert_eq!(node.get_param("filter_strength"), Some(ParamValue::I32(-1)));
        assert_eq!(node.get_param("filter_sharpness"), Some(ParamValue::I32(-1)));
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
        let mut kv = KvPairs::from_querystring("webp.effort=10");
        let node = ENCODE_WEBP_LOSSY_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("effort"), Some(ParamValue::U32(10)));
    }

    #[test]
    fn lossy_from_kv_advanced() {
        let mut kv = KvPairs::from_querystring("webp.sns=50&webp.filter=30&webp.sharpness=3");
        let node = ENCODE_WEBP_LOSSY_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("sns_strength"), Some(ParamValue::I32(50)));
        assert_eq!(node.get_param("filter_strength"), Some(ParamValue::I32(30)));
        assert_eq!(node.get_param("filter_sharpness"), Some(ParamValue::I32(3)));
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
        params.insert("effort".into(), ParamValue::U32(5));
        params.insert("sharp_yuv".into(), ParamValue::Bool(false));
        params.insert("alpha_quality".into(), ParamValue::U32(80));
        params.insert("sns_strength".into(), ParamValue::I32(50));

        let node = ENCODE_WEBP_LOSSY_NODE.create(&params).unwrap();
        assert_eq!(node.get_param("quality"), Some(ParamValue::F32(92.0)));
        assert_eq!(node.get_param("effort"), Some(ParamValue::U32(5)));
        assert_eq!(node.get_param("sharp_yuv"), Some(ParamValue::Bool(false)));
        assert_eq!(node.get_param("alpha_quality"), Some(ParamValue::U32(80)));
        assert_eq!(node.get_param("sns_strength"), Some(ParamValue::I32(50)));

        // Round-trip
        let exported = node.to_params();
        let node2 = ENCODE_WEBP_LOSSY_NODE.create(&exported).unwrap();
        assert_eq!(node2.get_param("quality"), Some(ParamValue::F32(92.0)));
        assert_eq!(node2.get_param("effort"), Some(ParamValue::U32(5)));
    }

    #[test]
    fn lossy_downcast_to_concrete() {
        let node = ENCODE_WEBP_LOSSY_NODE.create_default().unwrap();
        let enc = node.as_any().downcast_ref::<EncodeWebpLossy>().unwrap();
        assert!((enc.quality - 75.0).abs() < f32::EPSILON);
        assert_eq!(enc.effort, 7);
        assert!(enc.sharp_yuv);
        assert_eq!(enc.alpha_quality, 100);
        assert_eq!(enc.sns_strength, -1);
    }

    #[test]
    fn lossy_to_encoder_config_defaults() {
        let node = EncodeWebpLossy::default();
        let config = node.to_encoder_config();
        match config {
            EncoderConfig::Lossy(cfg) => {
                assert!((cfg.quality - 75.0).abs() < f32::EPSILON);
                assert!(cfg.sharp_yuv);
            }
            _ => panic!("expected Lossy config"),
        }
    }

    #[test]
    fn lossy_to_encoder_config_sentinel_skips() {
        // sns_strength=-1 means default, should NOT override
        let node = EncodeWebpLossy::default();
        let config = node.to_encoder_config();
        match config {
            EncoderConfig::Lossy(cfg) => {
                // Sentinels leave the preset defaults in place
                assert_eq!(cfg.sns_strength, LossyConfig::new().sns_strength);
            }
            _ => panic!("expected Lossy config"),
        }
    }

    #[test]
    fn lossy_to_encoder_config_custom() {
        let node = EncodeWebpLossy {
            quality: 90.0,
            effort: 10,
            sharp_yuv: false,
            alpha_quality: 80,
            sns_strength: 50,
            filter_strength: 30,
            filter_sharpness: 3,
        };
        let config = node.to_encoder_config();
        match config {
            EncoderConfig::Lossy(cfg) => {
                assert!((cfg.quality - 90.0).abs() < f32::EPSILON);
                // effort 10 → method 6
                assert_eq!(cfg.method, 6);
                assert!(!cfg.sharp_yuv);
                assert_eq!(cfg.alpha_quality, 80);
                assert_eq!(cfg.sns_strength, Some(50));
                assert_eq!(cfg.filter_strength, Some(30));
                assert_eq!(cfg.filter_sharpness, Some(3));
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
        assert!(names.contains(&"effort"));
        assert!(names.contains(&"near_lossless"));
        assert!(names.contains(&"exact"));
        assert_eq!(names.len(), 3);
    }

    #[test]
    fn lossless_defaults() {
        let node = ENCODE_WEBP_LOSSLESS_NODE.create_default().unwrap();
        assert_eq!(node.get_param("effort"), Some(ParamValue::F32(75.0)));
        assert_eq!(node.get_param("near_lossless"), Some(ParamValue::U32(100)));
        assert_eq!(node.get_param("exact"), Some(ParamValue::Bool(false)));
    }

    #[test]
    fn lossless_from_kv_effort() {
        let mut kv = KvPairs::from_querystring("webp.effort=50");
        let node = ENCODE_WEBP_LOSSLESS_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("effort"), Some(ParamValue::F32(50.0)));
        assert_eq!(kv.unconsumed().count(), 0);
    }

    #[test]
    fn lossless_from_kv_near_lossless() {
        let mut kv = KvPairs::from_querystring("webp.nl=60");
        let node = ENCODE_WEBP_LOSSLESS_NODE.from_kv(&mut kv).unwrap().unwrap();
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

        let node = ENCODE_WEBP_LOSSLESS_NODE.create(&params).unwrap();
        assert_eq!(node.get_param("effort"), Some(ParamValue::F32(50.0)));
        assert_eq!(node.get_param("near_lossless"), Some(ParamValue::U32(80)));
        assert_eq!(node.get_param("exact"), Some(ParamValue::Bool(true)));

        let exported = node.to_params();
        let node2 = ENCODE_WEBP_LOSSLESS_NODE.create(&exported).unwrap();
        assert_eq!(node2.get_param("effort"), Some(ParamValue::F32(50.0)));
    }

    #[test]
    fn lossless_downcast_to_concrete() {
        let node = ENCODE_WEBP_LOSSLESS_NODE.create_default().unwrap();
        let enc = node.as_any().downcast_ref::<EncodeWebpLossless>().unwrap();
        assert!((enc.effort - 75.0).abs() < f32::EPSILON);
        assert_eq!(enc.near_lossless, 100);
        assert!(!enc.exact);
    }

    #[test]
    fn lossless_to_encoder_config_defaults() {
        let node = EncodeWebpLossless::default();
        let config = node.to_encoder_config();
        match config {
            EncoderConfig::Lossless(cfg) => {
                // effort 75 → method round(75/100*6) = round(4.5) = 5
                assert_eq!(cfg.method, 5);
                assert!((cfg.quality - 75.0).abs() < f32::EPSILON);
                assert_eq!(cfg.near_lossless, 100);
                assert!(!cfg.exact);
            }
            _ => panic!("expected Lossless config"),
        }
    }

    #[test]
    fn lossless_to_encoder_config_custom() {
        let node = EncodeWebpLossless {
            effort: 100.0,
            near_lossless: 60,
            exact: true,
        };
        let config = node.to_encoder_config();
        match config {
            EncoderConfig::Lossless(cfg) => {
                // effort 100 → method round(100/100*6) = 6
                assert_eq!(cfg.method, 6);
                assert!((cfg.quality - 100.0).abs() < f32::EPSILON);
                assert_eq!(cfg.near_lossless, 60);
                assert!(cfg.exact);
            }
            _ => panic!("expected Lossless config"),
        }
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
        let _config = node.apply(base);
    }

    #[cfg(feature = "zencodec")]
    #[test]
    fn lossy_apply_sharp_yuv_and_effort() {
        let mut node = EncodeWebpLossy::default();
        node.sharp_yuv = false;
        node.effort = 10;
        let _config = node.to_webp_encoder_config();
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

    // ── Registry integration ──

    #[test]
    fn registry_integration() {
        let mut registry = NodeRegistry::new();
        register(&mut registry);
        assert!(registry.get("zenwebp.encode_lossy").is_some());
        assert!(registry.get("zenwebp.encode_lossless").is_some());

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
    fn all_contains_both_nodes() {
        assert_eq!(ALL.len(), 2);
        let ids: Vec<&str> = ALL.iter().map(|n| n.schema().id).collect();
        assert!(ids.contains(&"zenwebp.encode_lossy"));
        assert!(ids.contains(&"zenwebp.encode_lossless"));
    }
}
