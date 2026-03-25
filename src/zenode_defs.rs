//! zennode node definitions for WebP encoding.
//!
//! Defines [`EncodeWebpLossy`] and [`EncodeWebpLossless`] with RIAPI-compatible
//! querystring keys for WebP encoding parameters.

use zennode::*;

/// Lossy WebP encoding with quality, compression method, and sharp YUV options.
///
/// JSON API: `{ "quality": 80, "method": 4, "sharp_yuv": true }`
/// RIAPI: `?webp.quality=80&webp.method=4&webp.sharp_yuv=true`
#[derive(Node, Clone, Debug)]
#[node(id = "zenwebp.encode_lossy", group = Encode, role = Encode)]
#[node(tags("codec", "webp", "encode", "lossy"))]
pub struct EncodeWebpLossy {
    /// Generic quality 0-100 (mapped via with_generic_quality at execution time).
    ///
    /// When set (>= 0), this value is passed through zencodec's
    /// `with_generic_quality()` which maps it to the codec's native
    /// quality scale. Use this for uniform quality across all codecs.
    #[param(range(0..=100), default = -1, step = 1)]
    #[param(unit = "", section = "Main", label = "Quality")]
    #[kv("quality")]
    pub quality: i32,

    /// Codec-specific WebP quality override (0-100).
    ///
    /// Controls the DCT quantization level. Higher values produce
    /// larger files with better visual quality.
    /// When set (>= 0), takes precedence over the generic `quality` field.
    #[param(range(0.0..=100.0), default = -1.0, identity = 75.0, step = 1.0)]
    #[param(unit = "", section = "Main", label = "WebP Quality")]
    #[kv("webp.quality")]
    pub webp_quality: f32,

    /// Compression method (0 = fast, 6 = slower but better compression).
    ///
    /// Higher values use more CPU time for better compression ratios.
    /// Method 4 is a good default balancing speed and compression.
    #[param(range(0..=6), default = 4, step = 1)]
    #[param(unit = "", section = "Main", label = "Method")]
    #[kv("webp.method")]
    pub method: i32,

    /// Use sharp (iterative) YUV conversion for better color edge quality.
    ///
    /// Reduces color bleeding at sharp edges during RGB-to-YUV conversion.
    /// Slightly slower encoding but noticeable improvement on high-contrast
    /// edges and text.
    #[param(default = false)]
    #[param(section = "Advanced", label = "Sharp YUV")]
    #[kv("webp.sharp_yuv")]
    pub sharp_yuv: bool,
}

impl Default for EncodeWebpLossy {
    fn default() -> Self {
        Self {
            quality: -1,
            webp_quality: -1.0,
            method: 4,
            sharp_yuv: false,
        }
    }
}

/// Lossless WebP encoding with compression method control.
///
/// Produces pixel-perfect output using prediction, color transforms,
/// and LZ77 compression.
///
/// JSON API: `{ "method": 4 }`
/// RIAPI: `?webp.lossless.method=4`
#[derive(Node, Clone, Debug)]
#[node(id = "zenwebp.encode_lossless", group = Encode, role = Encode)]
#[node(tags("codec", "webp", "encode", "lossless"))]
pub struct EncodeWebpLossless {
    /// Compression method (0 = fast, 6 = slowest but best compression).
    ///
    /// Higher values use more CPU time for better compression ratios.
    /// Method 4 is a good default.
    #[param(range(0..=6), default = 4, step = 1)]
    #[param(unit = "", section = "Main", label = "Method")]
    #[kv("webp.lossless.method")]
    pub method: i32,
}

impl Default for EncodeWebpLossless {
    fn default() -> Self {
        Self { method: 4 }
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

    // ── EncodeWebpLossy tests ──

    #[test]
    fn lossy_schema_metadata() {
        let schema = ENCODE_WEBP_LOSSY_NODE.schema();
        assert_eq!(schema.id, "zenwebp.encode_lossy");
        assert_eq!(schema.group, NodeGroup::Encode);
        assert_eq!(schema.role, NodeRole::Encode);
        assert!(schema.tags.contains(&"webp"));
        assert!(schema.tags.contains(&"lossy"));
        assert!(schema.tags.contains(&"codec"));
        assert!(schema.tags.contains(&"encode"));
    }

    #[test]
    fn lossy_param_names() {
        let schema = ENCODE_WEBP_LOSSY_NODE.schema();
        let names: Vec<&str> = schema.params.iter().map(|p| p.name).collect();
        assert!(names.contains(&"quality"));
        assert!(names.contains(&"webp_quality"));
        assert!(names.contains(&"method"));
        assert!(names.contains(&"sharp_yuv"));
        assert_eq!(names.len(), 4);
    }

    #[test]
    fn lossy_defaults() {
        let node = ENCODE_WEBP_LOSSY_NODE.create_default().unwrap();
        assert_eq!(node.get_param("quality"), Some(ParamValue::I32(-1)));
        assert_eq!(node.get_param("webp_quality"), Some(ParamValue::F32(-1.0)));
        assert_eq!(node.get_param("method"), Some(ParamValue::I32(4)));
        assert_eq!(node.get_param("sharp_yuv"), Some(ParamValue::Bool(false)));
    }

    #[test]
    fn lossy_from_kv_webp_quality() {
        let mut kv = KvPairs::from_querystring("webp.quality=90&webp.sharp_yuv=true");
        let node = ENCODE_WEBP_LOSSY_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("webp_quality"), Some(ParamValue::F32(90.0)));
        assert_eq!(node.get_param("sharp_yuv"), Some(ParamValue::Bool(true)));
        assert_eq!(kv.unconsumed().count(), 0);
    }

    #[test]
    fn lossy_from_kv_generic_quality() {
        let mut kv = KvPairs::from_querystring("quality=80");
        let node = ENCODE_WEBP_LOSSY_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("quality"), Some(ParamValue::I32(80)));
        // webp_quality remains unset
        assert_eq!(node.get_param("webp_quality"), Some(ParamValue::F32(-1.0)));
    }

    #[test]
    fn lossy_from_kv_both_qualities() {
        let mut kv = KvPairs::from_querystring("quality=80&webp.quality=90");
        let node = ENCODE_WEBP_LOSSY_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("quality"), Some(ParamValue::I32(80)));
        assert_eq!(node.get_param("webp_quality"), Some(ParamValue::F32(90.0)));
        assert_eq!(kv.unconsumed().count(), 0);
    }

    #[test]
    fn lossy_from_kv_method() {
        let mut kv = KvPairs::from_querystring("webp.method=6");
        let node = ENCODE_WEBP_LOSSY_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("method"), Some(ParamValue::I32(6)));
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
        params.insert("quality".into(), ParamValue::I32(80));
        params.insert("webp_quality".into(), ParamValue::F32(92.0));
        params.insert("method".into(), ParamValue::I32(5));
        params.insert("sharp_yuv".into(), ParamValue::Bool(true));

        let node = ENCODE_WEBP_LOSSY_NODE.create(&params).unwrap();
        assert_eq!(node.get_param("quality"), Some(ParamValue::I32(80)));
        assert_eq!(node.get_param("webp_quality"), Some(ParamValue::F32(92.0)));
        assert_eq!(node.get_param("method"), Some(ParamValue::I32(5)));
        assert_eq!(node.get_param("sharp_yuv"), Some(ParamValue::Bool(true)));

        // Round-trip
        let exported = node.to_params();
        let node2 = ENCODE_WEBP_LOSSY_NODE.create(&exported).unwrap();
        assert_eq!(node2.get_param("quality"), Some(ParamValue::I32(80)));
        assert_eq!(node2.get_param("webp_quality"), Some(ParamValue::F32(92.0)));
        assert_eq!(node2.get_param("method"), Some(ParamValue::I32(5)));
    }

    #[test]
    fn lossy_downcast_to_concrete() {
        let node = ENCODE_WEBP_LOSSY_NODE.create_default().unwrap();
        let enc = node.as_any().downcast_ref::<EncodeWebpLossy>().unwrap();
        assert_eq!(enc.quality, -1);
        assert!((enc.webp_quality - (-1.0)).abs() < f32::EPSILON);
        assert_eq!(enc.method, 4);
        assert!(!enc.sharp_yuv);
    }

    // ── EncodeWebpLossless tests ──

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
        assert!(names.contains(&"method"));
        assert_eq!(names.len(), 1);
    }

    #[test]
    fn lossless_defaults() {
        let node = ENCODE_WEBP_LOSSLESS_NODE.create_default().unwrap();
        assert_eq!(node.get_param("method"), Some(ParamValue::I32(4)));
    }

    #[test]
    fn lossless_from_kv_method() {
        let mut kv = KvPairs::from_querystring("webp.lossless.method=2");
        let node = ENCODE_WEBP_LOSSLESS_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("method"), Some(ParamValue::I32(2)));
        assert_eq!(kv.unconsumed().count(), 0);
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
        params.insert("method".into(), ParamValue::I32(6));

        let node = ENCODE_WEBP_LOSSLESS_NODE.create(&params).unwrap();
        assert_eq!(node.get_param("method"), Some(ParamValue::I32(6)));

        let exported = node.to_params();
        let node2 = ENCODE_WEBP_LOSSLESS_NODE.create(&exported).unwrap();
        assert_eq!(node2.get_param("method"), Some(ParamValue::I32(6)));
    }

    #[test]
    fn lossless_downcast_to_concrete() {
        let node = ENCODE_WEBP_LOSSLESS_NODE.create_default().unwrap();
        let enc = node.as_any().downcast_ref::<EncodeWebpLossless>().unwrap();
        assert_eq!(enc.method, 4);
    }

    // ── Registry integration ──

    #[test]
    fn registry_integration() {
        let mut registry = NodeRegistry::new();
        register(&mut registry);
        assert!(registry.get("zenwebp.encode_lossy").is_some());
        assert!(registry.get("zenwebp.encode_lossless").is_some());

        // webp.quality triggers codec-specific path
        let result = registry.from_querystring("webp.quality=80&webp.sharp_yuv=true");
        assert_eq!(result.instances.len(), 1);
        assert_eq!(result.instances[0].schema().id, "zenwebp.encode_lossy");

        // generic quality also triggers the lossy node
        let result2 = registry.from_querystring("quality=80");
        assert_eq!(result2.instances.len(), 1);
        assert_eq!(result2.instances[0].schema().id, "zenwebp.encode_lossy");
    }

    #[test]
    fn registry_lossless_querystring() {
        let mut registry = NodeRegistry::new();
        register(&mut registry);

        let result = registry.from_querystring("webp.lossless.method=3");
        assert_eq!(result.instances.len(), 1);
        assert_eq!(result.instances[0].schema().id, "zenwebp.encode_lossless");
    }
}
