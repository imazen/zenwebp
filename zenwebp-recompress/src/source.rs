//! Source analysis: classify the input WebP and extract knobs the router
//! needs to consult the calibration table.

use crate::error::Error;
use zenwebp::detect::{BitstreamType, WebPProbe, probe};

/// Encoder family identification. Two families today (libwebp vs zenwebp);
/// we add fingerprints as we see them in the wild.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(dead_code)] // Zenwebp variant returned when fingerprint classifier ships.
pub enum EncoderFamily {
    /// libwebp (cwebp, Sharp, Pillow, ImageMagick, NodeJS @squoosh/webp, etc.)
    /// — all share libwebp's RD path.
    Libwebp,
    /// zenwebp's target-zensim or quality-dial encoder.
    Zenwebp,
    /// Unrecognized — fall back to the libwebp table with a risk margin.
    Other,
}

/// Coarse content classification. The default router uses this; the
/// MLP-refined router (under `analyzer` feature) replaces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(dead_code)] // Photo/Screen/LineArt consumed when analyzer feature lands.
pub enum ContentClass {
    Photo,
    Screen,
    LineArt,
    Mixed,
}

/// Source kind — distilled from the WebP probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SourceKind {
    /// VP8 lossy.
    LossyVp8,
    /// VP8L lossless.
    LosslessVp8L,
    /// Animated container (any frame kind).
    Animated,
}

/// Output of [`analyze_source`].
#[derive(Debug, Clone)]
#[non_exhaustive]
#[allow(dead_code)] // VP8 fingerprint fields consumed by CoeffEdit when it ships.
pub struct SourceAnalysis {
    /// Pixel dimensions.
    pub width: u32,
    /// Pixel dimensions.
    pub height: u32,
    /// Lossy / lossless / animated.
    pub kind: SourceKind,
    /// **Header-only** quality estimate in `[1.0, 100.0]` from the VP8 base
    /// quantizer. UNRELIABLE for segmented WebP (see
    /// `docs/QUALITY_DETECTION.md`) — kept for reporting / `plan()` only.
    /// The router uses [`Self::estimated_quality`] instead.
    pub source_q: f32,
    /// Decode-based effective quality estimate in `[1.0, 100.0]`, from
    /// recompression self-consistency ([`crate::estimate`]). Equal to
    /// [`Self::source_q`] until [`refine_from_decode`] runs (on the
    /// `recompress()` path); `plan()` leaves it at the header estimate.
    /// This is the reliable calibration key.
    pub estimated_quality: f32,
    /// Lossy-only: VP8 base quantizer index `[0, 127]`. `0` for lossless.
    pub vp8_quantizer_index: u8,
    /// Lossy-only: VP8 loop filter level. `0` for lossless.
    pub vp8_filter_level: u8,
    /// Lossy-only: VP8 sharpness level. `0` for lossless.
    pub vp8_sharpness_level: u8,
    /// Whether per-segment Q overrides are present (lossy only).
    pub has_segment_quant: bool,
    /// Whether the file has an alpha channel.
    pub has_alpha: bool,
    /// Whether the file has an ICCP color profile chunk.
    pub has_icc: bool,
    /// Best-effort encoder family.
    pub encoder_family: EncoderFamily,
    /// Best-effort content classification. Default is `Mixed`; refined by
    /// `analyzer` feature.
    pub content_class: ContentClass,
}

impl SourceAnalysis {
    /// `true` if recompression as lossy is *categorically* unsafe — e.g. the
    /// input is animated. The router falls back to `LosslessRemux` in that
    /// case.
    pub fn lossy_recompression_unsafe(&self) -> bool {
        matches!(self.kind, SourceKind::Animated)
    }
}

/// Probe the WebP, infer encoder family, and classify content into a coarse
/// bucket.
///
/// Returns [`Error::InvalidInput`] if the bytes are not parseable as WebP.
pub fn analyze_source(webp_bytes: &[u8]) -> Result<SourceAnalysis, Error> {
    let info = probe(webp_bytes)?;
    Ok(synthesize(info))
}

fn synthesize(info: WebPProbe) -> SourceAnalysis {
    let (kind, source_q, qi, fl, sl, segs) = match info.bitstream {
        BitstreamType::Lossy {
            quality_estimate,
            quantizer_index,
            has_segment_quant,
            filter_level,
            sharpness_level,
        } => {
            let kind = if info.has_animation {
                SourceKind::Animated
            } else {
                SourceKind::LossyVp8
            };
            (
                kind,
                quality_estimate.clamp(1.0, 100.0),
                quantizer_index,
                filter_level,
                sharpness_level,
                has_segment_quant,
            )
        }
        BitstreamType::Lossless => {
            let kind = if info.has_animation {
                SourceKind::Animated
            } else {
                SourceKind::LosslessVp8L
            };
            (kind, 100.0, 0, 0, 0, false)
        }
    };

    SourceAnalysis {
        width: info.width,
        height: info.height,
        kind,
        source_q,
        // Until a decode refines it, the reliable estimate falls back to
        // the (unreliable) header estimate.
        estimated_quality: source_q,
        vp8_quantizer_index: qi,
        vp8_filter_level: fl,
        vp8_sharpness_level: sl,
        has_segment_quant: segs,
        has_alpha: info.has_alpha,
        has_icc: info.icc_profile.is_some(),
        encoder_family: classify_encoder(qi, fl, sl, segs),
        // Default classification — refined by `refine_from_decode`.
        content_class: ContentClass::Mixed,
    }
}

/// Decode the source ONCE and refine both `content_class` (heuristic
/// classifier) and `estimated_quality` (recompression self-consistency).
///
/// [`analyze_source`] is header-only (cheap; `content_class = Mixed`,
/// `estimated_quality = ` the unreliable header estimate). This pass
/// decodes to RGBA and replaces both with decode-derived values. Used by
/// [`crate::recompress`] (which decodes anyway); NOT by [`crate::plan`].
///
/// On decode failure the analysis is returned unchanged.
pub fn refine_from_decode(analysis: &mut SourceAnalysis, webp_bytes: &[u8]) {
    if matches!(analysis.kind, SourceKind::Animated) {
        return;
    }
    if let Ok((rgba, w, h)) = zenwebp::oneshot::decode_rgba(webp_bytes) {
        analysis.content_class = crate::classify::classify(&rgba, w as usize, h as usize);
        // Lossless sources have no lossy quality; leave estimated_quality
        // at 100 (set via source_q=100 for lossless).
        if matches!(analysis.kind, SourceKind::LossyVp8)
            && let Ok(eq) =
                crate::estimate::estimate_quality_by_recompression(&rgba, w, h, webp_bytes.len())
        {
            analysis.estimated_quality = eq;
        }
    }
}

/// Decode the source and refine `content_class` only. Retained for
/// callers / tests that want classification without the extra encodes of
/// quality estimation.
#[allow(dead_code)]
pub fn refine_content_class(analysis: &mut SourceAnalysis, webp_bytes: &[u8]) {
    if matches!(analysis.kind, SourceKind::Animated) {
        return;
    }
    if let Ok((rgba, w, h)) = zenwebp::oneshot::decode_rgba(webp_bytes) {
        analysis.content_class = crate::classify::classify(&rgba, w as usize, h as usize);
    }
}

/// Heuristic encoder fingerprint. libwebp's default `cwebp -q` path
/// produces a recognizable (quantizer, filter, sharpness, segments)
/// signature; zenwebp's target-zensim path produces another. Unrecognized
/// signatures fall through to `Other`.
fn classify_encoder(qi: u8, fl: u8, sl: u8, segs: bool) -> EncoderFamily {
    // libwebp -m 6 default: segments=on, filter=0..30 depending on q,
    // sharpness=0. cwebp -m 4 default: similar.
    if segs && sl == 0 && fl <= 60 {
        // Both libwebp and zenwebp pass this, but libwebp dominates the
        // population. Without bitstream-level fingerprinting we can't
        // distinguish further. Default to libwebp; zenwebp callers
        // override via `expert::SourceAnalysis` mutation when they know.
        return EncoderFamily::Libwebp;
    }
    if !segs && sl <= 7 && fl == 0 && qi >= 40 {
        // No segments + no filter + non-trivial quantizer is consistent
        // with hand-built bitstreams (vp8enc, FFmpeg webpenc, etc.).
        return EncoderFamily::Other;
    }
    EncoderFamily::Libwebp
}
