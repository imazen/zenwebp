//! Decode → re-encode as VP8L (lossless).
//!
//! Wins for graphics, screen content, and posterized photos where source
//! quantization has already collapsed gradients into few values. Zero
//! further generation loss after the one decode.

use crate::error::Error;
use crate::source::{SourceAnalysis, SourceKind};
use zenwebp::encoder::{EncodeRequest, LosslessConfig};
use zenwebp::oneshot::decode_rgba;
use zenwebp::PixelLayout;

pub fn run_lossless_reencode(
    webp_bytes: &[u8],
    analysis: &SourceAnalysis,
) -> Result<Vec<u8>, Error> {
    if matches!(analysis.kind, SourceKind::Animated) {
        return Err(Error::AnimationNotSupported);
    }

    let (rgba, w, h) =
        decode_rgba(webp_bytes).map_err(|e| Error::DecodeFailed(format!("{e:?}")))?;

    let cfg = LosslessConfig::new().with_method(4).with_near_lossless(100);
    let bytes = EncodeRequest::lossless(&cfg, &rgba, PixelLayout::Rgba8, w, h)
        .encode()
        .map_err(|e| Error::EncodeFailed(format!("{e:?}")))?;
    Ok(bytes)
}
