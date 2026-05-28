//! Full decode → re-encode at calibrated quality.
//!
//! One IDCT/FDCT pair of generation loss. Strategy is "trust the
//! calibration table to pick the right output quality"; we don't run an
//! IQA loop here. Iteration is the [`crate::Budget::MaxIterations`]
//! path, handled by the router via [`run_reencode_at_q`].

use crate::api::RecompressOptions;
use crate::error::Error;
use crate::source::{SourceAnalysis, SourceKind};
use crate::target;
use zenwebp::PixelLayout;
use zenwebp::encoder::{EncodeRequest, LossyConfig};
use zenwebp::oneshot::decode_rgba;

/// Decode + re-encode at the calibrated quality (no deblock, no extras).
pub fn run_reencode(
    webp_bytes: &[u8],
    analysis: &SourceAnalysis,
    opts: &RecompressOptions,
) -> Result<Vec<u8>, Error> {
    let target_q = target::target_zensim_a_to_libwebp_q(opts.target_zensim_a);
    run_reencode_at_q(webp_bytes, analysis, target_q)
}

/// Decode + re-encode at an explicit libwebp quality. Used by the
/// iteration loop in [`Budget::MaxIterations`](crate::Budget::MaxIterations).
pub fn run_reencode_at_q(
    webp_bytes: &[u8],
    analysis: &SourceAnalysis,
    libwebp_q: u8,
) -> Result<Vec<u8>, Error> {
    if matches!(analysis.kind, SourceKind::Animated) {
        return Err(Error::AnimationNotSupported);
    }

    let (rgba, w, h) =
        decode_rgba(webp_bytes).map_err(|e| Error::DecodeFailed(format!("{e:?}")))?;

    let cfg = LossyConfig::new()
        .with_quality(libwebp_q.clamp(1, 100) as f32)
        .with_method(4)
        .with_alpha_quality(100);
    let bytes = EncodeRequest::lossy(&cfg, &rgba, PixelLayout::Rgba8, w, h)
        .encode()
        .map_err(|e| Error::EncodeFailed(format!("{e:?}")))?;
    Ok(bytes)
}
