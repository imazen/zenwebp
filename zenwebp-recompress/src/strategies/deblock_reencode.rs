//! Full decode → gradient-gated deblock → re-encode at calibrated Q.
//!
//! Removes block-boundary artifacts and ringing baked in by the source
//! encoder, then re-encodes. The cumulative zensim-A is improved at fixed
//! output bpp, so the router can pick a *lower* target_q than for the
//! plain `Reencode` strategy and still hit the same cumulative target.

use crate::api::RecompressOptions;
use crate::error::Error;
use crate::source::{SourceAnalysis, SourceKind};
use crate::strategies::deblock::{deblock_rgba, strength_from_quantizer};
use crate::target;
use zenwebp::PixelLayout;
use zenwebp::encoder::{EncodeRequest, LossyConfig};
use zenwebp::oneshot::decode_rgba;

pub fn run_deblock_reencode(
    webp_bytes: &[u8],
    analysis: &SourceAnalysis,
    opts: &RecompressOptions,
) -> Result<Vec<u8>, Error> {
    let target_q = target::target_zensim_a_to_libwebp_q(opts.target_zensim_a);
    run_deblock_reencode_at_q(webp_bytes, analysis, target_q)
}

pub fn run_deblock_reencode_at_q(
    webp_bytes: &[u8],
    analysis: &SourceAnalysis,
    libwebp_q: u8,
) -> Result<Vec<u8>, Error> {
    if matches!(analysis.kind, SourceKind::Animated) {
        return Err(Error::AnimationNotSupported);
    }

    let (mut rgba, w, h) =
        decode_rgba(webp_bytes).map_err(|e| Error::DecodeFailed(format!("{e:?}")))?;

    let strength = strength_from_quantizer(analysis.vp8_quantizer_index);
    if strength > 0 {
        deblock_rgba(&mut rgba, w as usize, h as usize, strength);
    }

    let cfg = LossyConfig::new()
        .with_quality(libwebp_q.clamp(1, 100) as f32)
        .with_method(4);
    let bytes = EncodeRequest::lossy(&cfg, &rgba, PixelLayout::Rgba8, w, h)
        .encode()
        .map_err(|e| Error::EncodeFailed(format!("{e:?}")))?;
    Ok(bytes)
}
