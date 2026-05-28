//! Full decode → block-artifact-aware deblock → re-encode at calibrated Q.
//!
//! # Measured caveat (2026-05-28)
//!
//! A paired-sweep experiment (`benchmarks/deblock_experiment_2026-05-28.md`)
//! showed that applying our deblock to **already-loop-filtered VP8 output**
//! is net-negative: −2.75 zensim cumulative and +3.9% size over 150
//! `qi ≥ 60` cells. VP8 applies an in-loop deblock during decode
//! reconstruction, so the decoded pixels are *already* deblocked; a second
//! pass over-smooths real detail.
//!
//! Therefore the deblock pass only fires when the source's loop filter was
//! weak (`vp8_filter_level < FILTER_WEAK_THRESHOLD`) — i.e. the source
//! encoder disabled or under-applied in-loop filtering, leaving genuine
//! block artifacts our filter can remove. When the source was strongly
//! loop-filtered (the common libwebp-default case), this strategy is
//! identical to plain `Reencode` and the router has no reason to prefer it.

use crate::api::RecompressOptions;
use crate::error::Error;
use crate::source::{SourceAnalysis, SourceKind};
use crate::strategies::deblock::{deblock_rgba, strength_from_quantizer};
use crate::target;
use zenwebp::PixelLayout;
use zenwebp::encoder::{EncodeRequest, LossyConfig};
use zenwebp::oneshot::decode_rgba;

/// Source loop-filter level below which we consider the source
/// under-deblocked and apply our own pass. Above this, VP8's in-loop
/// filter already handled boundaries and a second pass over-smooths.
pub const FILTER_WEAK_THRESHOLD: u8 = 8;

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

    // Only deblock if the source was weakly loop-filtered. Otherwise the
    // decoded pixels are already deblocked and a second pass hurts.
    let should_deblock = analysis.vp8_filter_level < FILTER_WEAK_THRESHOLD;
    let strength = if should_deblock {
        strength_from_quantizer(analysis.vp8_quantizer_index)
    } else {
        0
    };
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
