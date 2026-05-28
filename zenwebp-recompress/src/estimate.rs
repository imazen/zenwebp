//! Decode-based source quality estimation.
//!
//! `zenwebp::detect`'s header quantizer is unreliable for segmented WebP
//! (see `docs/QUALITY_DETECTION.md`). Since `recompress()` decodes the
//! source anyway, we estimate the source's effective libwebp quality by
//! **recompression self-consistency**: re-encode the decoded pixels across
//! a small probe sweep and find the quality whose output size matches the
//! source. A source encoded near quality Q reproduces ~its own byte size
//! when re-encoded at Q; far above Q it inflates, far below it shrinks.
//!
//! The size-match crossing is monotonic in true Q with a roughly constant
//! offset (the 2nd-generation re-encode runs a touch smaller than the
//! source at the same Q), which we subtract as `GENERATION_OFFSET`.

use crate::error::Error;
use zenwebp::PixelLayout;
use zenwebp::encoder::{EncodeRequest, LossyConfig};

/// Probe qualities for the size-match search. Four points spread across
/// the useful range bracket the source size; the estimator interpolates
/// between the two that straddle it. (Four keeps the per-recompress cost
/// to four encodes — measured enough to locate the crossing without a
/// dense sweep.)
const PROBE_QUALITIES: &[f32] = &[30.0, 50.0, 70.0, 90.0];

/// Re-encoding at the source's own quality runs slightly smaller than the
/// source (the source carries one less generation of quantization). The
/// size-match quality therefore sits a little *above* the true quality;
/// subtract this to de-bias. Calibrated from the clean photo sweep
/// (re-encode ratio ≈ 0.92–0.96 at the true q → crossing ~+10).
const GENERATION_OFFSET: f32 = 10.0;

/// Estimate the source's effective libwebp quality `[1, 100]` from its
/// decoded pixels and encoded size.
///
/// `rgba` is the decoded source (straight RGBA8, `width*height*4` bytes).
/// `source_len` is the encoded source's byte length. Returns a quality in
/// `[1, 100]`; falls back to `60.0` if probing fails.
///
/// Cost: `PROBE_QUALITIES.len()` lossy encodes (method 4). Used on the
/// `recompress()` path (which already decoded); `plan()` does not call it.
pub fn estimate_quality_by_recompression(
    rgba: &[u8],
    width: u32,
    height: u32,
    source_len: usize,
) -> Result<f32, Error> {
    if source_len == 0 || width == 0 || height == 0 {
        return Ok(60.0);
    }

    // Encode at each probe quality, record size.
    let mut points: Vec<(f32, usize)> = Vec::with_capacity(PROBE_QUALITIES.len());
    for &q in PROBE_QUALITIES {
        let cfg = LossyConfig::new().with_quality(q).with_method(4);
        let bytes = EncodeRequest::lossy(&cfg, rgba, PixelLayout::Rgba8, width, height)
            .encode()
            .map_err(|e| Error::EncodeFailed(format!("probe q{q}: {e:?}")))?;
        points.push((q, bytes.len()));
    }

    let target = source_len as f32;
    let crossing = size_match_quality(&points, target);
    Ok((crossing - GENERATION_OFFSET).clamp(1.0, 100.0))
}

/// Find the quality at which probe size crosses `target_size`, by linear
/// interpolation between the two bracketing probes. Probe sizes are
/// monotonically increasing in quality, so a single crossing exists (or we
/// clamp to an end).
fn size_match_quality(points: &[(f32, usize)], target: f32) -> f32 {
    if points.is_empty() {
        return 60.0;
    }
    // Below the smallest probe size → source is lower quality than the
    // lowest probe.
    if target <= points[0].1 as f32 {
        return points[0].0;
    }
    let last = points.len() - 1;
    if target >= points[last].1 as f32 {
        return points[last].0;
    }
    for w in points.windows(2) {
        let (q0, s0) = (w[0].0, w[0].1 as f32);
        let (q1, s1) = (w[1].0, w[1].1 as f32);
        if target >= s0 && target <= s1 {
            let frac = (target - s0) / (s1 - s0).max(1.0);
            return q0 + frac * (q1 - q0);
        }
    }
    points[last].0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_match_interpolates() {
        let pts = [(20.0, 1000usize), (50.0, 4000), (80.0, 9000)];
        // target exactly at a probe
        assert!((size_match_quality(&pts, 4000.0) - 50.0).abs() < 0.01);
        // midway between 20 and 50 in size (2500 is halfway 1000..4000)
        let q = size_match_quality(&pts, 2500.0);
        assert!((q - 35.0).abs() < 0.5, "expected ~35, got {q}");
        // clamp below / above
        assert_eq!(size_match_quality(&pts, 500.0), 20.0);
        assert_eq!(size_match_quality(&pts, 99999.0), 80.0);
    }

    #[test]
    fn empty_points_default() {
        assert_eq!(size_match_quality(&[], 1234.0), 60.0);
    }
}
