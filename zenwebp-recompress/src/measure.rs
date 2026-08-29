//! zensim scoring helpers used by the iterative refinement loop and by
//! `zwr-calibrate`.
//!
//! For the one-shot router path we never call into this module — the entire
//! point of one-shot is "no IQA loop". These functions exist for budgets
//! that allow measurement and for the calibration harness.
//!
//! # Why this pins the deprecated `ZensimProfile::A`
//!
//! zensim 0.3.0 deprecated generation-A (`zensim-a`, the v47-strict-QAT MLP)
//! in favour of the deterministic generation-B linear profile, and gated `A`
//! behind zensim's `deprecated-profiles` feature. That feature is ON in
//! zensim's default set, but this workspace takes zensim with
//! `default-features = false`, so `A` silently compiled out and this module
//! stopped building (see CHANGELOG 2026-08-29). The dep now names
//! `deprecated-profiles` explicitly.
//!
//! Switching to `ZensimProfile::B` / `codec_target()` would be a **score
//! change, not a rename**: zensim documents `B` vs `A` as a trade rather than
//! strict dominance, and every constant in `crate::calibration::calib_tables`
//! is an absolute zensim-**A** score (the `*_CUM` grids, the `*_SOURCE_CUM`
//! line fits) produced by a 248,501-cell A-scored sweep. The public
//! `target_zensim_a` knob is on the same scale. Re-pointing the profile
//! without re-fitting that table would leave the projections silently
//! miscalibrated and make every previously recorded number incomparable, so
//! `A` stays pinned until a B-scored recalibration sweep lands.

use crate::error::Error;
use bytemuck::try_cast_slice;
use zensim::{RgbaSlice, Zensim, ZensimProfile};
use zenwebp::oneshot::decode_rgba;

/// The one profile every score in this crate is expressed in.
///
/// Deliberately the **deprecated** generation-A profile — see the module
/// docs. Changing this constant re-bases every number the crate produces and
/// invalidates `crate::calibration::calib_tables`, so it is pinned here (one
/// place, one guard test) rather than named at the call site.
#[allow(deprecated)]
pub(crate) const PROFILE: ZensimProfile = ZensimProfile::A;

/// Score recompression generation loss against the source.
///
/// Decodes both `source_webp` and `output_webp` to RGBA, then runs
/// zensim Profile A. Returns the score in `[0.0, 100.0]`:
/// `100.0` = bit-identical, lower = more recompression damage.
///
/// This is the **generation-loss** signal: it tells you how much the
/// recompressor mangled the bits relative to the input. It does NOT tell
/// you the cumulative distance from the (unknown) original reference;
/// that's what the calibration table projects.
pub fn score_recompression(source_webp: &[u8], output_webp: &[u8]) -> Result<f32, Error> {
    let (src_rgba, src_w, src_h) =
        decode_rgba(source_webp).map_err(|e| Error::DecodeFailed(format!("{e:?}")))?;
    let (dst_rgba, dst_w, dst_h) =
        decode_rgba(output_webp).map_err(|e| Error::DecodeFailed(format!("{e:?}")))?;

    if src_w != dst_w || src_h != dst_h {
        return Err(Error::DecodeFailed(format!(
            "dimension mismatch source={}x{} output={}x{}",
            src_w, src_h, dst_w, dst_h
        )));
    }

    score_rgba(&src_rgba, &dst_rgba, src_w, src_h)
}

/// Score `output_webp` against an unencoded RGBA reference. Used by
/// `zwr-calibrate` when both reference and recompressed-derived-from-source
/// are in hand.
#[allow(dead_code)] // Available via expert::score_against_reference for harnesses.
pub fn score_against_reference(reference_rgba: &[u8], output_webp: &[u8]) -> Result<f32, Error> {
    let (out_rgba, w, h) =
        decode_rgba(output_webp).map_err(|e| Error::DecodeFailed(format!("{e:?}")))?;
    score_rgba(reference_rgba, &out_rgba, w, h)
}

/// Direct RGBA-vs-RGBA scoring helper. Both buffers must be
/// `width * height * 4` bytes of contiguous RGBA8 (straight alpha).
pub fn score_rgba(ref_rgba: &[u8], dst_rgba: &[u8], width: u32, height: u32) -> Result<f32, Error> {
    let ref_pixels: &[[u8; 4]] =
        try_cast_slice(ref_rgba).map_err(|e| Error::DecodeFailed(format!("ref cast: {e:?}")))?;
    let dst_pixels: &[[u8; 4]] =
        try_cast_slice(dst_rgba).map_err(|e| Error::DecodeFailed(format!("dst cast: {e:?}")))?;
    let ref_src = RgbaSlice::try_new(ref_pixels, width as usize, height as usize)
        .map_err(|e| Error::DecodeFailed(format!("zensim ref: {e:?}")))?;
    let dst_src = RgbaSlice::try_new(dst_pixels, width as usize, height as usize)
        .map_err(|e| Error::DecodeFailed(format!("zensim dst: {e:?}")))?;
    let result = Zensim::new(PROFILE)
        .compute(&ref_src, &dst_src)
        .map_err(|e| Error::DecodeFailed(format!("zensim compute: {e:?}")))?;
    Ok(result.score() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The calibration tables in `crate::calibration::calib_tables` are
    /// absolute zensim-**A** scores. If a future zensim drop of generation-A
    /// (or a well-meaning migration to `B` / `codec_target()`) re-points
    /// [`PROFILE`], every projection this crate makes silently shifts scale
    /// and stops being comparable to any previously recorded number.
    ///
    /// Fail loudly here instead. Re-pointing `PROFILE` is legitimate ONLY
    /// together with a re-fit of the calibration tables — at which point
    /// update this expectation in the same change.
    #[test]
    fn profile_is_pinned_to_zensim_a() {
        assert_eq!(
            PROFILE.name(),
            "zensim-a",
            "measure::PROFILE changed; the calibration tables are in \
             zensim-A units and must be re-fit before re-pointing it"
        );
    }
}
