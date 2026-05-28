//! zensim scoring helpers used by the iterative refinement loop and by
//! `zwr-calibrate`.
//!
//! For the one-shot router path we never call into this module — the entire
//! point of one-shot is "no IQA loop". These functions exist for budgets
//! that allow measurement and for the calibration harness.

use crate::error::Error;
use bytemuck::try_cast_slice;
use zensim::{RgbaSlice, Zensim, ZensimProfile};
use zenwebp::oneshot::decode_rgba;

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
    let result = Zensim::new(ZensimProfile::A)
        .compute(&ref_src, &dst_src)
        .map_err(|e| Error::DecodeFailed(format!("zensim compute: {e:?}")))?;
    Ok(result.score() as f32)
}
