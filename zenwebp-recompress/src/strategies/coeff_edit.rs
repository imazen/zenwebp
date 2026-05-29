//! VP8 coefficient-domain edit — the "least-disruptive" recompression path.
//! Parse the VP8 keyframe to quantized coefficient levels, edit them (drop
//! the high-frequency AC tail, or coarsen the level grid), and re-emit the
//! token stream with **no IDCT/FDCT spatial round-trip**.
//!
//! Implemented by the validated [`crate::vp8x`] transcoder, whose verbatim
//! (no-edit) path is pixel-exact with libwebp's decoder.
//!
//! ## Measured result: RD-dominated by `Reencode` for WebP (2026-05-28)
//!
//! The appealing theory is "edit coefficients → avoid generation loss." It
//! does not hold for VP8/WebP, and the project measured this carefully
//! (`benchmarks/coeff_edit_experiment_2026-05-28.md`):
//!
//! - The transcoder is correct: a verbatim transcode and a no-op
//!   requantize (`f = 1.0`) are bit-for-bit pixel-exact (MAD 0).
//! - But **any** coefficient edit that actually shrinks the file is RD-
//!   dominated by `Reencode` at matched output size — on 4 of 5 reference
//!   images decisively, the 5th a wash. Even a gentle `f_ac = 1.25`
//!   requantize produces MAD ≈ 15.6 against the source decode.
//!
//! The mechanism is **VP8 spatial intra-prediction drift**: every block is
//! predicted from its neighbours' *reconstructed* pixels, so perturbing one
//! block's coefficients shifts every downstream block's prediction base and
//! the error compounds across the frame. `Reencode` re-derives residuals
//! from clean decoded pixels and re-runs RD optimisation (trellis, mode
//! choice), so it spends bits far better at the same size. Coefficient-
//! domain editing is the right tool for codecs *without* inter-block
//! prediction (baseline JPEG, where requantization is drift-free and
//! competitive) — VP8's intra prediction is exactly what defeats it.
//!
//! CoeffEdit's no-generation-loss advantage is therefore only realisable at
//! the verbatim point (zero size change), which `LosslessRemux` already
//! provides. **The router does not select CoeffEdit** (see
//! `router::filter_candidates`); these functions remain available through
//! the `expert` API as validated research tools and to reproduce the
//! falsification. Each is guarded: a frame that can't be parsed, or a
//! result that doesn't decode, returns an error so callers fall back.

use crate::api::RecompressOptions;
use crate::error::Error;
use crate::source::{SourceAnalysis, SourceKind};
use crate::target;
use crate::vp8x;

/// Map a target zensim-A to the number of leading scan coefficients to
/// keep (1 = DC only, 16 = lossless verbatim). Lower targets keep fewer
/// AC coefficients (more aggressive high-frequency truncation).
///
/// Heuristic pending a keep_ac→zensim calibration sweep; the correctness
/// guard in [`run_coeff_edit`] keeps it safe regardless.
fn keep_ac_for_target(target_zensim_a: f32) -> usize {
    // libwebp-quality analog of the target, then a coarse mapping.
    let q = target::target_zensim_a_to_libwebp_q(target_zensim_a) as f32;
    // q95→keep ~15, q75→~10, q50→~6, q30→~3.
    ((q / 7.0).round() as usize).clamp(2, 15)
}

pub fn run_coeff_edit(
    webp_bytes: &[u8],
    analysis: &SourceAnalysis,
    opts: &RecompressOptions,
) -> Result<Vec<u8>, Error> {
    if !matches!(analysis.kind, SourceKind::LossyVp8) {
        return Err(Error::EncodeFailed(
            "CoeffEdit: source is not lossy VP8".into(),
        ));
    }
    let keep = keep_ac_for_target(opts.target_zensim_a);
    run_coeff_edit_keep(webp_bytes, keep)
}

/// CoeffEdit at an explicit `keep_ac`. Extracts the VP8 chunk, transcodes,
/// swaps it back into the container (preserving ALPH / ICCP / EXIF / etc.),
/// and validates the result decodes. Errors → caller falls back.
pub fn run_coeff_edit_keep(webp_bytes: &[u8], keep_ac: usize) -> Result<Vec<u8>, Error> {
    let (vp8, vp8_span) = find_vp8_chunk(webp_bytes)
        .ok_or_else(|| Error::DecodeFailed("CoeffEdit: no VP8 chunk".into()))?;
    let new_vp8 = vp8x::edit::transcode_drop_ac(vp8, keep_ac)
        .ok_or_else(|| Error::EncodeFailed("CoeffEdit: not a parseable VP8 keyframe".into()))?;

    let out = swap_vp8_chunk(webp_bytes, vp8_span, &new_vp8);

    // Correctness guard: the result MUST decode.
    zenwebp::oneshot::decode_rgba(&out)
        .map_err(|e| Error::EncodeFailed(format!("CoeffEdit output failed to decode: {e:?}")))?;
    Ok(out)
}

/// CoeffEdit by level-grid coarsening: scale the DC levels by `f_dc` and the
/// AC levels by `f_ac` (each `>= 1.0`; `1.0` = unchanged). Header and dequant
/// factors are untouched, so reconstructed values snap to a grid `f×` coarser.
/// Same container-preserving rewrap + decode guard as [`run_coeff_edit_keep`].
///
/// Exposed for research/reproduction; see the module docs — this is RD-
/// dominated by `Reencode` and is not selected by the router.
pub fn run_coeff_edit_requant(webp_bytes: &[u8], f_dc: f32, f_ac: f32) -> Result<Vec<u8>, Error> {
    let (vp8, vp8_span) = find_vp8_chunk(webp_bytes)
        .ok_or_else(|| Error::DecodeFailed("CoeffEdit: no VP8 chunk".into()))?;
    let new_vp8 = vp8x::edit::transcode_requantize(vp8, f_dc, f_ac)
        .ok_or_else(|| Error::EncodeFailed("CoeffEdit: not a parseable VP8 keyframe".into()))?;
    let out = swap_vp8_chunk(webp_bytes, vp8_span, &new_vp8);
    zenwebp::oneshot::decode_rgba(&out).map_err(|e| {
        Error::EncodeFailed(format!("CoeffEdit requant output failed to decode: {e:?}"))
    })?;
    Ok(out)
}

/// Locate the `VP8 ` chunk: returns its payload slice + the (start,end)
/// byte span of the *payload* within `webp` (for in-place swap).
fn find_vp8_chunk(webp: &[u8]) -> Option<(&[u8], (usize, usize))> {
    if webp.len() < 12 || &webp[0..4] != b"RIFF" || &webp[8..12] != b"WEBP" {
        return None;
    }
    let mut pos = 12;
    while pos + 8 <= webp.len() {
        let fourcc = &webp[pos..pos + 4];
        let size = u32::from_le_bytes([webp[pos + 4], webp[pos + 5], webp[pos + 6], webp[pos + 7]])
            as usize;
        let body = pos + 8;
        let end = body.checked_add(size)?;
        if end > webp.len() {
            return None;
        }
        if fourcc == b"VP8 " {
            return Some((&webp[body..end], (body, end)));
        }
        pos = end + (size & 1); // chunks are even-padded
    }
    None
}

/// Rebuild the container with the VP8 chunk payload replaced. Preserves
/// every other chunk (VP8X, ALPH, ICCP, EXIF, ANIM…) and fixes up the VP8
/// chunk size, its even-padding, and the RIFF size.
fn swap_vp8_chunk(webp: &[u8], payload_span: (usize, usize), new_vp8: &[u8]) -> Vec<u8> {
    let (start, end) = payload_span;
    // The chunk header (FourCC + size) is the 8 bytes before `start`.
    let header_start = start - 8;
    let old_padded = (end - start) + ((end - start) & 1);
    let pre = &webp[..header_start]; // RIFF header + any chunks before VP8
    let post = &webp[(start + old_padded).min(webp.len())..]; // chunks after VP8

    let mut out = Vec::with_capacity(pre.len() + 8 + new_vp8.len() + 1 + post.len());
    out.extend_from_slice(pre);
    out.extend_from_slice(b"VP8 ");
    out.extend_from_slice(&(new_vp8.len() as u32).to_le_bytes());
    out.extend_from_slice(new_vp8);
    if new_vp8.len() & 1 == 1 {
        out.push(0);
    }
    out.extend_from_slice(post);

    // Fix RIFF size = total file size - 8.
    let riff_size = (out.len() - 8) as u32;
    out[4..8].copy_from_slice(&riff_size.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenwebp::PixelLayout;
    use zenwebp::encoder::{EncodeRequest, LossyConfig};
    use zenwebp::oneshot::decode_rgba;

    #[test]
    fn coeff_edit_simple_container_shrinks_and_decodes() {
        let (w, h) = (128u32, 96u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let n = ((x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503)) >> 11) & 0x3f;
                rgba[i] = ((x * 2) + n) as u8;
                rgba[i + 1] = ((y * 2) + n) as u8;
                rgba[i + 2] = ((x + y) + n) as u8;
                rgba[i + 3] = 255;
            }
        }
        let webp = EncodeRequest::lossy(
            &LossyConfig::new().with_quality(90.0).with_method(4),
            &rgba,
            PixelLayout::Rgba8,
            w,
            h,
        )
        .encode()
        .unwrap();
        let out = run_coeff_edit_keep(&webp, 4).expect("coeff edit");
        assert!(
            out.len() < webp.len(),
            "must shrink: {} vs {}",
            out.len(),
            webp.len()
        );
        let (_px, ow, oh) = decode_rgba(&out).expect("decodes");
        assert_eq!((ow, oh), (w, h));
    }
}
