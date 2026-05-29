//! Coefficient-domain edits + the full transcode pipeline.
//!
//! The "least-disruptive" recompression path: edit the parsed quantized
//! coefficients and re-emit, with NO IDCT/FDCT spatial round-trip. The
//! only edit shipped today is high-frequency AC dropping — preserve DC and
//! the low-frequency AC exactly, zero the high-frequency tail. That removes
//! fine detail (entropy) for a smaller file while leaving the block
//! structure and low frequencies bit-identical.

use super::emit::emit_vp8_keyframe;
use super::parse::{Frame, parse_vp8_keyframe};
use super::tables::ZIGZAG;

/// Zero every AC coefficient at scan (zigzag) position `>= keep_ac` in
/// every block. `keep_ac` is the number of leading scan coefficients to
/// preserve (1 = DC only; 16 = no change). DC (position 0) is always kept.
pub fn drop_high_freq_ac(frame: &mut Frame, keep_ac: usize) {
    let keep = keep_ac.clamp(1, 16);
    if keep == 16 {
        return;
    }
    for mb in frame.mbs.iter_mut() {
        if mb.skip {
            continue;
        }
        for block in mb.coeffs.iter_mut() {
            // Zero positions keep..16 in SCAN order → de-zigzagged indices.
            for (&zz, _) in ZIGZAG
                .iter()
                .enumerate()
                .filter(|(scan, _)| *scan >= keep)
                .map(|(s, z)| (z, s))
            {
                block[zz as usize] = 0;
            }
        }
    }
}

/// Coarsen the quantized coefficient *levels* to a wider grid, with the
/// frame header (and thus the per-coefficient dequant factors) left
/// untouched. For each level `L`, the new level is `round(L / f)`, so the
/// decoder reconstructs `round(L/f) * factor` — i.e. the original value
/// snapped to a grid `f×` coarser. This is graceful, real-quantization-
/// shaped degradation (unlike [`drop_high_freq_ac`]'s hard truncation),
/// and it shrinks the levels → fewer token bits → smaller file. **No
/// IDCT/FDCT spatial round-trip**, so the surviving precision is exact.
///
/// `f_dc` scales the DC coefficient (scan position 0) of every block;
/// `f_ac` scales the 15 AC coefficients. Keeping `f_dc` near 1.0 preserves
/// block mean luma/chroma (avoids banding) while `f_ac >= 1` sheds the
/// high-entropy detail. `f <= 1.0` for a plane leaves it unchanged.
pub fn requantize(frame: &mut Frame, f_dc: f32, f_ac: f32) {
    let inv_dc = if f_dc > 1.0 { 1.0 / f_dc as f64 } else { 1.0 };
    let inv_ac = if f_ac > 1.0 { 1.0 / f_ac as f64 } else { 1.0 };
    if inv_dc == 1.0 && inv_ac == 1.0 {
        return;
    }
    let scale = |v: i32, inv: f64| (v as f64 * inv).round() as i32;
    for mb in frame.mbs.iter_mut() {
        if mb.skip {
            continue;
        }
        let has_y2 = mb.has_y2();
        for (bi, block) in mb.coeffs.iter_mut().enumerate() {
            match bi {
                // Block 0 is the Y2 (second-order WHT) block. When present it
                // carries the *DC* of all 16 luma subblocks (the Y subblocks'
                // own position-0 is then zero). Treat the whole Y2 block as
                // DC-domain so AC coarsening never mangles luma block means.
                0 if has_y2 => {
                    if inv_dc != 1.0 {
                        for c in block.iter_mut() {
                            *c = scale(*c, inv_dc);
                        }
                    }
                }
                // Y blocks (1..=16) when a Y2 block is present: position 0 is
                // zero (DC lives in Y2), so only positions 1..16 are real AC.
                1..=16 if has_y2 => {
                    if inv_ac != 1.0 {
                        for c in block.iter_mut().skip(1) {
                            *c = scale(*c, inv_ac);
                        }
                    }
                }
                // Otherwise (B_PRED luma blocks, or chroma) position 0 is the
                // block's own DC and 1..16 are AC.
                _ => {
                    if inv_dc != 1.0 {
                        block[0] = scale(block[0], inv_dc);
                    }
                    if inv_ac != 1.0 {
                        for c in block.iter_mut().skip(1) {
                            *c = scale(*c, inv_ac);
                        }
                    }
                }
            }
        }
    }
}

/// Parse a VP8 keyframe, drop high-frequency AC to `keep_ac`, re-emit.
/// Returns the new VP8 bitstream, or `None` if the frame can't be parsed
/// (non-keyframe / unsupported) — the caller then falls back.
pub fn transcode_drop_ac(vp8: &[u8], keep_ac: usize) -> Option<Vec<u8>> {
    let mut frame = parse_vp8_keyframe(vp8)?;
    drop_high_freq_ac(&mut frame, keep_ac);
    Some(emit_vp8_keyframe(&frame))
}

/// Parse a VP8 keyframe, coarsen levels by (`f_dc`, `f_ac`), re-emit.
pub fn transcode_requantize(vp8: &[u8], f_dc: f32, f_ac: f32) -> Option<Vec<u8>> {
    let mut frame = parse_vp8_keyframe(vp8)?;
    requantize(&mut frame, f_dc, f_ac);
    Some(emit_vp8_keyframe(&frame))
}

/// Verbatim transcode (parse → re-emit unchanged). Used to validate the
/// round-trip and as the `keep_ac = 16` identity.
pub fn transcode_verbatim(vp8: &[u8]) -> Option<Vec<u8>> {
    let frame = parse_vp8_keyframe(vp8)?;
    Some(emit_vp8_keyframe(&frame))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenwebp::PixelLayout;
    use zenwebp::encoder::{EncodeRequest, LossyConfig};
    use zenwebp::oneshot::decode_rgba;

    fn extract_vp8(webp: &[u8]) -> Option<&[u8]> {
        if webp.len() < 20 || &webp[0..4] != b"RIFF" || &webp[8..12] != b"WEBP" {
            return None;
        }
        let mut pos = 12;
        while pos + 8 <= webp.len() {
            let fourcc = &webp[pos..pos + 4];
            let size =
                u32::from_le_bytes([webp[pos + 4], webp[pos + 5], webp[pos + 6], webp[pos + 7]])
                    as usize;
            let body = pos + 8;
            if fourcc == b"VP8 " {
                return webp.get(body..body + size);
            }
            pos = body + size + (size & 1);
        }
        None
    }
    fn wrap_vp8(vp8: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        let riff_size = 4 + 8 + vp8.len() + (vp8.len() & 1);
        out.extend_from_slice(&(riff_size as u32).to_le_bytes());
        out.extend_from_slice(b"WEBP");
        out.extend_from_slice(b"VP8 ");
        out.extend_from_slice(&(vp8.len() as u32).to_le_bytes());
        out.extend_from_slice(vp8);
        if vp8.len() & 1 == 1 {
            out.push(0);
        }
        out
    }

    /// Dropping high-frequency AC produces a valid, smaller WebP that
    /// decodes to a recognizable (close) version of the source.
    #[test]
    fn drop_ac_shrinks_and_decodes() {
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
        let vp8 = extract_vp8(&webp).unwrap();

        // keep_ac = 16 (verbatim) must be pixel-exact.
        let v = transcode_verbatim(vp8).unwrap();
        let (o, _, _) = decode_rgba(&webp).unwrap();
        let (r, _, _) = decode_rgba(&wrap_vp8(&v)).unwrap();
        let vdiffs = o
            .chunks_exact(4)
            .zip(r.chunks_exact(4))
            .filter(|(a, b)| a[0..3] != b[0..3])
            .count();
        assert_eq!(vdiffs, 0, "verbatim transcode must be pixel-exact");

        // keep_ac = 4 drops high-freq → smaller, still decodes, still close.
        let edited = transcode_drop_ac(vp8, 4).unwrap();
        assert!(
            edited.len() < vp8.len(),
            "drop-AC must shrink: {} vs {}",
            edited.len(),
            vp8.len()
        );
        let (e, ew, eh) = decode_rgba(&wrap_vp8(&edited)).expect("edited decodes");
        assert_eq!((ew, eh), (w, h));
        // Mean abs RGB delta should be modest (we kept DC + low AC).
        let total: u64 = o
            .chunks_exact(4)
            .zip(e.chunks_exact(4))
            .map(|(a, b)| {
                (0..3)
                    .map(|c| (a[c] as i32 - b[c] as i32).unsigned_abs() as u64)
                    .sum::<u64>()
            })
            .sum();
        let mad = total as f64 / (o.len() as f64 / 4.0 * 3.0);
        eprintln!(
            "drop_ac(4): {} -> {} bytes ({:.1}%), MAD={:.2}",
            vp8.len(),
            edited.len(),
            100.0 * edited.len() as f64 / vp8.len() as f64,
            mad
        );
        assert!(mad < 40.0, "drop-AC distortion implausibly high: MAD={mad}");
    }
}
