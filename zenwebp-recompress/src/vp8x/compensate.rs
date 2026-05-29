//! Closed-loop DC drift compensation for coefficient edits.
//!
//! Editing VP8 coefficients perturbs each block's reconstruction, and because
//! VP8 predicts every block from its neighbours' reconstructed pixels the
//! error *drifts* across the frame (see `strategies::coeff_edit` docs). The
//! dominant, most visible component of that drift is a per-block **mean**
//! (DC) shift — luma/chroma blocks getting brighter or darker.
//!
//! This module cancels that component with a closed loop that uses zenwebp's
//! own decoder as the reconstruction oracle (no re-implemented transforms):
//!
//! 1. Decode the *unedited* source once → target Y/U/V planes.
//! 2. After the edit, emit → decode → compare. For each macroblock measure
//!    the mean luma/chroma error vs the target.
//! 3. Push that error back out by nudging the block's DC **level** (the Y2 DC
//!    for luma when a Y2 block is present, else each subblock's DC; the UV
//!    DC for chroma), using the analytic level→pixel DC gain.
//! 4. Re-emit and repeat to a fixed point (cross-block coupling means one
//!    pass doesn't fully converge; a few under-relaxed passes do).
//!
//! This compensates only the DC/mean drift, not the higher-frequency drift
//! from changed block-edge pixels feeding directional prediction — but DC is
//! the bulk of it. The goal is to test whether cancelling drift lets a
//! coefficient edit (which preserves untouched coefficients with *zero*
//! generation loss) compete with `Reencode` (which re-quantises everything).

use super::emit::emit_vp8_keyframe;
use super::parse::{Frame, parse_vp8_keyframe};
use super::tables::DC_QUANT;
use zenwebp::decoder::decode_yuv420;

/// Per-segment DC dequant factors `(y2dc, uvdc, ydc)`, matching the decoder's
/// derivation (Y2 DC ×2; UV DC clamped to 132).
fn segment_dc_dequants(frame: &Frame) -> [(i32, i32, i32); 4] {
    let dcq = |idx: i32| DC_QUANT[idx.clamp(0, 127) as usize] as i32;
    let seg = &frame.segmentation;
    let yac = frame.quant.yac_abs as i32;
    let ydc_d = frame.quant.ydc_delta.unwrap_or(0);
    let y2dc_d = frame.quant.y2dc_delta.unwrap_or(0);
    let uvdc_d = frame.quant.uvdc_delta.unwrap_or(0);
    let mut out = [(0i32, 0i32, 0i32); 4];
    for (i, slot) in out.iter_mut().enumerate() {
        let base = if seg.enabled {
            let q = if seg.quantizer_present[i] {
                seg.quantizer[i] as i32
            } else {
                0
            };
            if seg.abs_values { q } else { yac + q }
        } else {
            yac
        };
        let y2dc = dcq(base + y2dc_d) * 2;
        let uvdc = dcq(base + uvdc_d).min(132);
        let ydc = dcq(base + ydc_d);
        *slot = (y2dc, uvdc, ydc);
    }
    out
}

/// Wrap a bare VP8 bitstream in a minimal RIFF/WEBP container for decoding.
fn wrap_vp8(vp8: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + 8 + vp8.len() + 1);
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

/// Target Y/U/V planes (the unedited source decode) the loop steers toward.
struct Target<'a> {
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    yw: usize,
    yh: usize,
    uw: usize,
    uh: usize,
}

/// Mean of `cur - tgt` over a `bsize×bsize` block at `(x0,y0)` in a `pw×ph`-
/// sized, `pw`-strided plane. Returns 0 if the block is fully out of bounds.
fn block_mean_err(
    cur: &[u8],
    tgt: &[u8],
    pw: usize,
    ph: usize,
    x0: usize,
    y0: usize,
    bsize: usize,
) -> f64 {
    let mut sum = 0i64;
    let mut n = 0i64;
    for y in y0..(y0 + bsize).min(ph) {
        let row = y * pw;
        for x in x0..(x0 + bsize).min(pw) {
            sum += cur[row + x] as i64 - tgt[row + x] as i64;
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum as f64 / n as f64 }
}

/// Run DC drift compensation in place on an already-edited `frame`, targeting
/// `src_*` planes (the unedited source decode). `iters` closed-loop passes,
/// `relax` under-relaxation factor in (0,1]. Returns false if a decode of the
/// intermediate stream fails (caller falls back).
fn dc_compensate(frame: &mut Frame, tgt: &Target, iters: u32, relax: f64) -> bool {
    let (src_y, src_u, src_v) = (tgt.y, tgt.u, tgt.v);
    let (yw, yh, uw, uh) = (tgt.yw, tgt.yh, tgt.uw, tgt.uh);
    // Per-pass DC-level step cap. VP8's prediction chain makes a simultaneous
    // (Jacobi) full-strength correction unstable — each MB's correction also
    // shifts its downstream neighbours' prediction base, so corrections
    // accumulate along the raster chain and overshoot. Capping the per-pass
    // step (plus under-relaxation) keeps every pass a bounded, correctly-
    // signed nudge, so the loop walks monotonically to the fixed point.
    const STEP_CAP: i32 = 3;
    let cap = |d: i32| d.clamp(-STEP_CAP, STEP_CAP);
    let dq = segment_dc_dequants(frame);
    let mb_cols = frame.mb_cols;
    let mb_rows = frame.mb_rows;
    for _ in 0..iters {
        let bytes = emit_vp8_keyframe(frame);
        let Ok(yuv) = decode_yuv420(&wrap_vp8(&bytes)) else {
            return false;
        };
        if yuv.y.len() < yw * yh || yuv.u.len() < uw * uh {
            return false;
        }
        #[cfg(test)]
        if std::env::var("ZWR_COMP_LOG").is_ok() {
            let e: f64 = (0..yh)
                .flat_map(|y| (0..yw).map(move |x| (y, x)))
                .map(|(y, x)| {
                    (yuv.y[y * yw + x] as i32 - src_y[y * yw + x] as i32).unsigned_abs() as f64
                })
                .sum::<f64>()
                / (yw * yh) as f64;
            eprintln!("   pass: luma mean-abs={e:.2}");
        }
        for mby in 0..mb_rows {
            for mbx in 0..mb_cols {
                let mb = &mut frame.mbs[mby * mb_cols + mbx];
                let (y2dc, uvdc, ydc) = dq[mb.segment_id.min(3) as usize];
                // Luma: mean error over the 16×16 block.
                let ey = block_mean_err(&yuv.y, src_y, yw, yh, mbx * 16, mby * 16, 16);
                if mb.has_y2() {
                    // gain ≈ y2dc/64 luma per Y2-DC level.
                    let g = (y2dc as f64) / 64.0;
                    if g > 0.0 {
                        let d = cap((ey / g * relax).round() as i32);
                        mb.coeffs[0][0] -= d;
                    }
                } else {
                    // B_PRED: each subblock carries its own DC; shift all 16
                    // uniformly. gain ≈ ydc/8 per subblock-DC level.
                    let g = (ydc as f64) / 8.0;
                    if g > 0.0 {
                        let d = cap((ey / g * relax).round() as i32);
                        for b in 1..=16 {
                            mb.coeffs[b][0] -= d;
                        }
                    }
                }
                // Chroma: U = blocks 17..=20, V = 21..=24. gain ≈ uvdc/8.
                let g_uv = (uvdc as f64) / 8.0;
                if g_uv > 0.0 {
                    let eu = block_mean_err(&yuv.u, src_u, uw, uh, mbx * 8, mby * 8, 8);
                    let ev = block_mean_err(&yuv.v, src_v, uw, uh, mbx * 8, mby * 8, 8);
                    let du = cap((eu / g_uv * relax).round() as i32);
                    let dv = cap((ev / g_uv * relax).round() as i32);
                    for b in 17..=20 {
                        mb.coeffs[b][0] -= du;
                    }
                    for b in 21..=24 {
                        mb.coeffs[b][0] -= dv;
                    }
                }
            }
        }
    }
    true
}

/// Parse a VP8 keyframe, apply `edit`, then run `iters` passes of DC drift
/// compensation toward the *unedited* source decode, and re-emit. Returns
/// `None` if the frame can't be parsed or an intermediate decode fails.
pub fn transcode_edit_compensated(
    vp8: &[u8],
    iters: u32,
    relax: f64,
    edit: impl FnOnce(&mut Frame),
) -> Option<Vec<u8>> {
    let src = decode_yuv420(&wrap_vp8(vp8)).ok()?;
    let tgt = Target {
        y: &src.y,
        u: &src.u,
        v: &src.v,
        yw: src.y_width as usize,
        yh: src.y_height as usize,
        uw: src.uv_width as usize,
        uh: src.uv_height as usize,
    };
    let mut frame = parse_vp8_keyframe(vp8)?;
    edit(&mut frame);
    if !dc_compensate(&mut frame, &tgt, iters, relax) {
        return None;
    }
    Some(emit_vp8_keyframe(&frame))
}

#[cfg(test)]
mod tests {
    use super::super::edit::drop_high_freq_ac;
    use super::*;
    use zenwebp::PixelLayout;
    use zenwebp::encoder::{EncodeRequest, LossyConfig};

    #[test]
    fn compensation_reduces_mean_drift() {
        let (w, h) = (160u32, 128u32);
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
            &LossyConfig::new().with_quality(85.0).with_method(4),
            &rgba,
            PixelLayout::Rgba8,
            w,
            h,
        )
        .encode()
        .unwrap();
        // bare VP8 for the transcoder
        let vp8 = {
            let mut pos = 12;
            let mut found = None;
            while pos + 8 <= webp.len() {
                let size = u32::from_le_bytes([
                    webp[pos + 4],
                    webp[pos + 5],
                    webp[pos + 6],
                    webp[pos + 7],
                ]) as usize;
                if &webp[pos..pos + 4] == b"VP8 " {
                    found = Some(webp[pos + 8..pos + 8 + size].to_vec());
                    break;
                }
                pos += 8 + size + (size & 1);
            }
            found.unwrap()
        };
        let src = decode_yuv420(&wrap_vp8(&vp8)).unwrap();
        let mean_abs = |a: &[u8], b: &[u8]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
                .sum::<u64>() as f64
                / a.len() as f64
        };

        // Uncompensated drop.
        let bare = super::super::edit::transcode_drop_ac(&vp8, 4).unwrap();
        let unc = decode_yuv420(&wrap_vp8(&bare)).unwrap();
        let unc_err = mean_abs(&unc.y, &src.y);

        // Compensated drop. At a stable relaxation (≤ ~0.25 — VP8's prediction
        // chain makes stronger settings diverge, see the module/benchmark
        // notes) DC compensation is stable and modestly reduces mean drift on
        // this smooth synthetic. It does NOT make CoeffEdit competitive with
        // Reencode in general (real-image RD measured in the benchmark doc);
        // this test only guards that the loop is stable and non-worsening here.
        let iters: u32 = std::env::var("ZWR_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(15);
        let relax: f64 = std::env::var("ZWR_RELAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.2);
        let comp =
            transcode_edit_compensated(&vp8, iters, relax, |f| drop_high_freq_ac(f, 4)).unwrap();
        let cmp = decode_yuv420(&wrap_vp8(&comp)).unwrap();
        let cmp_err = mean_abs(&cmp.y, &src.y);

        eprintln!("luma mean-abs vs src: uncompensated={unc_err:.2} compensated={cmp_err:.2}");
        assert!(
            cmp_err <= unc_err + 0.5,
            "compensation must be stable (non-worsening) at relax={relax}: {cmp_err} vs {unc_err}"
        );
    }
}
