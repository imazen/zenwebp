//! Block-artifact-aware post-decode deblocking — an `expert` building
//! block. (The `DeblockReencode` strategy that used it is measured-
//! dominated by plain `Reencode`; see
//! `benchmarks/deblock_experiment_2026-05-28.md`. The filter itself is
//! correct and kept for callers who want it directly.)
//!
//! The decision is the H.264/VP8-style separation of a *blocking artifact*
//! from a *real edge*, evaluated per 4-pixel grid boundary on the decoded
//! RGBA. Edge detection uses Rec.709 luma; the alpha channel is passed
//! through untouched. See [`deblock_rgba`] for the precise gate and
//! smoothing.

/// Strength schedule keyed on VP8 quantizer index.
///
/// Below qi 60 the source is high enough quality that deblocking would
/// over-smooth detail. Above qi 110 the artifacts are too severe for a
/// gradient-gated filter to fix without ringing.
pub fn strength_from_quantizer(qi: u8) -> u8 {
    match qi {
        0..=59 => 0,
        60..=79 => (qi - 60) / 2 + 4,   // 4..14
        80..=109 => 14 + (qi - 80) / 3, // 14..23
        _ => 23,                        // 110+ → cap. VP8 spec says ≤127.
    }
}

const LUMA_R: i32 = 218; // 0.2126 * 1024
const LUMA_G: i32 = 732; // 0.7152 * 1024
const LUMA_B: i32 = 74; // 0.0722 * 1024

#[inline]
fn luma(rgba: &[u8], idx: usize) -> i32 {
    let r = rgba[idx] as i32;
    let g = rgba[idx + 1] as i32;
    let b = rgba[idx + 2] as i32;
    (r * LUMA_R + g * LUMA_G + b * LUMA_B + 512) >> 10
}

/// Apply block-artifact-aware deblock at 4-pixel boundaries on an RGBA
/// frame.
///
/// The decision is the H.264/VP8-style separation of *blocking artifacts*
/// from *real edges*. For each 4-pixel grid boundary with neighborhood
/// `… p1 p0 | q0 q1 …`:
///
/// - `d_boundary = |p0 − q0|` — the step at the grid line.
/// - `d_p = |p1 − p0|`, `d_q = |q1 − q0|` — interior gradients either side.
///
/// A blocking artifact is a *moderate* step at the boundary flanked by
/// *flatter* interiors: `0 < d_boundary ≤ edge_limit` AND
/// `d_p < d_boundary` AND `d_q < d_boundary`. When detected, each boundary
/// pixel is pulled 1/4 of the gap toward the other (a half-reduction of the
/// step), which removes the artifact without flattening real texture.
///
/// `strength` is `0..32`; 0 is a no-op. `edge_limit = base + strength`
/// scales the moderate-step ceiling with source quantization. The buffer is
/// mutated in place; alpha is untouched.
pub fn deblock_rgba(rgba: &mut [u8], width: usize, height: usize, strength: u8) {
    if strength == 0 || width < 8 || height < 8 || rgba.len() < width * height * 4 {
        return;
    }
    let edge_limit = 3i32 + strength as i32 * 3;
    let stride = width * 4;

    // Vertical boundaries (between column c-1 and c, c in {4, 8, …}).
    // Needs c-2 and c+1 in range → c in [4, width-2].
    for r in 0..height {
        let row_off = r * stride;
        let mut c = 4;
        while c + 1 < width {
            let i_p1 = row_off + (c - 2) * 4;
            let i_p0 = row_off + (c - 1) * 4;
            let i_q0 = row_off + c * 4;
            let i_q1 = row_off + (c + 1) * 4;
            if is_block_artifact(rgba, i_p1, i_p0, i_q0, i_q1, edge_limit) {
                reduce_step(rgba, i_p0, i_q0);
            }
            c += 4;
        }
    }

    // Horizontal boundaries (between row r-1 and r, r in {4, 8, …}).
    let mut r = 4;
    while r + 1 < height {
        for c in 0..width {
            let i_p1 = (r - 2) * stride + c * 4;
            let i_p0 = (r - 1) * stride + c * 4;
            let i_q0 = r * stride + c * 4;
            let i_q1 = (r + 1) * stride + c * 4;
            if is_block_artifact(rgba, i_p1, i_p0, i_q0, i_q1, edge_limit) {
                reduce_step(rgba, i_p0, i_q0);
            }
        }
        r += 4;
    }
}

/// True if the boundary between `p0` and `q0` is a blocking artifact (a
/// moderate step flanked by flatter interiors), not a real edge.
#[inline]
fn is_block_artifact(
    rgba: &[u8],
    i_p1: usize,
    i_p0: usize,
    i_q0: usize,
    i_q1: usize,
    edge_limit: i32,
) -> bool {
    let p1 = luma(rgba, i_p1);
    let p0 = luma(rgba, i_p0);
    let q0 = luma(rgba, i_q0);
    let q1 = luma(rgba, i_q1);
    let d_boundary = (p0 - q0).abs();
    if d_boundary == 0 || d_boundary > edge_limit {
        // No step, or too large to be anything but a real edge.
        return false;
    }
    let d_p = (p1 - p0).abs();
    let d_q = (q1 - q0).abs();
    // The step concentrates AT the grid line — interiors are flatter.
    d_p < d_boundary && d_q < d_boundary
}

/// Pull `p0` and `q0` each 1/4 of the gap toward the other, halving the
/// boundary step. Applied per RGB channel; alpha untouched.
#[inline]
fn reduce_step(rgba: &mut [u8], i_p0: usize, i_q0: usize) {
    for chan in 0..3 {
        let p = rgba[i_p0 + chan] as i32;
        let q = rgba[i_q0 + chan] as i32;
        let delta = (q - p) / 4;
        rgba[i_p0 + chan] = (p + delta).clamp(0, 255) as u8;
        rgba[i_q0 + chan] = (q - delta).clamp(0, 255) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strength_monotone_in_qi() {
        let s50 = strength_from_quantizer(50);
        let s80 = strength_from_quantizer(80);
        let s120 = strength_from_quantizer(120);
        assert!(s50 <= s80);
        assert!(s80 <= s120);
        assert_eq!(s50, 0); // high-quality source: no deblock
    }

    #[test]
    fn deblock_zero_strength_is_noop() {
        let original = vec![128u8; 16 * 16 * 4];
        let mut buf = original.clone();
        deblock_rgba(&mut buf, 16, 16, 0);
        assert_eq!(buf, original);
    }

    #[test]
    fn deblock_reduces_flat_flat_block_step() {
        // 8x8 RGBA: left half = 100, right half = 140. Boundary at col 4
        // has d_boundary=40, interiors perfectly flat (d_p=d_q=0) — the
        // textbook blocking-artifact signature.
        let mut buf = vec![0u8; 8 * 8 * 4];
        for r in 0..8 {
            for c in 0..8 {
                let i = (r * 8 + c) * 4;
                let v = if c < 4 { 100 } else { 140 };
                buf[i] = v;
                buf[i + 1] = v;
                buf[i + 2] = v;
                buf[i + 3] = 255;
            }
        }
        deblock_rgba(&mut buf, 8, 8, 20);

        // strength=20 → edge_limit = 3 + 60 = 63. d_boundary=40 ≤ 63 and
        // both interiors (0) < 40 → artifact. reduce_step: delta=(140-100)/4=10.
        let i_left = 3 * 4; // row 0, p0 (col 3)
        let i_right = 4 * 4; // row 0, q0 (col 4)
        assert_eq!(buf[i_left], 110, "p0 pulled +10 toward q0");
        assert_eq!(buf[i_right], 130, "q0 pulled -10 toward p0");
        // Step halved from 40 to 20.
        assert_eq!((buf[i_right] as i32 - buf[i_left] as i32), 20);
        // Alpha untouched.
        assert_eq!(buf[i_left + 3], 255);
    }

    #[test]
    fn deblock_preserves_strong_edge() {
        // d_boundary=150 — far above edge_limit; a real edge, not an
        // artifact. Untouched.
        let mut buf = vec![0u8; 8 * 8 * 4];
        for r in 0..8 {
            for c in 0..8 {
                let i = (r * 8 + c) * 4;
                let v = if c < 4 { 50 } else { 200 };
                buf[i] = v;
                buf[i + 1] = v;
                buf[i + 2] = v;
                buf[i + 3] = 255;
            }
        }
        let before = buf.clone();
        deblock_rgba(&mut buf, 8, 8, 20);
        assert_eq!(buf, before, "strong edge must be preserved");
    }

    #[test]
    fn deblock_preserves_textured_boundary() {
        // A moderate boundary step (30) but with textured interiors that
        // are ALSO stepping by >= 30 — this is real texture/edge, not a
        // block artifact. Must be preserved.
        let mut buf = vec![0u8; 8 * 8 * 4];
        for r in 0..8 {
            for c in 0..8 {
                let i = (r * 8 + c) * 4;
                // Ramp across the whole row: each column +35. Interior
                // gradients (35) exceed the boundary step, so no column
                // boundary looks like an isolated grid artifact.
                let v = (40 + c as i32 * 35).min(255) as u8;
                buf[i] = v;
                buf[i + 1] = v;
                buf[i + 2] = v;
                buf[i + 3] = 255;
            }
        }
        let before = buf.clone();
        deblock_rgba(&mut buf, 8, 8, 20);
        assert_eq!(
            buf, before,
            "uniform ramp (interior grad >= boundary) must be preserved"
        );
    }
}
