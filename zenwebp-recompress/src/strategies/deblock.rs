//! Edge-preserving post-decode deblocking for the `DeblockReencode`
//! strategy.
//!
//! VP8's in-loop filter mostly handles block boundaries during decode, but
//! at low source quantizer levels (qi ≥ 90) the loop filter alone leaves
//! visible 4×4 and 16×16 artifacts. Re-encoding a noisy source wastes
//! bits encoding the artifacts faithfully. This module applies a light
//! gradient-gated blur at 4-pixel boundaries on the decoded RGBA so the
//! re-encoder sees a cleaner signal.
//!
//! The algorithm:
//!
//! 1. For each vertical boundary `c ∈ {4, 8, 12, …}` (and analogous
//!    horizontal):
//!    a. Compute the luma gradient `|Y(r, c) − Y(r, c-1)|` at the
//!       boundary.
//!    b. If the gradient is below `threshold = base_thresh +
//!       strength * qi_factor`, blend the two pixels across the
//!       boundary with weights `[0.5, 0.5]` for the immediate two
//!       pixels and `[0.75, 0.25]` for the next two out.
//!    c. If above threshold, treat it as a real edge — do nothing.
//! 2. Identical pass for horizontal boundaries.
//!
//! `strength` is a `0..32` integer, calibrated against `source_q` /
//! `quantizer_index`. Higher → more aggressive smoothing.
//!
//! Edge detection uses Rec.709 luma — same coefficient as
//! `zenpixels-convert`'s sRGB→luma. We deliberately ignore alpha here;
//! the alpha channel is passed through untouched.

/// Strength schedule keyed on VP8 quantizer index.
///
/// Below qi 60 the source is high enough quality that deblocking would
/// over-smooth detail. Above qi 110 the artifacts are too severe for a
/// gradient-gated filter to fix without ringing.
pub fn strength_from_quantizer(qi: u8) -> u8 {
    match qi {
        0..=59 => 0,
        60..=79 => (qi - 60) / 2 + 4,    // 4..14
        80..=109 => 14 + (qi - 80) / 3,  // 14..23
        _ => 23,                         // 110+ → cap. VP8 spec says ≤127.
    }
}

const LUMA_R: i32 = 218; // 0.2126 * 1024
const LUMA_G: i32 = 732; // 0.7152 * 1024
const LUMA_B: i32 = 74;  // 0.0722 * 1024

#[inline]
fn luma(rgba: &[u8], idx: usize) -> i32 {
    let r = rgba[idx] as i32;
    let g = rgba[idx + 1] as i32;
    let b = rgba[idx + 2] as i32;
    (r * LUMA_R + g * LUMA_G + b * LUMA_B + 512) >> 10
}

/// Apply edge-preserving deblock at 4-pixel boundaries on an RGBA frame.
///
/// `strength` is `0..32`; 0 is a no-op. The buffer is mutated in place.
pub fn deblock_rgba(rgba: &mut [u8], width: usize, height: usize, strength: u8) {
    if strength == 0 || width < 8 || height < 8 || rgba.len() < width * height * 4 {
        return;
    }
    let base_thresh = 4i32;
    let strength_thresh = strength as i32 * 2;

    // Vertical boundaries (between column c-1 and c, c in {4, 8, …}).
    for r in 0..height {
        let row_off = r * width * 4;
        let mut c = 4;
        while c < width {
            let i_left = row_off + (c - 1) * 4;
            let i_right = row_off + c * 4;
            let yl = luma(rgba, i_left);
            let yr = luma(rgba, i_right);
            let grad = (yr - yl).abs();
            if grad <= base_thresh + strength_thresh {
                blend_boundary(rgba, i_left, i_right);
                if c >= 2 && c + 1 < width {
                    let i_left2 = row_off + (c - 2) * 4;
                    let i_right2 = row_off + (c + 1) * 4;
                    // Slight outer-tap pull toward the boundary midline.
                    quarter_blend(rgba, i_left2, i_left);
                    quarter_blend(rgba, i_right2, i_right);
                }
            }
            c += 4;
        }
    }

    // Horizontal boundaries (between row r-1 and r, r in {4, 8, …}).
    let stride = width * 4;
    let mut r = 4;
    while r < height {
        for c in 0..width {
            let i_up = (r - 1) * stride + c * 4;
            let i_dn = r * stride + c * 4;
            let yu = luma(rgba, i_up);
            let yd = luma(rgba, i_dn);
            let grad = (yd - yu).abs();
            if grad <= base_thresh + strength_thresh {
                blend_boundary(rgba, i_up, i_dn);
                if r >= 2 && r + 1 < height {
                    let i_up2 = (r - 2) * stride + c * 4;
                    let i_dn2 = (r + 1) * stride + c * 4;
                    quarter_blend(rgba, i_up2, i_up);
                    quarter_blend(rgba, i_dn2, i_dn);
                }
            }
        }
        r += 4;
    }
}

/// 50/50 blend of the R/G/B channels (alpha untouched).
#[inline]
fn blend_boundary(rgba: &mut [u8], a: usize, b: usize) {
    for chan in 0..3 {
        let av = rgba[a + chan] as u16;
        let bv = rgba[b + chan] as u16;
        let mid = (av + bv + 1) / 2;
        rgba[a + chan] = mid as u8;
        rgba[b + chan] = mid as u8;
    }
}

/// 75/25 blend toward the boundary tap.
#[inline]
fn quarter_blend(rgba: &mut [u8], outer: usize, inner: usize) {
    for chan in 0..3 {
        let ov = rgba[outer + chan] as u16;
        let iv = rgba[inner + chan] as u16;
        let nv = (ov * 3 + iv + 2) / 4;
        rgba[outer + chan] = nv as u8;
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
    fn deblock_smooths_block_boundary() {
        // 8x8 RGBA buffer: left half = 100, right half = 140 (a 4-block
        // boundary at column 4 with grad=40).
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
        let before = buf.clone();
        deblock_rgba(&mut buf, 8, 8, 20);

        // With strength=20 (qi≈85), strength_thresh=40, base=4 → threshold=44.
        // The boundary gradient (40) is BELOW threshold → blend applied.
        let i_left = (0 * 8 + 3) * 4;
        let i_right = (0 * 8 + 4) * 4;
        let l_after = buf[i_left];
        let r_after = buf[i_right];
        assert_ne!(l_after, before[i_left], "left tap should have changed");
        assert_ne!(r_after, before[i_right], "right tap should have changed");
        // 50/50 blend brings both toward 120.
        assert_eq!(l_after, 120);
        assert_eq!(r_after, 120);
    }

    #[test]
    fn deblock_preserves_real_edges() {
        // Same 8x8 but with grad=80 (above any reasonable threshold).
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
        // Real edge — should be untouched.
        assert_eq!(buf, before);
    }
}
