//! SIMD vs scalar parity for the non-sharp gamma-corrected chroma path.
//!
//! The `convert_image_yuv_rgb_fast` function uses zenyuv for the Y plane
//! (within ±2 of scalar — see `zenyuv_parity.rs`) and the magetypes-generic
//! gamma_chroma_rows kernel for U/V. This test verifies that the SIMD chroma
//! path produces **identical** u8 values to the scalar gamma-corrected path
//! for the same RGB input.

#![cfg(feature = "fast-yuv")]

use zenwebp::test_helpers;

fn deterministic_rgb(w: u32, h: u32, seed0: u64) -> Vec<u8> {
    let n = (w * h * 3) as usize;
    let mut buf = Vec::with_capacity(n);
    let mut s = seed0;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        buf.push((s >> 33) as u8);
    }
    buf
}

#[test]
fn chroma_simd_matches_scalar_exact() {
    // Use an mb-aligned size to exercise the bulk SIMD path heavily.
    for (w, h, seed) in [
        (64u32, 48u32, 0xcafe),
        (160, 120, 0xbeef),
        (400, 300, 0xf00d),
    ] {
        let rgb = deterministic_rgb(w, h, seed);

        let (_, u_scalar, v_scalar) =
            test_helpers::convert_image_yuv_rgb(&rgb, w as u16, h as u16, w as usize);
        let (_, u_fast, v_fast) =
            test_helpers::convert_image_yuv_rgb_fast(&rgb, w as u16, h as u16, w as usize);

        assert_eq!(
            u_scalar.len(),
            u_fast.len(),
            "{w}x{h}: U plane length mismatch"
        );
        assert_eq!(
            v_scalar.len(),
            v_fast.len(),
            "{w}x{h}: V plane length mismatch"
        );
        assert_eq!(
            u_scalar, u_fast,
            "{w}x{h}: U chroma SIMD output diverges from scalar"
        );
        assert_eq!(
            v_scalar, v_fast,
            "{w}x{h}: V chroma SIMD output diverges from scalar"
        );
    }
}

#[test]
fn chroma_simd_odd_dimensions() {
    // Non-mb-aligned widths exercise the scalar tail after the SIMD bulk.
    for (w, h, seed) in [(17u32, 13u32, 0xabc), (35, 21, 0xdef), (97, 103, 0x123)] {
        let rgb = deterministic_rgb(w, h, seed);

        let (_, u_scalar, v_scalar) =
            test_helpers::convert_image_yuv_rgb(&rgb, w as u16, h as u16, w as usize);
        let (_, u_fast, v_fast) =
            test_helpers::convert_image_yuv_rgb_fast(&rgb, w as u16, h as u16, w as usize);

        assert_eq!(u_scalar, u_fast, "{w}x{h}: U mismatch");
        assert_eq!(v_scalar, v_fast, "{w}x{h}: V mismatch");
    }
}

#[test]
fn chroma_simd_small_below_bulk_threshold() {
    // Width < 16 has no SIMD bulk — pure scalar fallback.
    for (w, h, seed) in [(7u32, 7u32, 0x01), (15, 15, 0x02)] {
        let rgb = deterministic_rgb(w, h, seed);
        let (_, u_scalar, v_scalar) =
            test_helpers::convert_image_yuv_rgb(&rgb, w as u16, h as u16, w as usize);
        let (_, u_fast, v_fast) =
            test_helpers::convert_image_yuv_rgb_fast(&rgb, w as u16, h as u16, w as usize);
        assert_eq!(u_scalar, u_fast, "{w}x{h}: U mismatch");
        assert_eq!(v_scalar, v_fast, "{w}x{h}: V mismatch");
    }
}
