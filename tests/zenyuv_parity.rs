//! Compare two limited-range BT.601 4:4:4 equations on nonlinear RGB.
//! Chroma is co-sited per pixel. The WebP equation uses 16-bit coefficients
//! and half-up rounding. The external zenyuv BT.601 kernels use 15-bit
//! coefficients or f32, with their own rounding.
//!
//! Run: `cargo test --release --test zenyuv_parity -- --nocapture`

#[test]
fn external_zenyuv_vs_webp_fixed_limited_bt601_444() {
    // zenwebp's scalar constants (from decoder/yuv.rs)
    const YUV_FIX: i32 = 16;
    const YUV_HALF: i32 = 1 << (YUV_FIX - 1);

    let mut max_y = 0u8;
    let mut max_u = 0u8;
    let mut max_v = 0u8;
    let mut sum_y = 0u64;
    let mut sum_u = 0u64;
    let mut sum_v = 0u64;
    let mut count = 0u64;

    // Test every 5th value (52×52×52 = 140,608 pixels)
    for r in (0..=255u8).step_by(5) {
        for g in (0..=255u8).step_by(5) {
            for b in (0..=255u8).step_by(5) {
                // zenwebp scalar (16-bit fixed-point, YUV_FIX=16)
                let zy = ((16839 * r as i32
                    + 33059 * g as i32
                    + 6420 * b as i32
                    + (16 << YUV_FIX)
                    + YUV_HALF)
                    >> YUV_FIX) as u8;
                let zu = ((-9719 * r as i32 - 19081 * g as i32
                    + 28800 * b as i32
                    + (128 << YUV_FIX)
                    + YUV_HALF)
                    >> YUV_FIX) as u8;
                let zv = ((28800 * r as i32 - 24116 * g as i32 - 4684 * b as i32
                    + (128 << YUV_FIX)
                    + YUV_HALF)
                    >> YUV_FIX) as u8;

                // zenyuv (15-bit fixed-point / f32, Limited BT.601)
                let rgb = [r, g, b];
                let mut y = [0u8; 1];
                let mut cb = [0u8; 1];
                let mut cr = [0u8; 1];
                let mut ctx =
                    zenyuv::YuvContext::new(zenyuv::Range::Limited, zenyuv::Matrix::Bt601);
                ctx.encode_444_u8(&rgb, &mut y, &mut cb, &mut cr, 1, 1);

                if rgb == [0, 0, 0] || rgb == [255, 255, 255] {
                    let y_anchor = if r == 0 { 16 } else { 235 };
                    assert_eq!([zy, zu, zv], [y_anchor, 128, 128]);
                    assert_eq!([y[0], cb[0], cr[0]], [y_anchor, 128, 128]);
                }

                let dy = zy.abs_diff(y[0]);
                let du = zu.abs_diff(cb[0]);
                let dv = zv.abs_diff(cr[0]);
                max_y = max_y.max(dy);
                max_u = max_u.max(du);
                max_v = max_v.max(dv);
                sum_y += dy as u64;
                sum_u += du as u64;
                sum_v += dv as u64;
                count += 1;

                if dy > 2 || du > 2 || dv > 2 {
                    eprintln!(
                        "R={r} G={g} B={b}: zenwebp Y={zy} U={zu} V={zv}, zenyuv Y={} U={} V={} diff Y={dy} U={du} V={dv}",
                        y[0], cb[0], cr[0]
                    );
                }
            }
        }
    }

    let mean_y = sum_y as f64 / count as f64;
    let mean_u = sum_u as f64 / count as f64;
    let mean_v = sum_v as f64 / count as f64;
    eprintln!("Tested {count} pixels");
    eprintln!("Max diff: Y={max_y} U={max_u} V={max_v}");
    eprintln!("Mean diff: Y={mean_y:.4} U={mean_u:.4} V={mean_v:.4}");
    assert!(max_y <= 2, "Y max diff {max_y} > 2");
    assert!(max_u <= 2, "U max diff {max_u} > 2");
    assert!(max_v <= 2, "V max diff {max_v} > 2");
}

#[test]
fn production_420_is_byte_exact_across_dispatch_tiers_for_odd_and_strided_images() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
    use zenwebp::__test_helpers::convert_image_yuv_rgb_fast;

    for (width, height) in [(1u16, 1u16), (3, 5), (17, 3), (33, 17), (64, 7), (65, 19)] {
        let stride = usize::from(width) + 5;
        let mut rgb = vec![0xe1; stride * usize::from(height) * 3];
        let mut seed = 0x1234_5678u32;
        for row in rgb.chunks_exact_mut(stride * 3) {
            for code in &mut row[..usize::from(width) * 3] {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                *code = (seed >> 24) as u8;
            }
        }
        for exact in [true, false] {
            let baseline = convert_image_yuv_rgb_fast(&rgb, width, height, stride, exact);
            let report = for_each_token_permutation(CompileTimePolicy::Fail, |permutation| {
                let actual = convert_image_yuv_rgb_fast(&rgb, width, height, stride, exact);
                for (name, actual, expected) in [
                    ("Y", &actual.0, &baseline.0),
                    ("U", &actual.1, &baseline.1),
                    ("V", &actual.2, &baseline.2),
                ] {
                    assert_eq!(actual.len(), expected.len());
                    assert!(
                        actual == expected,
                        "{name} {width}x{height} (strict={exact}, disabled {:?}): first differing code {:?}",
                        permutation.disabled,
                        actual
                            .iter()
                            .zip(expected)
                            .enumerate()
                            .find(|(_, (a, b))| a != b)
                    );
                }
            });
            assert!(report.permutations_run > 0);
        }
    }
}
