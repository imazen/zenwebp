//! Verify zenyuv Limited BT.601 output matches zenwebp's scalar rgb_to_y/u/v.
//!
//! Run: `cargo test --release --test zenyuv_parity -- --nocapture`

#[test]
fn zenyuv_vs_zenwebp_scalar_limited_bt601() {
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
