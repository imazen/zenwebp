//! Production targets must report the adopted revision-7 A score.
#![cfg(feature = "target-zensim")]
use zensim_a::{RgbSlice, Zensim, ZensimProfile};
use zenwebp::{DecodeConfig, DecodeRequest, EncodeRequest, LossyConfig, PixelLayout, ZensimTarget};

#[test]
fn returned_target_score_is_revision7_a() {
    let (w, h) = (96, 64);
    let pixels: Vec<[u8; 3]> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            [(x * 2) as u8, (y * 3) as u8, ((x ^ y) * 2) as u8]
        })
        .collect();
    let config = LossyConfig::new()
        .with_method(4)
        .with_target_zensim(ZensimTarget::new(60.0).with_max_passes(3));
    let (bytes, metrics) = EncodeRequest::lossy(
        &config,
        bytemuck::cast_slice(&pixels),
        PixelLayout::Rgb8,
        w,
        h,
    )
    .encode_with_metrics()
    .unwrap();
    let (decoded, dw, dh) = DecodeRequest::new(&DecodeConfig::default(), &bytes)
        .decode_rgb()
        .unwrap();
    assert_eq!((dw, dh), (w, h));
    assert_eq!(ZensimProfile::codec_target(), ZensimProfile::A);
    let metric = Zensim::new(ZensimProfile::A);
    let score = metric
        .compute(
            &RgbSlice::new(&pixels, w as usize, h as usize),
            &RgbSlice::new(bytemuck::cast_slice(&decoded), w as usize, h as usize),
        )
        .unwrap()
        .score() as f32;
    assert!(
        (score - metrics.achieved_score).abs() < 0.0001,
        "reported {}, revision-7 A {score}",
        metrics.achieved_score
    );
}

#[test]
#[cfg(feature = "__expert")]
fn color_boundary_quality_reachability() {
    let (w, h) = (256, 256);
    let pixels: Vec<[u8; 3]> = (0..w * h)
        .map(|i| match (i / w < h / 2, i % w < w / 2) {
            (true, true) => [220, 50, 50],
            (true, false) => [60, 200, 80],
            (false, true) => [50, 80, 220],
            (false, false) => [200, 200, 60],
        })
        .collect();
    let metric = Zensim::new(ZensimProfile::A);
    let source = RgbSlice::new(&pixels, w as usize, h as usize);
    let pre = metric.precompute_reference(&source).unwrap();
    let old = zensim::Zensim::new(zensim::ZensimProfile::latest());
    let old_pre = old
        .precompute_reference(&zensim::RgbSlice::new(&pixels, w as usize, h as usize))
        .unwrap();
    for sharp in [false, true] {
        for q in (0..=100).step_by(5) {
            let config = LossyConfig::new()
                .with_method(4)
                .with_quality(q as f32)
                .with_sharp_yuv(sharp);
            let bytes = EncodeRequest::lossy(
                &config,
                bytemuck::cast_slice(&pixels),
                PixelLayout::Rgb8,
                w,
                h,
            )
            .encode()
            .unwrap();
            let (decoded, dw, dh) = DecodeRequest::new(&DecodeConfig::default(), &bytes)
                .decode_rgb()
                .unwrap();
            assert_eq!((dw, dh), (w, h));
            let decoded: &[[u8; 3]] = bytemuck::cast_slice(&decoded);
            let a = metric
                .compute_with_ref(&pre, &RgbSlice::new(decoded, w as usize, h as usize))
                .unwrap()
                .score();
            let legacy = old
                .compute_with_ref(
                    &old_pre,
                    &zensim::RgbSlice::new(decoded, w as usize, h as usize),
                )
                .unwrap()
                .score();
            assert!(a.is_finite() && legacy.is_finite());
            eprintln!(
                "COLOR_BOUNDARY sharp={sharp} q={q} bytes={} A={a:.6} legacy={legacy:.6}",
                bytes.len()
            );
        }
    }
}

#[test]
#[cfg(feature = "__expert")]
fn explicit_sharp_yuv_target_and_strict_unreachable_floor() {
    let (w, h) = (256, 256);
    let pixels: Vec<[u8; 3]> = (0..w * h)
        .map(|i| match (i / w < h / 2, i % w < w / 2) {
            (true, true) => [220, 50, 50],
            (true, false) => [60, 200, 80],
            (false, true) => [50, 80, 220],
            (false, false) => [200, 200, 60],
        })
        .collect();
    let target = ZensimTarget::new(82.0)
        .with_max_overshoot(Some(2.0))
        .with_max_undershoot(Some(0.5))
        .with_max_passes(6);
    let sharp = LossyConfig::new()
        .with_method(4)
        .with_sharp_yuv(true)
        .with_target_zensim(target);
    let (_, metrics) = EncodeRequest::lossy(
        &sharp,
        bytemuck::cast_slice(&pixels),
        PixelLayout::Rgb8,
        w,
        h,
    )
    .encode_with_metrics()
    .unwrap();
    assert!(metrics.targets_met, "{metrics:?}");
    assert!((81.5..=84.0).contains(&metrics.achieved_score));
    assert!(metrics.passes_used <= 6);
    let ordinary = LossyConfig::new().with_method(4).with_target_zensim(target);
    let error = match EncodeRequest::lossy(
        &ordinary,
        bytemuck::cast_slice(&pixels),
        PixelLayout::Rgb8,
        w,
        h,
    )
    .encode_with_metrics()
    {
        Err(error) => error,
        Ok(_) => panic!("strict unreachable A target must not ship"),
    };
    assert!(
        matches!(error.error(), zenwebp::EncodeError::InvalidBufferSize(message) if message.contains("below floor"))
    );
}
