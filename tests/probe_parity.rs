//! Probe-vs-decode parity tests.
//!
//! Verifies that lightweight header probing (`detect::probe` and `ImageInfo::from_webp`)
//! produces metadata consistent with full decode. This catches divergence between
//! the quick header scanner and the full parser.

use zenwebp::detect::{BitstreamType, probe};
use zenwebp::{EncodeRequest, EncoderConfig, ImageInfo, PixelLayout};

/// Encode an opaque RGB image, then verify probe and decode metadata agree.
#[test]
fn probe_vs_decode_lossy_rgb() {
    let (w, h) = (64, 48);
    let pixels: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
    let cfg = EncoderConfig::new_lossy().with_quality(75.0);
    let webp = EncodeRequest::new(&cfg, &pixels, PixelLayout::Rgb8, w as u32, h as u32)
        .encode()
        .expect("encode failed");

    assert_probe_matches_decode(&webp, w as u32, h as u32, false, false, false);
}

/// Encode an RGBA image with alpha, then verify probe and decode metadata agree.
#[test]
fn probe_vs_decode_lossy_rgba() {
    let (w, h) = (32, 32);
    let pixels: Vec<u8> = (0..w * h * 4)
        .map(|i| if i % 4 == 3 { 200 } else { (i % 223) as u8 })
        .collect();
    let cfg = EncoderConfig::new_lossy().with_quality(80.0);
    let webp = EncodeRequest::new(&cfg, &pixels, PixelLayout::Rgba8, w as u32, h as u32)
        .encode()
        .expect("encode failed");

    assert_probe_matches_decode(&webp, w as u32, h as u32, true, false, false);
}

/// Encode a lossless image, then verify probe and decode metadata agree.
#[test]
fn probe_vs_decode_lossless() {
    let (w, h) = (16, 16);
    let pixels: Vec<u8> = (0..w * h * 3).map(|i| (i % 179) as u8).collect();
    let cfg = EncoderConfig::new_lossless();
    let webp = EncodeRequest::new(&cfg, &pixels, PixelLayout::Rgb8, w as u32, h as u32)
        .encode()
        .expect("encode failed");

    assert_probe_matches_decode(&webp, w as u32, h as u32, false, true, false);
}

/// Encode a lossless RGBA image, then verify probe and decode metadata agree.
#[test]
fn probe_vs_decode_lossless_rgba() {
    let (w, h) = (24, 24);
    let pixels: Vec<u8> = (0..w * h * 4)
        .map(|i| if i % 4 == 3 { 128 } else { (i * 7 % 256) as u8 })
        .collect();
    let cfg = EncoderConfig::new_lossless();
    let webp = EncodeRequest::new(&cfg, &pixels, PixelLayout::Rgba8, w as u32, h as u32)
        .encode()
        .expect("encode failed");

    assert_probe_matches_decode(&webp, w as u32, h as u32, true, true, false);
}

/// Core assertion: probe metadata must match decode metadata.
///
/// Checks dimensions, alpha, animation, lossy/lossless, and ICC profile
/// presence across three independent paths:
///   1. `detect::probe()` — lightweight header-only scanner
///   2. `ImageInfo::from_webp()` — full parser without pixel decode
///   3. Actual decoded pixel dimensions
fn assert_probe_matches_decode(
    webp: &[u8],
    expected_w: u32,
    expected_h: u32,
    expect_alpha: bool,
    expect_lossless: bool,
    expect_icc: bool,
) {
    // Path 1: lightweight probe
    let probed = probe(webp).expect("probe failed");

    // Path 2: full-parser metadata
    let info = ImageInfo::from_webp(webp).expect("ImageInfo::from_webp failed");

    // Path 3: actual decode dimensions
    let (decoded, dec_w, dec_h) = if expect_alpha {
        zenwebp::oneshot::decode_rgba(webp).expect("decode_rgba failed")
    } else {
        zenwebp::oneshot::decode_rgb(webp).expect("decode_rgb failed")
    };
    let bpp = if expect_alpha { 4 } else { 3 };
    assert_eq!(
        decoded.len(),
        (dec_w * dec_h) as usize * bpp,
        "decoded buffer size mismatch"
    );

    // --- Dimensions ---
    assert_eq!(probed.width, expected_w, "probe width");
    assert_eq!(probed.height, expected_h, "probe height");
    assert_eq!(info.width, expected_w, "ImageInfo width");
    assert_eq!(info.height, expected_h, "ImageInfo height");
    assert_eq!(dec_w, expected_w, "decoded width");
    assert_eq!(dec_h, expected_h, "decoded height");

    // --- Alpha ---
    // For lossy RGB (no alpha channel), the probe's VP8X extended header might
    // not be present and has_alpha depends on the container format. We check
    // that probe and ImageInfo agree with each other.
    assert_eq!(
        probed.has_alpha, info.has_alpha,
        "probe vs ImageInfo alpha mismatch: probe={}, info={}",
        probed.has_alpha, info.has_alpha
    );
    if expect_alpha {
        assert!(info.has_alpha, "expected alpha but ImageInfo says no alpha");
    }

    // --- Animation ---
    assert!(!probed.has_animation, "probe: unexpected animation");
    assert!(!info.has_animation, "ImageInfo: unexpected animation");
    assert_eq!(probed.frame_count, 1, "probe: expected 1 frame");
    assert_eq!(info.frame_count, 1, "ImageInfo: expected 1 frame");

    // --- Lossy/lossless ---
    let probe_is_lossless = matches!(probed.bitstream, BitstreamType::Lossless);
    assert_eq!(
        probe_is_lossless, expect_lossless,
        "probe lossy/lossless mismatch"
    );
    assert_eq!(info.is_lossy, !expect_lossless, "ImageInfo is_lossy mismatch");

    // --- ICC profile ---
    assert_eq!(
        probed.icc_profile.is_some(),
        expect_icc,
        "probe ICC presence"
    );
    assert_eq!(
        info.icc_profile.is_some(),
        expect_icc,
        "ImageInfo ICC presence"
    );

    // --- Cross-check: probe and ImageInfo ICC content must match ---
    assert_eq!(
        probed.icc_profile, info.icc_profile,
        "probe vs ImageInfo ICC content mismatch"
    );
}

/// Encode with an embedded ICC profile, verify both probe paths see it.
#[test]
fn probe_vs_decode_with_icc() {
    let (w, h) = (8, 8);
    let pixels: Vec<u8> = vec![128u8; w * h * 3];
    // Minimal ICC profile header (enough to be recognized as present)
    let fake_icc = vec![0u8; 128];
    let cfg = EncoderConfig::new_lossy().with_quality(50.0);
    let webp = EncodeRequest::new(&cfg, &pixels, PixelLayout::Rgb8, w as u32, h as u32)
        .with_icc_profile(&fake_icc)
        .encode()
        .expect("encode with ICC failed");

    assert_probe_matches_decode(&webp, w as u32, h as u32, false, false, true);
}

/// Odd dimensions: verify probe and decode agree on non-multiple-of-16 sizes.
#[test]
fn probe_vs_decode_odd_dimensions() {
    for (w, h) in [(7, 13), (1, 1), (3, 5), (100, 1), (1, 100)] {
        let pixels: Vec<u8> = (0..w * h * 3).map(|i| (i % 199) as u8).collect();
        let cfg = EncoderConfig::new_lossy().with_quality(60.0);
        let webp = EncodeRequest::new(&cfg, &pixels, PixelLayout::Rgb8, w as u32, h as u32)
            .encode()
            .unwrap_or_else(|e| panic!("encode {w}x{h} failed: {e}"));

        assert_probe_matches_decode(&webp, w as u32, h as u32, false, false, false);
    }
}
