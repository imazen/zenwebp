//! Roundtrip tests for the VP8 lossy decoder — encode then decode and verify.

use zenwebp::{DecodeConfig, DecodeRequest};

/// Encode a test image as lossy WebP, decode, verify dimensions and pixel sanity.
fn roundtrip_check(width: u32, height: u32, rgb: &[u8]) {
    let config = zenwebp::EncoderConfig::new_lossy()
        .with_quality(75.0)
        .with_method(4);
    let webp_data =
        zenwebp::EncodeRequest::new(&config, rgb, zenwebp::PixelLayout::Rgb8, width, height)
            .encode()
            .expect("encode failed");

    let decode_config = DecodeConfig::default();
    let (decoded, dec_w, dec_h) = DecodeRequest::new(&decode_config, &webp_data)
        .decode_rgb()
        .expect("decode failed");

    assert_eq!(dec_w, width, "width mismatch");
    assert_eq!(dec_h, height, "height mismatch");
    assert_eq!(
        decoded.len(),
        (width as usize) * (height as usize) * 3,
        "buffer length mismatch"
    );

    // Lossy codec: pixels won't match exactly, but should be within reason.
    // Check that average error is small (< 10 per channel for Q75).
    let total_error: u64 = rgb
        .iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| u64::from(a.abs_diff(b)))
        .sum();
    let avg_error = total_error as f64 / rgb.len() as f64;
    assert!(
        avg_error < 10.0,
        "average per-channel error too high: {avg_error:.2}"
    );
}

/// Generate a gradient image.
fn gradient_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = (((x + y) * 128) / (w + h).max(1)) as u8;
            rgb.extend_from_slice(&[r, g, b]);
        }
    }
    rgb
}

#[test]
fn roundtrip_small() {
    let rgb = vec![100u8; 2 * 2 * 3];
    roundtrip_check(2, 2, &rgb);
}

#[test]
fn roundtrip_16x16() {
    let rgb = gradient_rgb(16, 16);
    roundtrip_check(16, 16, &rgb);
}

#[test]
fn roundtrip_odd_size() {
    let rgb = gradient_rgb(33, 17);
    roundtrip_check(33, 17, &rgb);
}

#[test]
fn roundtrip_medium() {
    let rgb = gradient_rgb(128, 96);
    roundtrip_check(128, 96, &rgb);
}

#[test]
fn roundtrip_photo_size() {
    let rgb = gradient_rgb(512, 384);
    roundtrip_check(512, 384, &rgb);
}

#[test]
fn roundtrip_uniform() {
    let rgb = vec![42u8; 64 * 64 * 3];
    roundtrip_check(64, 64, &rgb);
}
