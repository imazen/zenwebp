//! Tests for the v2 VP8 decoder — verifies pixel-identical output to v1.

use zenwebp::{DecodeConfig, DecodeRequest};

/// Encode a test image as lossy WebP, decode with v1 and v2, compare.
fn roundtrip_compare(width: u32, height: u32, rgb: &[u8]) {
    let config = zenwebp::EncoderConfig::new_lossy()
        .with_quality(75.0)
        .with_method(4);
    let webp_data =
        zenwebp::EncodeRequest::new(&config, rgb, zenwebp::PixelLayout::Rgb8, width, height)
            .encode()
            .expect("encode failed");

    // Decode with v1
    let decode_config = DecodeConfig::default();
    let (v1_rgb, v1_w, v1_h) = DecodeRequest::new(&decode_config, &webp_data)
        .decode_rgb()
        .expect("v1 decode failed");

    // Decode with v2
    let (v2_rgb, v2_w, v2_h) = DecodeRequest::new(&decode_config, &webp_data)
        .decode_rgb_v2()
        .expect("v2 decode failed");

    assert_eq!(v1_w, u32::from(v2_w), "width mismatch: v1={v1_w} v2={v2_w}");
    assert_eq!(
        v1_h,
        u32::from(v2_h),
        "height mismatch: v1={v1_h} v2={v2_h}"
    );
    assert_eq!(
        v1_rgb.len(),
        v2_rgb.len(),
        "buffer length mismatch: v1={} v2={}",
        v1_rgb.len(),
        v2_rgb.len()
    );

    // Compare pixel by pixel
    let mut max_diff = 0u8;
    let mut diff_count = 0usize;
    for (i, (&a, &b)) in v1_rgb.iter().zip(v2_rgb.iter()).enumerate() {
        let d = a.abs_diff(b);
        if d > 0 {
            diff_count += 1;
            if d > max_diff {
                max_diff = d;
            }
            if diff_count <= 10 {
                let pixel = i / 3;
                let channel = ["R", "G", "B"][i % 3];
                eprintln!("  diff at pixel {pixel} {channel}: v1={a} v2={b} diff={d}");
            }
        }
    }

    assert_eq!(
        diff_count, 0,
        "v2 differs from v1 at {diff_count} bytes, max diff = {max_diff}"
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
fn v2_matches_v1_small() {
    // 2x2 (single MB, minimal image)
    let rgb = vec![100u8; 2 * 2 * 3];
    roundtrip_compare(2, 2, &rgb);
}

#[test]
fn v2_matches_v1_16x16() {
    let rgb = gradient_rgb(16, 16);
    roundtrip_compare(16, 16, &rgb);
}

#[test]
fn v2_matches_v1_odd_size() {
    // Non-MB-aligned dimensions
    let rgb = gradient_rgb(33, 17);
    roundtrip_compare(33, 17, &rgb);
}

#[test]
fn v2_matches_v1_medium() {
    let rgb = gradient_rgb(128, 96);
    roundtrip_compare(128, 96, &rgb);
}

#[test]
fn v2_matches_v1_photo_size() {
    let rgb = gradient_rgb(512, 384);
    roundtrip_compare(512, 384, &rgb);
}

#[test]
fn v2_matches_v1_uniform() {
    // All-same-color image tests DC-only path
    let rgb = vec![42u8; 64 * 64 * 3];
    roundtrip_compare(64, 64, &rgb);
}
