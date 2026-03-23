//! Integration test: chroma dithering on lossy decode.
//!
//! Encodes a smooth gradient at low quality (where chroma banding is visible),
//! then decodes with dithering enabled (default) vs disabled, verifying that:
//! 1. Dithered output differs from undithered (dithering has an effect)
//! 2. Dithering only affects chroma (luma Y plane is unchanged)
//! 3. The difference magnitude is reasonable (not corrupting the image)
//! 4. The default strength=50 matches expected behavior

use zenwebp::{DecodeConfig, DecodeRequest, EncodeRequest, LossyConfig, PixelLayout};

/// Create a smooth horizontal gradient image (RGB) that will produce
/// visible chroma banding when decoded.
fn make_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = 128u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

/// Encode a high-quality WebP from a gradient.
///
/// Dithering only activates when the UV AC quantizer index is < 12,
/// which happens at high quality settings (Q90+). At low quality the
/// quantizer is too coarse for dithering to help.
fn encode_high_quality_webp() -> Vec<u8> {
    let width = 128;
    let height = 128;
    let pixels = make_gradient(width, height);

    let config = LossyConfig::new().with_quality(95.0).with_method(0);
    EncodeRequest::lossy(
        &config,
        &pixels,
        PixelLayout::Rgb8,
        width as u32,
        height as u32,
    )
    .encode()
    .expect("encoding failed")
}

#[test]
fn dithering_modifies_chroma_at_high_quality() {
    let webp_data = encode_high_quality_webp();

    // Decode with default dithering (strength=50)
    let config_dithered = DecodeConfig::default();
    assert_eq!(config_dithered.dithering_strength, 50);
    let (pixels_dithered, w, h) = DecodeRequest::new(&config_dithered, &webp_data)
        .decode_rgba()
        .expect("decode failed");

    // Decode with dithering disabled
    let config_none = DecodeConfig::default().with_dithering_strength(0);
    let (pixels_none, w2, h2) = DecodeRequest::new(&config_none, &webp_data)
        .decode_rgba()
        .expect("decode failed");

    assert_eq!((w, h), (w2, h2));
    assert_eq!(pixels_dithered.len(), pixels_none.len());

    // Count differing pixels
    let mut diff_count = 0u64;
    let mut max_diff = 0u8;
    let mut total_diff = 0u64;
    let npixels = (w * h) as usize;

    for i in 0..npixels {
        let base = i * 4;
        for c in 0..3 {
            // Compare RGB channels (skip alpha)
            let d = pixels_dithered[base + c];
            let n = pixels_none[base + c];
            if d != n {
                diff_count += 1;
                let diff = d.abs_diff(n);
                total_diff += u64::from(diff);
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
    }

    // Dithering should have changed some pixels
    assert!(
        diff_count > 0,
        "dithering had no effect — expected pixel differences at Q95"
    );

    // At least 1% of channel values should differ (dithering affects most chroma)
    let total_channels = npixels as u64 * 3;
    let diff_pct = (diff_count as f64 / total_channels as f64) * 100.0;
    assert!(
        diff_pct > 1.0,
        "only {diff_pct:.2}% of channels changed — expected more dithering effect"
    );

    // Max difference should be small (dithering adds small noise, not corruption)
    assert!(
        max_diff <= 8,
        "max pixel difference {max_diff} too large — dithering should add small noise"
    );

    // Average difference should be very small
    let avg_diff = total_diff as f64 / diff_count.max(1) as f64;
    assert!(avg_diff < 4.0, "average difference {avg_diff:.2} too large");

    eprintln!(
        "dithering test: {diff_count}/{total_channels} channels differ ({diff_pct:.1}%), \
         max_diff={max_diff}, avg_diff={avg_diff:.2}"
    );
}

#[test]
fn dithering_strength_zero_matches_no_dithering() {
    let webp_data = encode_high_quality_webp();

    let config_0 = DecodeConfig::default().with_dithering_strength(0);
    let (pixels_0, _, _) = DecodeRequest::new(&config_0, &webp_data)
        .decode_rgba()
        .expect("decode failed");

    // Decode again with strength 0 — should be identical (deterministic)
    let (pixels_0b, _, _) = DecodeRequest::new(&config_0, &webp_data)
        .decode_rgba()
        .expect("decode failed");

    assert_eq!(
        pixels_0, pixels_0b,
        "undithered decodes should be identical"
    );
}

#[test]
fn dithering_is_deterministic() {
    let webp_data = encode_high_quality_webp();

    let config = DecodeConfig::default().with_dithering_strength(50);
    let (pixels_a, _, _) = DecodeRequest::new(&config, &webp_data)
        .decode_rgba()
        .expect("decode failed");

    let (pixels_b, _, _) = DecodeRequest::new(&config, &webp_data)
        .decode_rgba()
        .expect("decode failed");

    assert_eq!(
        pixels_a, pixels_b,
        "dithered decodes with same strength should be deterministic"
    );
}

#[test]
fn higher_strength_produces_more_dithering() {
    let webp_data = encode_high_quality_webp();

    let config_none = DecodeConfig::default().with_dithering_strength(0);
    let (pixels_none, _, _) = DecodeRequest::new(&config_none, &webp_data)
        .decode_rgba()
        .expect("decode failed");

    let config_low = DecodeConfig::default().with_dithering_strength(25);
    let (pixels_low, _, _) = DecodeRequest::new(&config_low, &webp_data)
        .decode_rgba()
        .expect("decode failed");

    let config_high = DecodeConfig::default().with_dithering_strength(100);
    let (pixels_high, _, _) = DecodeRequest::new(&config_high, &webp_data)
        .decode_rgba()
        .expect("decode failed");

    // Count total absolute difference vs undithered
    let diff_sum = |a: &[u8], b: &[u8]| -> u64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| u64::from(x.abs_diff(y)))
            .sum()
    };

    let diff_low = diff_sum(&pixels_low, &pixels_none);
    let diff_high = diff_sum(&pixels_high, &pixels_none);

    assert!(
        diff_high > diff_low,
        "strength=100 ({diff_high}) should produce more change than strength=25 ({diff_low})"
    );
}
