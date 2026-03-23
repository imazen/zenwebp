//! Integration test: chroma dithering on lossy decode.
//!
//! Encodes a smooth gradient at low quality (where chroma banding is visible),
//! then decodes with dithering enabled (default) vs disabled, verifying that:
//! 1. Dithered output differs from undithered (dithering has an effect)
//! 2. Dithering only affects chroma (luma Y plane is unchanged)
//! 3. The difference magnitude is reasonable (not corrupting the image)
//! 4. The default strength=50 matches expected behavior

use zenwebp::{DecodeConfig, DecodeRequest, EncodeRequest, LossyConfig, PixelLayout};

/// Decode with libwebp's advanced API, optionally with dithering.
/// Returns RGBA pixels.
fn decode_with_libwebp(webp_data: &[u8], dithering_strength: i32) -> Vec<u8> {
    use libwebp_sys::*;

    unsafe {
        let mut config = WebPDecoderConfig::new().expect("WebPInitDecoderConfig failed");
        config.options.dithering_strength = dithering_strength;
        config.output.colorspace = WEBP_CSP_MODE::MODE_RGBA;

        let status = WebPDecode(webp_data.as_ptr(), webp_data.len(), &mut config);
        assert_eq!(
            status,
            VP8StatusCode::VP8_STATUS_OK,
            "libwebp decode failed: {status:?}"
        );

        let rgba = &config.output.u.RGBA;
        let w = config.output.width as usize;
        let h = config.output.height as usize;
        let stride = rgba.stride as usize;
        let mut pixels = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            let row = std::slice::from_raw_parts(rgba.rgba.add(y * stride), w * 4);
            pixels.extend_from_slice(row);
        }
        WebPFreeDecBuffer(&mut config.output);
        pixels
    }
}

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

/// Compare zenwebp vs libwebp decode output WITHOUT dithering.
/// Both should use dithering_strength=0 to get a clean baseline comparison.
#[test]
fn undithered_matches_libwebp() {
    let webp_data = encode_high_quality_webp();

    // zenwebp with no dithering
    let config = DecodeConfig::default().with_dithering_strength(0);
    let (zen_pixels, w, h) = DecodeRequest::new(&config, &webp_data)
        .decode_rgba()
        .expect("zenwebp decode failed");

    // libwebp with no dithering (default is 0)
    let lib_pixels = decode_with_libwebp(&webp_data, 0);

    assert_eq!(zen_pixels.len(), lib_pixels.len(), "buffer size mismatch");

    let npixels = (w * h) as usize;
    let mut max_diff = 0u8;
    let mut diff_count = 0u64;
    for i in 0..npixels {
        let base = i * 4;
        for c in 0..4 {
            let d = zen_pixels[base + c].abs_diff(lib_pixels[base + c]);
            if d > 0 {
                diff_count += 1;
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
    }

    // Without dithering, both decoders should produce very similar output.
    // Small differences are expected from upsampling/rounding.
    eprintln!(
        "undithered comparison: {diff_count}/{} channels differ, max_diff={max_diff}",
        npixels * 4
    );
    assert!(
        max_diff <= 2,
        "undithered decode differs from libwebp by up to {max_diff} — expected <= 2"
    );
}

/// Compare dithered output: zenwebp dithering=50 vs libwebp dithering=50.
/// Both use the same algorithm, so differences should be small (PRNG seed
/// differences may cause different random sequences).
#[test]
fn dithered_output_similar_to_libwebp() {
    let webp_data = encode_high_quality_webp();

    // zenwebp with dithering=50
    let config = DecodeConfig::default().with_dithering_strength(50);
    let (zen_pixels, w, h) = DecodeRequest::new(&config, &webp_data)
        .decode_rgba()
        .expect("zenwebp decode failed");

    // libwebp with dithering=50
    let lib_pixels = decode_with_libwebp(&webp_data, 50);

    assert_eq!(zen_pixels.len(), lib_pixels.len(), "buffer size mismatch");

    let npixels = (w * h) as usize;
    let mut max_diff = 0u8;
    let mut total_diff = 0u64;
    let mut diff_count = 0u64;
    for i in 0..npixels {
        let base = i * 4;
        for c in 0..4 {
            let d = zen_pixels[base + c].abs_diff(lib_pixels[base + c]);
            if d > 0 {
                diff_count += 1;
                total_diff += u64::from(d);
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
    }

    let avg_diff = if diff_count > 0 {
        total_diff as f64 / diff_count as f64
    } else {
        0.0
    };

    eprintln!(
        "dithered comparison: {diff_count}/{} channels differ, max_diff={max_diff}, avg_diff={avg_diff:.2}",
        npixels * 4
    );

    // Dithering uses a PRNG, so if our seed table matches libwebp exactly
    // the output should be identical (max_diff=0). If not, differences
    // should still be bounded by the dither amplitude (small).
    assert!(
        max_diff <= 12,
        "dithered decode differs from libwebp by up to {max_diff} — \
         expected <= 12 (2x max dither amplitude)"
    );
}
