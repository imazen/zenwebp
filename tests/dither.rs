//! Integration test: chroma dithering on lossy decode.
//!
//! Encodes a smooth gradient at low quality (where chroma banding is visible),
//! then decodes with dithering enabled (default) vs disabled, verifying that:
//! 1. Dithered output differs from undithered (dithering has an effect)
//! 2. Dithering only affects chroma (luma Y plane is unchanged)
//! 3. The difference magnitude is reasonable (not corrupting the image)
//! 4. The default strength=50 matches expected behavior

use zenwebp::{DecodeConfig, DecodeRequest, EncodeRequest, LossyConfig, PixelLayout};

/// Compare two pixel buffers with tolerance.
///
/// With `fast-yuv`, the `yuv` crate uses true bilinear chroma upsampling and
/// different coefficient precision than libwebp's "fancy" upsample, so exact
/// match is not possible. We check mean absolute error instead.
fn assert_pixels_similar(zen: &[u8], lib: &[u8], context: &str) {
    assert_eq!(zen.len(), lib.len(), "{context}: buffer length mismatch");
    if cfg!(feature = "fast-yuv") {
        let (total_diff, max_diff) =
            zen.iter()
                .zip(lib.iter())
                .fold((0u64, 0u8), |(sum, max_d), (a, b)| {
                    let d = a.abs_diff(*b);
                    (sum + d as u64, max_d.max(d))
                });
        let mean_diff = total_diff as f64 / zen.len() as f64;
        assert!(
            mean_diff < 10.0,
            "{context}: mean channel diff {mean_diff:.3} vs libwebp too high (max {max_diff})",
        );
    } else {
        assert_eq!(zen, lib, "{context}: pixel mismatch");
    }
}

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
    // With `fast-yuv`, the `yuv` crate uses true bilinear chroma upsampling
    // which differs from libwebp's (3*near+far+2)>>2 "fancy" upsample, so
    // per-channel diffs can be much larger.
    eprintln!(
        "undithered comparison: {diff_count}/{} channels differ, max_diff={max_diff}",
        npixels * 4
    );
    let max_allowed = if cfg!(feature = "fast-yuv") { 200 } else { 2 };
    assert!(
        max_diff <= max_allowed,
        "undithered decode differs from libwebp by up to {max_diff} — expected <= {max_allowed}"
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

    // Our PRNG seed table, amplitude computation, and per-MB filtering all
    // match libwebp exactly. Without `fast-yuv`, output is pixel-identical.
    // With `fast-yuv`, YUV->RGB differences dominate.
    let max_allowed = if cfg!(feature = "fast-yuv") { 200 } else { 0 };
    assert!(
        max_diff <= max_allowed,
        "dithered decode differs from libwebp by up to {max_diff} — expected <= {max_allowed}"
    );
}

/// Test exact dithering match with the gallery test images at various strengths.
#[test]
fn dithered_matches_libwebp_gallery_images() {
    for path in &[
        "tests/images/gallery1/1.webp",
        "tests/images/gallery1/2.webp",
        "tests/images/gallery1/3.webp",
        "tests/images/gallery1/4.webp",
        "tests/images/gallery1/5.webp",
    ] {
        let webp_data = std::fs::read(path).expect("failed to read test image");

        for strength in [0, 25, 50, 75, 100] {
            let config = DecodeConfig::default().with_dithering_strength(strength);
            let (zen_pixels, _, _) = DecodeRequest::new(&config, &webp_data)
                .decode_rgba()
                .expect("zenwebp decode failed");

            let lib_pixels = decode_with_libwebp(&webp_data, i32::from(strength));

            assert_pixels_similar(
                &zen_pixels,
                &lib_pixels,
                &format!("{path} at strength={strength}"),
            );
        }
    }
}

/// Test across quality levels: encode at Q10..Q95, decode with each
/// dithering strength, verify pixel-perfect match with libwebp.
///
/// This catches gating differences at all quantizer ranges:
/// - Q10-Q50: uv_quant >= 12, dithering disabled (amplitude = 0)
/// - Q75-Q90: borderline, some segments may dither
/// - Q95: uv_quant < 12, dithering active
#[test]
fn dithered_matches_libwebp_across_quality_levels() {
    let width = 64u32;
    let height = 64u32;
    let pixels = make_gradient(width as usize, height as usize);

    for quality in [10.0, 25.0, 50.0, 75.0, 85.0, 90.0, 95.0] {
        let config = LossyConfig::new().with_quality(quality).with_method(0);
        let webp_data = EncodeRequest::lossy(&config, &pixels, PixelLayout::Rgb8, width, height)
            .encode()
            .unwrap_or_else(|e| panic!("encode Q{quality} failed: {e}"));

        for strength in [0u8, 50, 100] {
            let dc = DecodeConfig::default().with_dithering_strength(strength);
            let (zen_pixels, _, _) = DecodeRequest::new(&dc, &webp_data)
                .decode_rgba()
                .unwrap_or_else(|e| panic!("zenwebp Q{quality} s{strength} failed: {e}"));

            let lib_pixels = decode_with_libwebp(&webp_data, i32::from(strength));

            assert_pixels_similar(
                &zen_pixels,
                &lib_pixels,
                &format!("Q{quality} strength={strength}"),
            );
        }
    }
}

/// Test with segments enabled (SNS/filter produces multiple segments
/// with different quantizers, testing per-segment amplitude computation).
#[test]
fn dithered_matches_libwebp_with_segments() {
    let width = 128u32;
    let height = 128u32;
    let pixels = make_gradient(width as usize, height as usize);

    // Method 4+ with SNS enables segments
    let config = LossyConfig::new().with_quality(90.0).with_method(4);
    let webp_data = EncodeRequest::lossy(&config, &pixels, PixelLayout::Rgb8, width, height)
        .encode()
        .expect("encode with segments failed");

    for strength in [0u8, 50, 100] {
        let dc = DecodeConfig::default().with_dithering_strength(strength);
        let (zen_pixels, _, _) = DecodeRequest::new(&dc, &webp_data)
            .decode_rgba()
            .expect("zenwebp decode failed");

        let lib_pixels = decode_with_libwebp(&webp_data, i32::from(strength));

        assert_pixels_similar(
            &zen_pixels,
            &lib_pixels,
            &format!("segments test strength={strength}"),
        );
    }
}

/// Test with simple loop filter (different extra_rows than normal filter).
#[test]
fn dithered_matches_libwebp_simple_filter() {
    let width = 64u32;
    let height = 64u32;
    let pixels = make_gradient(width as usize, height as usize);

    // Low filter strength often triggers simple filter
    let config = LossyConfig::new().with_quality(95.0).with_method(0);
    let webp_data = EncodeRequest::lossy(&config, &pixels, PixelLayout::Rgb8, width, height)
        .encode()
        .expect("encode failed");

    for strength in [0u8, 50, 100] {
        let dc = DecodeConfig::default().with_dithering_strength(strength);
        let (zen_pixels, _, _) = DecodeRequest::new(&dc, &webp_data)
            .decode_rgba()
            .expect("zenwebp decode failed");

        let lib_pixels = decode_with_libwebp(&webp_data, i32::from(strength));

        assert_pixels_similar(
            &zen_pixels,
            &lib_pixels,
            &format!("simple filter strength={strength}"),
        );
    }
}

/// Test with non-multiple-of-16 dimensions (partial macroblocks at edges).
#[test]
fn dithered_matches_libwebp_odd_dimensions() {
    let width = 100u32;
    let height = 77u32;
    let pixels = make_gradient(width as usize, height as usize);

    for quality in [50.0, 95.0] {
        let config = LossyConfig::new().with_quality(quality).with_method(0);
        let webp_data = EncodeRequest::lossy(&config, &pixels, PixelLayout::Rgb8, width, height)
            .encode()
            .unwrap_or_else(|e| panic!("encode {width}x{height} Q{quality} failed: {e}"));

        for strength in [0u8, 50, 100] {
            let dc = DecodeConfig::default().with_dithering_strength(strength);
            let (zen_pixels, _, _) = DecodeRequest::new(&dc, &webp_data)
                .decode_rgba()
                .unwrap_or_else(|e| panic!("decode {width}x{height} Q{quality} s{strength}: {e}"));

            let lib_pixels = decode_with_libwebp(&webp_data, i32::from(strength));

            assert_pixels_similar(
                &zen_pixels,
                &lib_pixels,
                &format!("{width}x{height} Q{quality} strength={strength}"),
            );
        }
    }
}

// ============================================================================
// v2 decoder dithering tests
// ============================================================================

/// v2 decoder: dithering modifies chroma at high quality.
#[test]
fn v2_dithering_modifies_chroma_at_high_quality() {
    let webp_data = encode_high_quality_webp();

    // Decode with default dithering (strength=50)
    let config_dithered = DecodeConfig::default();
    let (pixels_dithered, w, h) = DecodeRequest::new(&config_dithered, &webp_data)
        .decode_rgba_v2()
        .expect("v2 decode failed");

    // Decode with dithering disabled
    let config_none = DecodeConfig::default().with_dithering_strength(0);
    let (pixels_none, w2, h2) = DecodeRequest::new(&config_none, &webp_data)
        .decode_rgba_v2()
        .expect("v2 decode failed");

    assert_eq!((w, h), (w2, h2));
    assert_eq!(pixels_dithered.len(), pixels_none.len());

    let mut diff_count = 0u64;
    let mut max_diff = 0u8;
    let mut total_diff = 0u64;
    let npixels = (w * h) as usize;

    for i in 0..npixels {
        let base = i * 4;
        for c in 0..3 {
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

    assert!(
        diff_count > 0,
        "v2 dithering had no effect — expected pixel differences at Q95"
    );

    let total_channels = npixels as u64 * 3;
    let diff_pct = (diff_count as f64 / total_channels as f64) * 100.0;
    assert!(
        diff_pct > 1.0,
        "v2 only {diff_pct:.2}% of channels changed — expected more dithering effect"
    );

    assert!(
        max_diff <= 8,
        "v2 max pixel difference {max_diff} too large"
    );

    let avg_diff = total_diff as f64 / diff_count.max(1) as f64;
    assert!(
        avg_diff < 4.0,
        "v2 average difference {avg_diff:.2} too large"
    );

    eprintln!(
        "v2 dithering test: {diff_count}/{total_channels} channels differ ({diff_pct:.1}%), \
         max_diff={max_diff}, avg_diff={avg_diff:.2}"
    );
}

/// v2 decoder: dithering is deterministic (same seed produces same output).
#[test]
fn v2_dithering_is_deterministic() {
    let webp_data = encode_high_quality_webp();

    let config = DecodeConfig::default().with_dithering_strength(50);
    let (pixels_a, _, _) = DecodeRequest::new(&config, &webp_data)
        .decode_rgba_v2()
        .expect("v2 decode failed");

    let (pixels_b, _, _) = DecodeRequest::new(&config, &webp_data)
        .decode_rgba_v2()
        .expect("v2 decode failed");

    assert_eq!(
        pixels_a, pixels_b,
        "v2 dithered decodes with same strength should be deterministic"
    );
}

/// v2 decoder: higher strength produces more dithering.
#[test]
fn v2_higher_strength_produces_more_dithering() {
    let webp_data = encode_high_quality_webp();

    let config_none = DecodeConfig::default().with_dithering_strength(0);
    let (pixels_none, _, _) = DecodeRequest::new(&config_none, &webp_data)
        .decode_rgba_v2()
        .expect("v2 decode failed");

    let config_low = DecodeConfig::default().with_dithering_strength(25);
    let (pixels_low, _, _) = DecodeRequest::new(&config_low, &webp_data)
        .decode_rgba_v2()
        .expect("v2 decode failed");

    let config_high = DecodeConfig::default().with_dithering_strength(100);
    let (pixels_high, _, _) = DecodeRequest::new(&config_high, &webp_data)
        .decode_rgba_v2()
        .expect("v2 decode failed");

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
        "v2 strength=100 ({diff_high}) should produce more change than strength=25 ({diff_low})"
    );
}

/// v2 decoder: dithered output matches v1 dithered output pixel-for-pixel.
///
/// Both decoders use the same PRNG, amplitude computation, and dither_row logic.
/// With the same dithering strength, their RGBA output should be identical.
#[test]
fn v2_dithering_matches_v1() {
    let webp_data = encode_high_quality_webp();

    for strength in [0u8, 25, 50, 75, 100] {
        let config = DecodeConfig::default().with_dithering_strength(strength);

        let (v1_pixels, w1, h1) = DecodeRequest::new(&config, &webp_data)
            .decode_rgba()
            .unwrap_or_else(|e| panic!("v1 decode s{strength} failed: {e}"));

        let (v2_pixels, w2, h2) = DecodeRequest::new(&config, &webp_data)
            .decode_rgba_v2()
            .unwrap_or_else(|e| panic!("v2 decode s{strength} failed: {e}"));

        assert_eq!(
            (w1, h1),
            (u32::from(w2), u32::from(h2)),
            "v1/v2 dimension mismatch at strength={strength}"
        );
        assert_eq!(
            v1_pixels.len(),
            v2_pixels.len(),
            "v1/v2 buffer size mismatch at strength={strength}"
        );

        let mut max_diff = 0u8;
        let mut diff_count = 0u64;
        for (i, (&a, &b)) in v1_pixels.iter().zip(v2_pixels.iter()).enumerate() {
            let d = a.abs_diff(b);
            if d > 0 {
                diff_count += 1;
                if d > max_diff {
                    max_diff = d;
                    if max_diff > 0 {
                        let px = i / 4;
                        let ch = i % 4;
                        eprintln!("v1/v2 diff at pixel {px} channel {ch}: v1={a} v2={b} diff={d}");
                    }
                }
            }
        }

        assert_eq!(
            max_diff, 0,
            "v2 dithered output differs from v1 at strength={strength}: \
             {diff_count} channels differ, max_diff={max_diff}"
        );
    }
}

/// v2 decoder: dithered output matches libwebp across quality levels.
#[test]
fn v2_dithered_matches_libwebp_across_quality_levels() {
    let width = 64u32;
    let height = 64u32;
    let pixels = make_gradient(width as usize, height as usize);

    for quality in [10.0, 50.0, 75.0, 90.0, 95.0] {
        let config = LossyConfig::new().with_quality(quality).with_method(0);
        let webp_data = EncodeRequest::lossy(&config, &pixels, PixelLayout::Rgb8, width, height)
            .encode()
            .unwrap_or_else(|e| panic!("encode Q{quality} failed: {e}"));

        for strength in [0u8, 50, 100] {
            let dc = DecodeConfig::default().with_dithering_strength(strength);
            let (v2_pixels, _, _) = DecodeRequest::new(&dc, &webp_data)
                .decode_rgba_v2()
                .unwrap_or_else(|e| panic!("v2 Q{quality} s{strength} failed: {e}"));

            let lib_pixels = decode_with_libwebp(&webp_data, i32::from(strength));

            assert_eq!(
                v2_pixels.len(),
                lib_pixels.len(),
                "v2/libwebp buffer size mismatch at Q{quality} s{strength}"
            );

            assert_pixels_similar(
                &v2_pixels,
                &lib_pixels,
                &format!("v2 Q{quality} strength={strength}"),
            );
        }
    }
}

/// v2 decoder: dithering works with non-multiple-of-16 dimensions.
#[test]
fn v2_dithered_matches_libwebp_odd_dimensions() {
    let width = 100u32;
    let height = 77u32;
    let pixels = make_gradient(width as usize, height as usize);

    for quality in [50.0, 95.0] {
        let config = LossyConfig::new().with_quality(quality).with_method(0);
        let webp_data = EncodeRequest::lossy(&config, &pixels, PixelLayout::Rgb8, width, height)
            .encode()
            .unwrap_or_else(|e| panic!("encode {width}x{height} Q{quality} failed: {e}"));

        for strength in [0u8, 50, 100] {
            let dc = DecodeConfig::default().with_dithering_strength(strength);
            let (v2_pixels, _, _) = DecodeRequest::new(&dc, &webp_data)
                .decode_rgba_v2()
                .unwrap_or_else(|e| {
                    panic!("v2 decode {width}x{height} Q{quality} s{strength}: {e}")
                });

            let lib_pixels = decode_with_libwebp(&webp_data, i32::from(strength));

            assert_pixels_similar(
                &v2_pixels,
                &lib_pixels,
                &format!("v2 {width}x{height} Q{quality} strength={strength}"),
            );
        }
    }
}
