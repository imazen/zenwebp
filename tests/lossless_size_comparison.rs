//! Compare lossless file sizes between zenwebp (VP8L encoder) and libwebp.
//!
//! Tests at multiple image sizes to verify backward references parity.

use zenwebp::encoder::vp8l::{Vp8lConfig, Vp8lQuality, encode_vp8l};

fn generate_photo_like(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut rgb = vec![0u8; w * h * 3];
    let mut rng = seed;
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let base_r = ((x * 255) / w.max(1)) as u8;
            let base_g = ((y * 255) / h.max(1)) as u8;
            let base_b = (((x + y) * 128) / (w + h).max(1)) as u8;
            rng = ((rng as u64 * 48271) % 2147483647) as u32;
            let noise = (rng & 0x1F) as u8;
            rgb[idx] = base_r.wrapping_add(noise);
            rgb[idx + 1] = base_g.wrapping_add(noise >> 1);
            rgb[idx + 2] = base_b.wrapping_add(noise >> 2);
        }
    }
    rgb
}

fn zenwebp_lossless(rgb: &[u8], width: u32, height: u32, quality: u8, method: u8) -> Vec<u8> {
    let mut config = Vp8lConfig::default();
    config.quality = Vp8lQuality { quality, method };
    encode_vp8l(rgb, width, height, false, &config, &enough::Unstoppable)
        .expect("encode failed")
}

fn libwebp_lossless(rgb: &[u8], width: u32, height: u32, quality: f32, method: u8) -> Vec<u8> {
    let config = webpx::EncoderConfig::new()
        .lossless(true)
        .quality(quality)
        .method(method);
    config.encode_rgb(rgb, width, height, &webpx::Unstoppable)
        .expect("encode failed")
}

/// Test file sizes at 512x512 (where backward refs parity is best measured).
/// At this size, predictor transform and meta-huffman work well for all methods.
#[test]
fn backward_refs_parity_512x512() {
    let width = 512;
    let height = 512;
    let rgb = generate_photo_like(width, height, 1);

    println!("\nBackward refs parity test: Photo-like {}x{}", width, height);
    println!("{:<10} {:>12} {:>12} {:>8}", "Method", "zenwebp", "libwebp", "ratio");
    println!("{}", "-".repeat(46));

    let mut max_ratio = 0.0f64;
    for method in 0..=6u8 {
        let zen = zenwebp_lossless(&rgb, width, height, 75, method);
        let lib = libwebp_lossless(&rgb, width, height, 75.0, method);
        let ratio = zen.len() as f64 / lib.len() as f64;
        println!("{:<10} {:>12} {:>12} {:>8.4}x",
            format!("m{}", method), zen.len(), lib.len(), ratio);
        // Skip m0 since it uses different mode selection
        if method > 0 && ratio > max_ratio {
            max_ratio = ratio;
        }
    }

    // At 512x512, methods 1-6 should be within 2% of libwebp
    assert!(max_ratio < 1.02,
        "worst ratio {:.4}x exceeds 2% target", max_ratio);
}

/// Test at multiple sizes to verify consistency.
#[test]
fn backward_refs_parity_multiple_sizes() {
    let sizes: &[(u32, u32)] = &[
        (256, 256), (384, 256), (512, 384), (512, 512), (640, 480),
    ];

    println!("\nBackward refs parity at method 4, quality 75:");
    println!("{:<15} {:>12} {:>12} {:>8}", "Size", "zenwebp", "libwebp", "ratio");
    println!("{}", "-".repeat(50));

    let mut max_ratio = 0.0f64;
    for &(w, h) in sizes {
        let rgb = generate_photo_like(w, h, w.wrapping_mul(h));
        let zen = zenwebp_lossless(&rgb, w, h, 75, 4);
        let lib = libwebp_lossless(&rgb, w, h, 75.0, 4);
        let ratio = zen.len() as f64 / lib.len() as f64;
        println!("{:<15} {:>12} {:>12} {:>8.4}x",
            format!("{}x{}", w, h), zen.len(), lib.len(), ratio);
        if ratio > max_ratio {
            max_ratio = ratio;
        }
    }

    // All sizes at m4 should be within 2% of libwebp
    assert!(max_ratio < 1.02,
        "worst ratio {:.4}x exceeds 2% target", max_ratio);
}

/// Verify lossless roundtrip correctness with all methods.
#[test]
fn lossless_roundtrip_all_methods() {
    let width = 64;
    let height = 64;
    let rgb = generate_photo_like(width, height, 42);

    for method in 0..=6u8 {
        let vp8l = zenwebp_lossless(&rgb, width, height, 75, method);

        // Wrap in RIFF container
        let mut webp = Vec::new();
        webp.extend_from_slice(b"RIFF");
        let riff_size = 4 + 8 + vp8l.len() + (vp8l.len() % 2);
        webp.extend_from_slice(&(riff_size as u32).to_le_bytes());
        webp.extend_from_slice(b"WEBP");
        webp.extend_from_slice(b"VP8L");
        webp.extend_from_slice(&(vp8l.len() as u32).to_le_bytes());
        webp.extend_from_slice(&vp8l);
        if vp8l.len() % 2 != 0 {
            webp.push(0);
        }

        let (decoded, dw, dh) = zenwebp::decode_rgba(&webp).expect("decode failed");
        assert_eq!(dw, width);
        assert_eq!(dh, height);

        let total = (width * height) as usize;
        let mut mismatches = 0;
        for i in 0..total {
            if decoded[i * 4] != rgb[i * 3]
                || decoded[i * 4 + 1] != rgb[i * 3 + 1]
                || decoded[i * 4 + 2] != rgb[i * 3 + 2]
            {
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "method {} had {} mismatches", method, mismatches);
    }
}
