//! Compare lossless file sizes between zenwebp (VP8L encoder) and libwebp at each method level.

use zenwebp::encoder::vp8l::{Vp8lConfig, Vp8lQuality, encode_vp8l};

fn generate_photo_like(width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let base_r = ((x * 255) / w.max(1)) as u8;
            let base_g = ((y * 255) / h.max(1)) as u8;
            let base_b = (((x + y) * 128) / (w + h).max(1)) as u8;
            let noise = ((x.wrapping_mul(7) ^ y.wrapping_mul(13)) & 0x1F) as u8;
            rgb[idx] = base_r.wrapping_add(noise);
            rgb[idx + 1] = base_g.wrapping_add(noise >> 1);
            rgb[idx + 2] = base_b.wrapping_add(noise >> 2);
        }
    }
    rgb
}

fn zenwebp_encode_lossless(rgb: &[u8], width: u32, height: u32, quality: u8, method: u8) -> Vec<u8> {
    let mut config = Vp8lConfig::default();
    config.quality = Vp8lQuality { quality, method };
    encode_vp8l(rgb, width, height, false, &config, &enough::Unstoppable)
        .expect("encode failed")
}

fn zenwebp_encode_lossless_spatial_subgreen(rgb: &[u8], width: u32, height: u32, quality: u8, method: u8) -> Vec<u8> {
    let mut config = Vp8lConfig::default();
    config.quality = Vp8lQuality { quality, method };
    // Force subtract green + predictor (SpatialSubGreen mode)
    config.use_subtract_green = true;
    config.use_predictor = true;
    config.use_cross_color = true;
    config.use_palette = false;
    config.use_meta_huffman = true;
    encode_vp8l(rgb, width, height, false, &config, &enough::Unstoppable)
        .expect("encode failed")
}

fn libwebp_encode_lossless(rgb: &[u8], width: u32, height: u32, quality: f32, method: u8) -> Vec<u8> {
    let config = webpx::EncoderConfig::new()
        .lossless(true)
        .quality(quality)
        .method(method);
    config.encode_rgb(rgb, width, height, &webpx::Unstoppable)
        .expect("encode failed")
}

#[test]
fn compare_sizes_photo_128() {
    let width = 128;
    let height = 128;
    let rgb = generate_photo_like(width, height);

    println!("\nPhoto-like {}x{} lossless size comparison:", width, height);
    println!("{:<10} {:>12} {:>12} {:>8}", "Method", "zenwebp", "libwebp", "ratio");
    println!("{}", "-".repeat(46));

    for method in 0..=6u8 {
        let quality = 75u8;
        let zen = zenwebp_encode_lossless(&rgb, width, height, quality, method);
        let lib = libwebp_encode_lossless(&rgb, width, height, quality as f32, method);
        let ratio = zen.len() as f64 / lib.len() as f64;
        println!("{:<10} {:>12} {:>12} {:>8.4}x",
            format!("m{}", method), zen.len(), lib.len(), ratio);
    }

    // Check that method 4 is within 5% of libwebp
    let zen4 = zenwebp_encode_lossless(&rgb, width, height, 75, 4);
    let lib4 = libwebp_encode_lossless(&rgb, width, height, 75.0, 4);
    let ratio_m4 = zen4.len() as f64 / lib4.len() as f64;
    assert!(ratio_m4 < 1.05, "method 4 ratio {:.4}x is too high (should be < 1.05)", ratio_m4);
}

#[test]
fn compare_sizes_photo_256() {
    let width = 256;
    let height = 256;
    let rgb = generate_photo_like(width, height);

    println!("\nPhoto-like {}x{} lossless size comparison:", width, height);
    println!("{:<10} {:>12} {:>12} {:>8}", "Method", "zenwebp", "libwebp", "ratio");
    println!("{}", "-".repeat(46));

    for method in 0..=6u8 {
        let quality = 75u8;
        let zen = zenwebp_encode_lossless(&rgb, width, height, quality, method);
        let lib = libwebp_encode_lossless(&rgb, width, height, quality as f32, method);
        let ratio = zen.len() as f64 / lib.len() as f64;
        println!("{:<10} {:>12} {:>12} {:>8.4}x",
            format!("m{}", method), zen.len(), lib.len(), ratio);
    }
}

#[test]
fn compare_sizes_forced_spatial_subgreen() {
    // Force spatial+subgreen mode to isolate backward refs from mode selection
    let width = 128;
    let height = 128;
    let rgb = generate_photo_like(width, height);

    println!("\nForced SpatialSubGreen {}x{} comparison:", width, height);
    println!("{:<10} {:>12} {:>12} {:>8}", "Method", "zen_forced", "libwebp", "ratio");
    println!("{}", "-".repeat(50));

    for method in [0u8, 2, 4, 6] {
        let quality = 75u8;
        let zen = zenwebp_encode_lossless_spatial_subgreen(&rgb, width, height, quality, method);
        let lib = libwebp_encode_lossless(&rgb, width, height, quality as f32, method);
        let ratio = zen.len() as f64 / lib.len() as f64;
        println!("{:<10} {:>12} {:>12} {:>8.4}x",
            format!("m{}", method), zen.len(), lib.len(), ratio);
    }
}
