//! Compare lossless file sizes between zenwebp (VP8L encoder) and libwebp at each method level.

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
            // Lehmer RNG for deterministic noise
            rng = ((rng as u64 * 48271) % 2147483647) as u32;
            let noise = (rng & 0x1F) as u8;
            rgb[idx] = base_r.wrapping_add(noise);
            rgb[idx + 1] = base_g.wrapping_add(noise >> 1);
            rgb[idx + 2] = base_b.wrapping_add(noise >> 2);
        }
    }
    rgb
}

fn generate_complex(width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut rgb = vec![0u8; w * h * 3];
    let mut rng = 42u32;
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            // Multiple frequency components
            let fx = (x as f32 / w as f32 * 6.28) as u8;
            let fy = (y as f32 / h as f32 * 6.28) as u8;
            rng = ((rng as u64 * 48271) % 2147483647) as u32;
            let noise = (rng % 40) as u8;
            rgb[idx] = fx.wrapping_add(fy).wrapping_add(noise);
            rgb[idx + 1] = fx.wrapping_mul(2).wrapping_add(noise >> 1);
            rgb[idx + 2] = fy.wrapping_mul(3).wrapping_add(noise >> 2);
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

#[test]
fn compare_sizes_512x512() {
    let width = 512;
    let height = 512;
    let rgb = generate_photo_like(width, height, 1);

    println!("\nPhoto-like {}x{} lossless size comparison:", width, height);
    println!("{:<10} {:>12} {:>12} {:>8}", "Method", "zenwebp", "libwebp", "ratio");
    println!("{}", "-".repeat(46));

    for method in 0..=6u8 {
        let zen = zenwebp_lossless(&rgb, width, height, 75, method);
        let lib = libwebp_lossless(&rgb, width, height, 75.0, method);
        let ratio = zen.len() as f64 / lib.len() as f64;
        println!("{:<10} {:>12} {:>12} {:>8.4}x",
            format!("m{}", method), zen.len(), lib.len(), ratio);
    }
}

#[test]
fn compare_sizes_complex_512x512() {
    let width = 512;
    let height = 512;
    let rgb = generate_complex(width, height);

    println!("\nComplex {}x{} lossless size comparison:", width, height);
    println!("{:<10} {:>12} {:>12} {:>8}", "Method", "zenwebp", "libwebp", "ratio");
    println!("{}", "-".repeat(46));

    for method in 0..=6u8 {
        let zen = zenwebp_lossless(&rgb, width, height, 75, method);
        let lib = libwebp_lossless(&rgb, width, height, 75.0, method);
        let ratio = zen.len() as f64 / lib.len() as f64;
        println!("{:<10} {:>12} {:>12} {:>8.4}x",
            format!("m{}", method), zen.len(), lib.len(), ratio);
    }
}

#[test]
fn compare_sizes_multiple_images() {
    let sizes = [(256, 256), (384, 256), (512, 512)];

    println!("\nMultiple image sizes, method 4, quality 75:");
    println!("{:<15} {:>12} {:>12} {:>8}", "Size", "zenwebp", "libwebp", "ratio");
    println!("{}", "-".repeat(50));

    for (w, h) in sizes {
        let rgb = generate_photo_like(w, h, w * h);
        let zen = zenwebp_lossless(&rgb, w, h, 75, 4);
        let lib = libwebp_lossless(&rgb, w, h, 75.0, 4);
        let ratio = zen.len() as f64 / lib.len() as f64;
        println!("{:<15} {:>12} {:>12} {:>8.4}x",
            format!("{}x{}", w, h), zen.len(), lib.len(), ratio);
    }
}
