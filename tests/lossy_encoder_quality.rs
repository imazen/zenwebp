//! Lossy encoder quality and size comparison tests
//!
//! These tests compare the image-webp lossy encoder against libwebp:
//! - Bitstream compatibility: libwebp can decode our output
//! - Quality metrics: PSNR, DSSIM, and SSIMULACRA2 of decoded output
//! - Size comparison: our output vs libwebp's output at same quality setting
//!
//! Current status:
//! - At same file size: ~99% of libwebp's PSNR quality
//! - At same Q setting: files are ~1.2-1.6x larger (less efficient encoding)
//! - Missing optimizations: adaptive token probabilities, segment quantization
//!
//! The encoder implements RD-based mode selection with VP8Matrix biased
//! quantization and skip detection for zero macroblocks.

use dssim_core::Dssim;
use fast_ssim2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
use image_webp::{ColorType, EncoderParams, WebPEncoder};
use imgref::ImgVec;
use rgb::RGBA;

/// Simple PSNR calculation (not great for perceptual quality, but simple)
fn calculate_psnr(original: &[u8], decoded: &[u8]) -> f64 {
    assert_eq!(original.len(), decoded.len());

    let mse: f64 = original
        .iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b as f64;
            diff * diff
        })
        .sum::<f64>()
        / original.len() as f64;

    if mse == 0.0 {
        return f64::INFINITY;
    }

    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Calculate DSSIM (perceptual quality metric)
/// Returns 0 for identical images, higher values indicate more difference
/// DSSIM = 1/SSIM - 1, so 0.01 means SSIM ≈ 0.99
fn calculate_dssim(original: &[u8], decoded: &[u8], width: u32, height: u32) -> f64 {
    let w = width as usize;
    let h = height as usize;

    // Convert RGB8 to RGBA<f32> in linear light for dssim
    // sRGB gamma decode: (x / 255)^2.2 approximation
    fn srgb_to_linear(v: u8) -> f32 {
        let x = v as f32 / 255.0;
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    }

    let orig_rgba: Vec<RGBA<f32>> = original
        .chunks_exact(3)
        .map(|p| {
            RGBA::new(
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
                1.0,
            )
        })
        .collect();
    let dec_rgba: Vec<RGBA<f32>> = decoded
        .chunks_exact(3)
        .map(|p| {
            RGBA::new(
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
                1.0,
            )
        })
        .collect();

    let orig_img: ImgVec<RGBA<f32>> = ImgVec::new(orig_rgba, w, h);
    let dec_img: ImgVec<RGBA<f32>> = ImgVec::new(dec_rgba, w, h);

    let dssim = Dssim::new();
    let orig_dssim = dssim.create_image(&orig_img).unwrap();
    let dec_dssim = dssim.create_image(&dec_img).unwrap();

    let (dssim_val, _) = dssim.compare(&orig_dssim, dec_dssim);
    dssim_val.into()
}

/// Calculate SSIMULACRA2 (state-of-the-art perceptual quality metric)
/// Returns score where: 90+ = excellent, 70-90 = good, 50-70 = acceptable, <50 = poor
/// Higher is better (opposite of DSSIM)
fn calculate_ssimulacra2(original: &[u8], decoded: &[u8], width: u32, height: u32) -> f64 {
    let w = width as usize;
    let h = height as usize;

    // sRGB gamma decode
    fn srgb_to_linear(v: u8) -> f32 {
        let x = v as f32 / 255.0;
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    }

    // Convert to linear RGB f32 format for fast-ssim2
    let orig_rgb: Vec<[f32; 3]> = original
        .chunks_exact(3)
        .map(|p| {
            [
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
            ]
        })
        .collect();
    let dec_rgb: Vec<[f32; 3]> = decoded
        .chunks_exact(3)
        .map(|p| {
            [
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
            ]
        })
        .collect();

    let orig_img = Rgb::new(
        orig_rgb,
        w,
        h,
        TransferCharacteristic::Linear,
        ColorPrimaries::BT709,
    )
    .unwrap();
    let dec_img = Rgb::new(
        dec_rgb,
        w,
        h,
        TransferCharacteristic::Linear,
        ColorPrimaries::BT709,
    )
    .unwrap();

    compute_frame_ssimulacra2(orig_img, dec_img).unwrap()
}

/// Helper to create lossy encoder params
fn lossy_params(quality: u8) -> EncoderParams {
    EncoderParams::lossy(quality)
}

/// Test that libwebp can decode our lossy output
#[test]
fn libwebp_can_decode_our_lossy_output() {
    // Create a simple gradient test image
    let width = 64u32;
    let height = 64u32;
    let mut img = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            img[idx] = (x * 4) as u8; // R: horizontal gradient
            img[idx + 1] = (y * 4) as u8; // G: vertical gradient
            img[idx + 2] = 128; // B: constant
        }
    }

    // Encode with image-webp lossy (quality 75)
    let mut output = Vec::new();
    let mut encoder = WebPEncoder::new(&mut output);
    encoder.set_params(lossy_params(75));
    encoder
        .encode(&img, width, height, ColorType::Rgb8)
        .expect("Encoding failed");

    println!("Encoded {} bytes", output.len());

    // Verify it's a valid WebP that libwebp can decode
    let decoded = webp::Decoder::new(&output)
        .decode()
        .expect("libwebp failed to decode our output");

    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);

    // For lossy, we can't expect exact match, but PSNR should be reasonable
    let psnr = calculate_psnr(&img, &decoded);
    println!("PSNR: {:.2} dB (target: > 20 dB for basic lossy)", psnr);
    assert!(psnr > 15.0, "PSNR too low: {:.2} dB", psnr);
}

/// Compare our encoder size vs libwebp at same quality
#[test]
fn size_comparison_vs_libwebp() {
    let width = 128u32;
    let height = 128u32;

    // Create test image with some texture (checkerboard + gradient)
    let mut img = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            let checker = ((x / 8 + y / 8) % 2) as u8 * 64;
            img[idx] = ((x * 2) as u8).wrapping_add(checker);
            img[idx + 1] = ((y * 2) as u8).wrapping_add(checker);
            img[idx + 2] = 128u8.wrapping_add(checker);
        }
    }

    for quality in [50u8, 75, 90] {
        // Encode with image-webp
        let mut our_output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut our_output);
        encoder.set_params(lossy_params(quality));
        encoder
            .encode(&img, width, height, ColorType::Rgb8)
            .expect("Our encoding failed");

        // Encode with libwebp
        let libwebp_encoder = webp::Encoder::from_rgb(&img, width, height);
        let libwebp_output = libwebp_encoder.encode(quality as f32);

        let our_size = our_output.len();
        let libwebp_size = libwebp_output.len();
        let ratio = our_size as f64 / libwebp_size as f64;

        println!(
            "Quality {}: ours = {} bytes, libwebp = {} bytes, ratio = {:.2}x",
            quality, our_size, libwebp_size, ratio
        );

        // We should be within reasonable bounds of libwebp's size.
        // Note: synthetic checkerboard test images are harder than real images.
        // Real images (Kodak corpus) show ~1.2-1.5x ratio at various quality levels.
        // Allow up to 2.1x for this synthetic test.
        assert!(
            ratio < 2.1,
            "Our output is {:.2}x larger than libwebp at quality {}",
            ratio,
            quality
        );
    }
}

/// Compare decoded quality metrics at same quality setting
/// Note: Same Q setting produces different file sizes (ours ~1.4x larger at Q75)
/// but quality metrics are comparable since we're using more bits.
#[test]
fn quality_comparison_at_same_quality_setting() {
    let width = 128u32;
    let height = 128u32;

    // Load a real test image or create a complex one
    let mut img = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            // Create some texture
            let noise = ((x * 7 + y * 13) % 64) as u8;
            img[idx] = ((x * 2) as u8).wrapping_add(noise);
            img[idx + 1] = ((y * 2) as u8).wrapping_add(noise);
            img[idx + 2] = (((x + y) * 2) as u8).wrapping_add(noise);
        }
    }

    // Encode with libwebp at quality 75
    let libwebp_encoder = webp::Encoder::from_rgb(&img, width, height);
    let libwebp_output = libwebp_encoder.encode(75.0);

    // Encode with our encoder at same quality
    let mut our_output = Vec::new();
    let mut encoder = WebPEncoder::new(&mut our_output);
    encoder.set_params(lossy_params(75));
    encoder
        .encode(&img, width, height, ColorType::Rgb8)
        .expect("Our encoding failed");

    // Decode both
    let our_decoded = webp::Decoder::new(&our_output)
        .decode()
        .expect("Failed to decode our output");
    let libwebp_decoded = webp::Decoder::new(&libwebp_output)
        .decode()
        .expect("Failed to decode libwebp output");

    // Calculate PSNR, DSSIM, and SSIMULACRA2
    let our_psnr = calculate_psnr(&img, &our_decoded);
    let libwebp_psnr = calculate_psnr(&img, &libwebp_decoded);
    let our_dssim = calculate_dssim(&img, &our_decoded, width, height);
    let libwebp_dssim = calculate_dssim(&img, &libwebp_decoded, width, height);
    let our_ssim2 = calculate_ssimulacra2(&img, &our_decoded, width, height);
    let libwebp_ssim2 = calculate_ssimulacra2(&img, &libwebp_decoded, width, height);

    println!("=== Quality Comparison at Q75 ===");
    println!(
        "Our encoder: {} bytes, PSNR = {:.2} dB, DSSIM = {:.6}, SSIMULACRA2 = {:.2}",
        our_output.len(),
        our_psnr,
        our_dssim,
        our_ssim2
    );
    println!(
        "libwebp:     {} bytes, PSNR = {:.2} dB, DSSIM = {:.6}, SSIMULACRA2 = {:.2}",
        libwebp_output.len(),
        libwebp_psnr,
        libwebp_dssim,
        libwebp_ssim2
    );
    println!(
        "Ratio: size {:.2}x, PSNR {:.1}%, DSSIM {:.2}x, SSIMULACRA2 {:.1}%",
        our_output.len() as f64 / libwebp_output.len() as f64,
        100.0 * our_psnr / libwebp_psnr,
        our_dssim / libwebp_dssim.max(0.0001), // avoid div by zero
        100.0 * our_ssim2 / libwebp_ssim2.max(0.01)
    );

    // Our quality should be at least 80% of libwebp's PSNR
    let quality_ratio = our_psnr / libwebp_psnr;
    assert!(
        quality_ratio > 0.8,
        "Our quality ({:.2} dB) is less than 80% of libwebp ({:.2} dB)",
        our_psnr,
        libwebp_psnr
    );

    // DSSIM should not be more than 3x worse than libwebp
    // (DSSIM: lower is better, so we check if ours is less than 3x theirs)
    assert!(
        our_dssim < libwebp_dssim * 3.0,
        "Our DSSIM ({:.6}) is more than 3x worse than libwebp ({:.6})",
        our_dssim,
        libwebp_dssim
    );
}

/// Test various image types for encoder robustness
#[test]
fn encode_various_image_types() {
    let test_cases = [
        ("solid_color", create_solid_color_image(64, 64)),
        ("gradient", create_gradient_image(64, 64)),
        ("checkerboard", create_checkerboard_image(64, 64)),
        ("noise", create_noise_image(64, 64)),
    ];

    for (name, img) in test_cases {
        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(lossy_params(75));
        encoder
            .encode(&img, 64, 64, ColorType::Rgb8)
            .unwrap_or_else(|e| panic!("Failed to encode {}: {:?}", name, e));

        // Verify libwebp can decode
        let decoded = webp::Decoder::new(&output)
            .decode()
            .unwrap_or_else(|| panic!("libwebp failed to decode {}", name));

        let psnr = calculate_psnr(&img, &decoded);
        println!("{}: {} bytes, PSNR = {:.2} dB", name, output.len(), psnr);

        // Minimum acceptable PSNR (noise is a pathological case, allow lower threshold)
        let min_psnr = if name == "noise" { 10.0 } else { 20.0 };
        assert!(
            psnr > min_psnr,
            "{} has unacceptably low PSNR: {:.2} (min: {:.0})",
            name,
            psnr,
            min_psnr
        );
    }
}

// Helper functions to create test images
fn create_solid_color_image(w: u32, h: u32) -> Vec<u8> {
    vec![128u8; (w * h * 3) as usize]
}

fn create_gradient_image(w: u32, h: u32) -> Vec<u8> {
    let mut img = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            img[idx] = (x * 255 / w) as u8;
            img[idx + 1] = (y * 255 / h) as u8;
            img[idx + 2] = ((x + y) * 255 / (w + h)) as u8;
        }
    }
    img
}

fn create_checkerboard_image(w: u32, h: u32) -> Vec<u8> {
    let mut img = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            let val = if (x / 8 + y / 8) % 2 == 0 { 255 } else { 0 };
            img[idx] = val;
            img[idx + 1] = val;
            img[idx + 2] = val;
        }
    }
    img
}

fn create_noise_image(w: u32, h: u32) -> Vec<u8> {
    use rand::RngCore;
    let mut img = vec![0u8; (w * h * 3) as usize];
    rand::thread_rng().fill_bytes(&mut img);
    img
}
