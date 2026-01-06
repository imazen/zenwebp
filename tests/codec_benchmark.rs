//! Rate-distortion benchmark comparing our encoder vs libwebp
//!
//! Uses the Kodak image corpus for standardized comparison.
//! Measures quality at equal bits-per-pixel for apples-to-apples comparison.

use image_webp::{ColorType, EncoderParams, WebPEncoder};
use std::path::Path;

const KODAK_PATH: &str = concat!(env!("HOME"), "/work/codec-corpus/kodak");

/// Calculate PSNR between two images
fn psnr(original: &[u8], decoded: &[u8]) -> f64 {
    assert_eq!(original.len(), decoded.len());
    let mse: f64 = original
        .iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| {
            let diff = f64::from(a) - f64::from(b);
            diff * diff
        })
        .sum::<f64>()
        / original.len() as f64;

    if mse < 0.0001 {
        return 99.0; // Cap at 99 dB for near-identical
    }
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Load a PNG image and return RGB data
fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    // Convert to RGB if needed
    let rgb_data = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity((info.width * info.height * 3) as usize);
            for chunk in buf[..info.buffer_size()].chunks(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        _ => return None,
    };

    Some((rgb_data, info.width, info.height))
}

/// Encode with our encoder at given quality
fn encode_ours(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    let mut output = Vec::new();
    let mut encoder = WebPEncoder::new(&mut output);
    encoder.set_params(EncoderParams::lossy(quality));
    encoder
        .encode(rgb, width, height, ColorType::Rgb8)
        .expect("Our encoding failed");
    output
}

/// Encode with libwebp at given quality
fn encode_libwebp(rgb: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let encoder = webp::Encoder::from_rgb(rgb, width, height);
    encoder.encode(quality).to_vec()
}

/// Decode a WebP file
fn decode_webp(data: &[u8]) -> Vec<u8> {
    let decoded = webp::Decoder::new(data)
        .decode()
        .expect("Failed to decode WebP");
    decoded.to_vec()
}

/// Result for a single encode
#[derive(Debug, Clone)]
struct EncodeResult {
    quality: u8,
    size_bytes: usize,
    bpp: f64,
    psnr: f64,
}

/// Benchmark results for one image
#[derive(Debug)]
struct ImageResults {
    name: String,
    width: u32,
    height: u32,
    pixels: u32,
    ours: Vec<EncodeResult>,
    libwebp: Vec<EncodeResult>,
}

fn benchmark_image(path: &Path) -> Option<ImageResults> {
    let name = path.file_stem()?.to_string_lossy().to_string();
    let (rgb, width, height) = load_png(path)?;
    let pixels = width * height;

    let qualities = [20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95];

    let mut ours = Vec::new();
    let mut libwebp_results = Vec::new();

    for &q in &qualities {
        // Our encoder
        let our_data = encode_ours(&rgb, width, height, q);
        let our_decoded = decode_webp(&our_data);
        let our_psnr = psnr(&rgb, &our_decoded);
        let our_bpp = (our_data.len() * 8) as f64 / pixels as f64;
        ours.push(EncodeResult {
            quality: q,
            size_bytes: our_data.len(),
            bpp: our_bpp,
            psnr: our_psnr,
        });

        // libwebp
        let lib_data = encode_libwebp(&rgb, width, height, q as f32);
        let lib_decoded = decode_webp(&lib_data);
        let lib_psnr = psnr(&rgb, &lib_decoded);
        let lib_bpp = (lib_data.len() * 8) as f64 / pixels as f64;
        libwebp_results.push(EncodeResult {
            quality: q,
            size_bytes: lib_data.len(),
            bpp: lib_bpp,
            psnr: lib_psnr,
        });
    }

    Some(ImageResults {
        name,
        width,
        height,
        pixels,
        ours,
        libwebp: libwebp_results,
    })
}

/// Find quality setting that produces closest BPP to target
fn find_quality_for_bpp(results: &[EncodeResult], target_bpp: f64) -> Option<&EncodeResult> {
    results.iter().min_by(|a, b| {
        let diff_a = (a.bpp - target_bpp).abs();
        let diff_b = (b.bpp - target_bpp).abs();
        diff_a.partial_cmp(&diff_b).unwrap()
    })
}

/// Interpolate PSNR at exact BPP using linear interpolation
fn interpolate_psnr_at_bpp(results: &[EncodeResult], target_bpp: f64) -> Option<f64> {
    // Sort by BPP
    let mut sorted: Vec<_> = results.iter().collect();
    sorted.sort_by(|a, b| a.bpp.partial_cmp(&b.bpp).unwrap());

    // Find bracketing points
    for i in 0..sorted.len() - 1 {
        if sorted[i].bpp <= target_bpp && sorted[i + 1].bpp >= target_bpp {
            let t = (target_bpp - sorted[i].bpp) / (sorted[i + 1].bpp - sorted[i].bpp);
            return Some(sorted[i].psnr + t * (sorted[i + 1].psnr - sorted[i].psnr));
        }
    }

    // Extrapolate if outside range
    if target_bpp < sorted[0].bpp {
        Some(sorted[0].psnr)
    } else if target_bpp > sorted.last()?.bpp {
        Some(sorted.last()?.psnr)
    } else {
        None
    }
}

#[test]
#[ignore] // Run with: cargo test codec_benchmark --release -- --ignored --nocapture
fn codec_benchmark_kodak() {
    let kodak_dir = Path::new(KODAK_PATH);
    if !kodak_dir.exists() {
        eprintln!("Kodak corpus not found at {}", KODAK_PATH);
        eprintln!("Skipping benchmark");
        return;
    }

    println!("\n=== Kodak Corpus Rate-Distortion Benchmark ===\n");

    let mut all_results = Vec::new();

    // Benchmark all images
    for i in 1..=24 {
        let path = kodak_dir.join(format!("{}.png", i));
        if let Some(results) = benchmark_image(&path) {
            println!(
                "Image {:2}: {}x{} ({} pixels)",
                i, results.width, results.height, results.pixels
            );
            all_results.push(results);
        }
    }

    println!("\n=== Per-Quality Comparison ===\n");
    println!(
        "{:>5} | {:>10} {:>8} {:>8} | {:>10} {:>8} {:>8} | {:>8} {:>8}",
        "Q", "Ours Size", "BPP", "PSNR", "libwebp", "BPP", "PSNR", "Size %", "PSNR Δ"
    );
    println!("{}", "-".repeat(95));

    let qualities = [20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95];
    for (qi, &q) in qualities.iter().enumerate() {
        let mut our_total_size = 0usize;
        let mut lib_total_size = 0usize;
        let mut our_total_psnr = 0.0f64;
        let mut lib_total_psnr = 0.0f64;
        let mut total_pixels = 0u64;

        for result in &all_results {
            our_total_size += result.ours[qi].size_bytes;
            lib_total_size += result.libwebp[qi].size_bytes;
            our_total_psnr += result.ours[qi].psnr;
            lib_total_psnr += result.libwebp[qi].psnr;
            total_pixels += result.pixels as u64;
        }

        let our_avg_bpp = (our_total_size * 8) as f64 / total_pixels as f64;
        let lib_avg_bpp = (lib_total_size * 8) as f64 / total_pixels as f64;
        let our_avg_psnr = our_total_psnr / all_results.len() as f64;
        let lib_avg_psnr = lib_total_psnr / all_results.len() as f64;
        let size_ratio = 100.0 * our_total_size as f64 / lib_total_size as f64;
        let psnr_diff = our_avg_psnr - lib_avg_psnr;

        println!(
            "{:>5} | {:>10} {:>8.3} {:>8.2} | {:>10} {:>8.3} {:>8.2} | {:>7.1}% {:>+8.2}",
            q,
            our_total_size,
            our_avg_bpp,
            our_avg_psnr,
            lib_total_size,
            lib_avg_bpp,
            lib_avg_psnr,
            size_ratio,
            psnr_diff
        );
    }

    println!("\n=== Quality at Equal BPP (Apples-to-Apples) ===\n");
    println!(
        "{:>8} | {:>8} {:>8} | {:>8} {:>8} | {:>8}",
        "BPP", "Ours Q", "PSNR", "libwebp Q", "PSNR", "PSNR Δ"
    );
    println!("{}", "-".repeat(65));

    let target_bpps = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0];
    for &target_bpp in &target_bpps {
        let mut our_psnr_sum = 0.0f64;
        let mut lib_psnr_sum = 0.0f64;
        let mut our_q_sum = 0u32;
        let mut lib_q_sum = 0u32;
        let mut count = 0u32;

        for result in &all_results {
            if let (Some(our_psnr), Some(lib_psnr)) = (
                interpolate_psnr_at_bpp(&result.ours, target_bpp),
                interpolate_psnr_at_bpp(&result.libwebp, target_bpp),
            ) {
                our_psnr_sum += our_psnr;
                lib_psnr_sum += lib_psnr;

                if let Some(our_r) = find_quality_for_bpp(&result.ours, target_bpp) {
                    our_q_sum += our_r.quality as u32;
                }
                if let Some(lib_r) = find_quality_for_bpp(&result.libwebp, target_bpp) {
                    lib_q_sum += lib_r.quality as u32;
                }
                count += 1;
            }
        }

        if count > 0 {
            let our_avg_psnr = our_psnr_sum / count as f64;
            let lib_avg_psnr = lib_psnr_sum / count as f64;
            let our_avg_q = our_q_sum / count;
            let lib_avg_q = lib_q_sum / count;
            let psnr_diff = our_avg_psnr - lib_avg_psnr;

            println!(
                "{:>8.2} | {:>8} {:>8.2} | {:>9} {:>8.2} | {:>+8.2}",
                target_bpp, our_avg_q, our_avg_psnr, lib_avg_q, lib_avg_psnr, psnr_diff
            );
        }
    }

    println!("\n=== Summary ===\n");

    // Calculate BD-rate style metric (simplified)
    // Average PSNR difference at common BPP points
    let mut psnr_diffs = Vec::new();
    for &target_bpp in &[0.5, 1.0, 1.5, 2.0, 3.0] {
        let mut our_sum = 0.0f64;
        let mut lib_sum = 0.0f64;
        let mut count = 0u32;

        for result in &all_results {
            if let (Some(our_p), Some(lib_p)) = (
                interpolate_psnr_at_bpp(&result.ours, target_bpp),
                interpolate_psnr_at_bpp(&result.libwebp, target_bpp),
            ) {
                our_sum += our_p;
                lib_sum += lib_p;
                count += 1;
            }
        }
        if count > 0 {
            psnr_diffs.push(our_sum / count as f64 - lib_sum / count as f64);
        }
    }

    let avg_psnr_diff: f64 = psnr_diffs.iter().sum::<f64>() / psnr_diffs.len() as f64;
    println!(
        "Average PSNR difference at equal BPP: {:+.2} dB",
        avg_psnr_diff
    );

    if avg_psnr_diff < -1.0 {
        println!("Status: SIGNIFICANTLY WORSE than libwebp");
    } else if avg_psnr_diff < -0.5 {
        println!("Status: Noticeably worse than libwebp");
    } else if avg_psnr_diff < 0.0 {
        println!("Status: Slightly worse than libwebp");
    } else if avg_psnr_diff < 0.5 {
        println!("Status: Comparable to libwebp");
    } else {
        println!("Status: Better than libwebp (verify this is real!)");
    }
}
