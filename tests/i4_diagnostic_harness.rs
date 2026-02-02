//! I4 Encoding Efficiency Decomposition Harness
//!
//! Compares zenwebp and libwebp VP8 bitstreams stage-by-stage to identify
//! where encoding efficiency diverges. Decodes both outputs with diagnostic
//! capture to compare modes, coefficient levels, and probability tables.
//!
//! Run with: cargo test --release --features _corpus_tests --test i4_diagnostic_harness -- --nocapture

#![cfg(feature = "_corpus_tests")]

use std::fs;
use std::io::BufReader;
use webpx::Unstoppable;
use zenwebp::decoder::vp8::{DiagnosticFrame, Vp8Decoder};
use zenwebp::decoder::LumaMode;
use zenwebp::{EncoderConfig, Preset};

// ============================================================================
// Image Loading
// ============================================================================

fn load_png(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let width = info.width;
    let height = info.height;

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        _ => return None,
    };

    Some((rgb, width, height))
}

/// Create a synthetic test image with known patterns for predictable encoding
fn create_test_image(width: u32, height: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            // Checkerboard with noise pattern - good for I4 mode selection
            let checker = ((x / 4) + (y / 4)) % 2 == 0;
            let base = if checker { 200u8 } else { 50u8 };
            // Add slight variation to avoid degenerate DCT
            let noise = ((x * 7 + y * 13) % 20) as u8;
            rgb[idx] = base.saturating_add(noise);
            rgb[idx + 1] = base.saturating_add(noise / 2);
            rgb[idx + 2] = base.saturating_sub(noise / 2);
        }
    }
    rgb
}

// ============================================================================
// VP8 Chunk Extraction
// ============================================================================

/// Extract raw VP8 data from a WebP container
fn extract_vp8_chunk(webp: &[u8]) -> Option<&[u8]> {
    // WebP format: RIFF header (12 bytes) + chunks
    // RIFF <size> WEBP
    if webp.len() < 12 || &webp[0..4] != b"RIFF" || &webp[8..12] != b"WEBP" {
        return None;
    }

    let mut pos = 12;
    while pos + 8 <= webp.len() {
        let chunk_fourcc = &webp[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(webp[pos + 4..pos + 8].try_into().ok()?) as usize;

        if chunk_fourcc == b"VP8 " {
            // VP8 lossy chunk - data starts at pos + 8
            let data_start = pos + 8;
            let data_end = (data_start + chunk_size).min(webp.len());
            return Some(&webp[data_start..data_end]);
        }

        // Move to next chunk (chunks are padded to even size)
        pos += 8 + chunk_size + (chunk_size & 1);
    }

    None
}

// ============================================================================
// Encoding Helpers with Matched Settings
// ============================================================================

/// Encode with zenwebp using diagnostic-friendly settings
fn encode_zenwebp(rgb: &[u8], width: u32, height: u32, quality: f32, method: u8) -> Vec<u8> {
    EncoderConfig::with_preset(Preset::Default, quality)
        .method(method)
        // Disable SNS and filtering for cleaner comparison
        .sns_strength(0)
        .filter_strength(0)
        .filter_sharpness(0)
        .segments(1) // Single segment simplifies quantizer comparison
        .encode_rgb(rgb, width, height)
        .expect("zenwebp encoding failed")
}

/// Encode with libwebp (via webpx) using matched settings
fn encode_libwebp(rgb: &[u8], width: u32, height: u32, quality: f32, method: u8) -> Vec<u8> {
    let config = webpx::EncoderConfig::with_preset(webpx::Preset::Default, quality)
        .method(method)
        .sns_strength(0)
        .filter_strength(0)
        .filter_sharpness(0)
        .segments(1);

    config
        .encode_rgb(rgb, width, height, Unstoppable)
        .expect("libwebp encoding failed")
}

// ============================================================================
// Diagnostic Comparison
// ============================================================================

/// Statistics from comparing two DiagnosticFrames
#[derive(Debug, Default)]
struct ComparisonStats {
    /// Number of macroblocks with matching luma mode
    luma_mode_matches: usize,
    /// Number of macroblocks with different luma mode
    luma_mode_mismatches: usize,

    /// I4 blocks where both chose I4
    i4_blocks_both: usize,
    /// I4 sub-block mode matches (within blocks where both are I4)
    i4_mode_matches: usize,
    /// I4 sub-block mode mismatches
    i4_mode_mismatches: usize,

    /// Coefficient level comparisons
    coeff_blocks_compared: usize,
    coeff_blocks_exact: usize,
    total_level_diff: i64,
    max_level_diff: i32,

    /// First few mismatches for debugging
    first_mismatches: Vec<String>,
}

impl ComparisonStats {
    fn report(&self) {
        let total_mbs = self.luma_mode_matches + self.luma_mode_mismatches;
        let mode_pct = if total_mbs > 0 {
            100.0 * self.luma_mode_matches as f64 / total_mbs as f64
        } else {
            0.0
        };

        println!("\n=== Diagnostic Comparison Results ===");
        println!(
            "Mode decisions: {}/{} match ({:.1}%)",
            self.luma_mode_matches, total_mbs, mode_pct
        );

        if self.i4_blocks_both > 0 {
            let i4_total = self.i4_mode_matches + self.i4_mode_mismatches;
            let i4_pct = 100.0 * self.i4_mode_matches as f64 / i4_total as f64;
            println!(
                "I4 sub-block modes: {}/{} match ({:.1}%)",
                self.i4_mode_matches, i4_total, i4_pct
            );
        }

        if self.coeff_blocks_compared > 0 {
            let coeff_pct = 100.0 * self.coeff_blocks_exact as f64 / self.coeff_blocks_compared as f64;
            println!(
                "Coefficient blocks: {}/{} exact ({:.1}%)",
                self.coeff_blocks_exact, self.coeff_blocks_compared, coeff_pct
            );
            println!(
                "Total |level_diff|: {}, max: {}",
                self.total_level_diff, self.max_level_diff
            );
        }

        if !self.first_mismatches.is_empty() {
            println!("\nFirst mismatches:");
            for m in &self.first_mismatches {
                println!("  {}", m);
            }
        }
    }
}

/// Compare two diagnostic frames and collect statistics
fn compare_diagnostics(zen: &DiagnosticFrame, lib: &DiagnosticFrame) -> ComparisonStats {
    let mut stats = ComparisonStats::default();

    // Check dimensions match
    assert_eq!(zen.mb_width, lib.mb_width, "MB width mismatch");
    assert_eq!(zen.mb_height, lib.mb_height, "MB height mismatch");
    assert_eq!(
        zen.macroblocks.len(),
        lib.macroblocks.len(),
        "MB count mismatch"
    );

    // Compare segment quantizers
    println!("\nSegment quantizers:");
    for (i, (z, l)) in zen.segments.iter().zip(lib.segments.iter()).enumerate() {
        let match_str = if z == l { "[MATCH]" } else { "[DIFFER]" };
        println!(
            "  Seg {}: zen={:?} lib={:?} {}",
            i, z, l, match_str
        );
    }

    // Compare macroblocks
    for (mb_idx, (z_mb, l_mb)) in zen.macroblocks.iter().zip(lib.macroblocks.iter()).enumerate() {
        let mbx = mb_idx % zen.mb_width as usize;
        let mby = mb_idx / zen.mb_width as usize;

        // Compare luma mode
        if z_mb.luma_mode == l_mb.luma_mode {
            stats.luma_mode_matches += 1;
        } else {
            stats.luma_mode_mismatches += 1;
            if stats.first_mismatches.len() < 5 {
                stats.first_mismatches.push(format!(
                    "MB({},{}) luma mode: zen={:?} lib={:?}",
                    mbx, mby, z_mb.luma_mode, l_mb.luma_mode
                ));
            }
        }

        // If both are I4 (LumaMode::B), compare sub-block modes
        if z_mb.luma_mode == LumaMode::B && l_mb.luma_mode == LumaMode::B {
            stats.i4_blocks_both += 1;
            for (blk_idx, (z_mode, l_mode)) in z_mb
                .bpred_modes
                .iter()
                .zip(l_mb.bpred_modes.iter())
                .enumerate()
            {
                if z_mode == l_mode {
                    stats.i4_mode_matches += 1;
                } else {
                    stats.i4_mode_mismatches += 1;
                    if stats.first_mismatches.len() < 5 {
                        stats.first_mismatches.push(format!(
                            "MB({},{}) I4 block {}: zen={:?} lib={:?}",
                            mbx, mby, blk_idx, z_mode, l_mode
                        ));
                    }
                }
            }

            // Compare Y block coefficients (only when both are I4)
            for (blk_idx, (z_blk, l_blk)) in z_mb
                .y_blocks
                .iter()
                .zip(l_mb.y_blocks.iter())
                .enumerate()
            {
                stats.coeff_blocks_compared += 1;
                let mut exact = true;
                let mut block_diff = 0i64;

                for pos in 0..16 {
                    let z_level = z_blk.levels[pos];
                    let l_level = l_blk.levels[pos];
                    let diff = (z_level - l_level).abs();
                    if diff != 0 {
                        exact = false;
                        block_diff += diff as i64;
                        stats.max_level_diff = stats.max_level_diff.max(diff);

                        if stats.first_mismatches.len() < 5 {
                            stats.first_mismatches.push(format!(
                                "MB({},{}) Y block {} pos {}: zen={} lib={}",
                                mbx, mby, blk_idx, pos, z_level, l_level
                            ));
                        }
                    }
                }

                if exact {
                    stats.coeff_blocks_exact += 1;
                }
                stats.total_level_diff += block_diff;
            }
        }
    }

    stats
}

// ============================================================================
// Test Cases
// ============================================================================

#[test]
fn single_mb_diagnostic() {
    println!("\n=== Single MB Diagnostic (16x16, method 2, Q75) ===");

    let rgb = create_test_image(16, 16);
    let quality = 75.0;
    let method = 2; // No trellis

    let zen_webp = encode_zenwebp(&rgb, 16, 16, quality, method);
    let lib_webp = encode_libwebp(&rgb, 16, 16, quality, method);

    println!(
        "File sizes: zenwebp={} bytes, libwebp={} bytes, ratio={:.3}x",
        zen_webp.len(),
        lib_webp.len(),
        zen_webp.len() as f64 / lib_webp.len() as f64
    );

    // Extract VP8 chunks
    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8 from zenwebp");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8 from libwebp");

    // Decode with diagnostics
    let (_, zen_diag) =
        Vp8Decoder::decode_diagnostic(zen_vp8).expect("Failed to decode zenwebp with diagnostics");
    let (_, lib_diag) =
        Vp8Decoder::decode_diagnostic(lib_vp8).expect("Failed to decode libwebp with diagnostics");

    println!(
        "Partition 0: zenwebp={} bytes, libwebp={} bytes",
        zen_diag.partition0_size, lib_diag.partition0_size
    );

    let stats = compare_diagnostics(&zen_diag, &lib_diag);
    stats.report();
}

#[test]
fn small_image_diagnostic() {
    println!("\n=== Small Image Diagnostic (64x64, method 2, Q75) ===");

    let rgb = create_test_image(64, 64);
    let quality = 75.0;
    let method = 2;

    let zen_webp = encode_zenwebp(&rgb, 64, 64, quality, method);
    let lib_webp = encode_libwebp(&rgb, 64, 64, quality, method);

    println!(
        "File sizes: zenwebp={} bytes, libwebp={} bytes, ratio={:.3}x",
        zen_webp.len(),
        lib_webp.len(),
        zen_webp.len() as f64 / lib_webp.len() as f64
    );

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8 from zenwebp");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8 from libwebp");

    let (_, zen_diag) =
        Vp8Decoder::decode_diagnostic(zen_vp8).expect("Failed to decode zenwebp with diagnostics");
    let (_, lib_diag) =
        Vp8Decoder::decode_diagnostic(lib_vp8).expect("Failed to decode libwebp with diagnostics");

    let stats = compare_diagnostics(&zen_diag, &lib_diag);
    stats.report();

    // Count mode breakdown
    let zen_i4 = zen_diag
        .macroblocks
        .iter()
        .filter(|m| m.luma_mode == LumaMode::B)
        .count();
    let lib_i4 = lib_diag
        .macroblocks
        .iter()
        .filter(|m| m.luma_mode == LumaMode::B)
        .count();
    let total = zen_diag.macroblocks.len();

    println!(
        "\nMode breakdown: zenwebp {} I4 / {} total, libwebp {} I4 / {} total",
        zen_i4, total, lib_i4, total
    );
}

#[test]
fn small_image_m4_diagnostic() {
    println!("\n=== Small Image Diagnostic (64x64, method 4 with trellis, Q75) ===");

    let rgb = create_test_image(64, 64);
    let quality = 75.0;
    let method = 4; // With trellis

    let zen_webp = encode_zenwebp(&rgb, 64, 64, quality, method);
    let lib_webp = encode_libwebp(&rgb, 64, 64, quality, method);

    println!(
        "File sizes: zenwebp={} bytes, libwebp={} bytes, ratio={:.3}x",
        zen_webp.len(),
        lib_webp.len(),
        zen_webp.len() as f64 / lib_webp.len() as f64
    );

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8 from zenwebp");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8 from libwebp");

    let (_, zen_diag) =
        Vp8Decoder::decode_diagnostic(zen_vp8).expect("Failed to decode zenwebp with diagnostics");
    let (_, lib_diag) =
        Vp8Decoder::decode_diagnostic(lib_vp8).expect("Failed to decode libwebp with diagnostics");

    let stats = compare_diagnostics(&zen_diag, &lib_diag);
    stats.report();
}

#[test]
fn benchmark_image_diagnostic() {
    println!("\n=== Benchmark Image Diagnostic (792079.png, method 2, Q75) ===");

    // Try to load the benchmark image
    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    println!("Image size: {}x{}", width, height);

    let quality = 75.0;
    let method = 2;

    let zen_webp = encode_zenwebp(&rgb, width, height, quality, method);
    let lib_webp = encode_libwebp(&rgb, width, height, quality, method);

    println!(
        "File sizes: zenwebp={} bytes, libwebp={} bytes, ratio={:.3}x",
        zen_webp.len(),
        lib_webp.len(),
        zen_webp.len() as f64 / lib_webp.len() as f64
    );

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8 from zenwebp");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8 from libwebp");

    let (_, zen_diag) =
        Vp8Decoder::decode_diagnostic(zen_vp8).expect("Failed to decode zenwebp with diagnostics");
    let (_, lib_diag) =
        Vp8Decoder::decode_diagnostic(lib_vp8).expect("Failed to decode libwebp with diagnostics");

    let stats = compare_diagnostics(&zen_diag, &lib_diag);
    stats.report();

    // Aggregate nonzero coefficient stats
    let mut zen_nonzero = 0usize;
    let mut lib_nonzero = 0usize;

    for mb in &zen_diag.macroblocks {
        for blk in &mb.y_blocks {
            zen_nonzero += blk.levels.iter().filter(|&&l| l != 0).count();
        }
    }
    for mb in &lib_diag.macroblocks {
        for blk in &mb.y_blocks {
            lib_nonzero += blk.levels.iter().filter(|&&l| l != 0).count();
        }
    }

    println!(
        "\nNonzero Y coefficients: zenwebp={}, libwebp={}, ratio={:.3}x",
        zen_nonzero,
        lib_nonzero,
        zen_nonzero as f64 / lib_nonzero as f64
    );
}

#[test]
fn benchmark_image_m4_diagnostic() {
    println!("\n=== Benchmark Image Diagnostic (792079.png, method 4 with trellis, Q75) ===");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    println!("Image size: {}x{}", width, height);

    let quality = 75.0;
    let method = 4;

    let zen_webp = encode_zenwebp(&rgb, width, height, quality, method);
    let lib_webp = encode_libwebp(&rgb, width, height, quality, method);

    println!(
        "File sizes: zenwebp={} bytes, libwebp={} bytes, ratio={:.3}x",
        zen_webp.len(),
        lib_webp.len(),
        zen_webp.len() as f64 / lib_webp.len() as f64
    );

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8 from zenwebp");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8 from libwebp");

    let (_, zen_diag) =
        Vp8Decoder::decode_diagnostic(zen_vp8).expect("Failed to decode zenwebp with diagnostics");
    let (_, lib_diag) =
        Vp8Decoder::decode_diagnostic(lib_vp8).expect("Failed to decode libwebp with diagnostics");

    let stats = compare_diagnostics(&zen_diag, &lib_diag);
    stats.report();

    // Aggregate nonzero coefficient stats
    let mut zen_nonzero = 0usize;
    let mut lib_nonzero = 0usize;

    for mb in &zen_diag.macroblocks {
        for blk in &mb.y_blocks {
            zen_nonzero += blk.levels.iter().filter(|&&l| l != 0).count();
        }
    }
    for mb in &lib_diag.macroblocks {
        for blk in &mb.y_blocks {
            lib_nonzero += blk.levels.iter().filter(|&&l| l != 0).count();
        }
    }

    println!(
        "\nNonzero Y coefficients: zenwebp={}, libwebp={}, ratio={:.3}x",
        zen_nonzero,
        lib_nonzero,
        zen_nonzero as f64 / lib_nonzero as f64
    );
}

/// Fair trellis comparison: our m4 (has trellis) vs libwebp m6 (has trellis)
#[test]
fn benchmark_trellis_fair_comparison() {
    println!("\n=== Fair Trellis: zenwebp m4 vs libwebp m6 (both with trellis) ===");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    println!("Image size: {}x{}", width, height);

    let quality = 75.0;

    // Our m4 has trellis; libwebp needs m6 for trellis
    let zen_webp = encode_zenwebp(&rgb, width, height, quality, 4);
    let lib_webp = encode_libwebp(&rgb, width, height, quality, 6);

    println!(
        "File sizes: zenwebp(m4)={} bytes, libwebp(m6)={} bytes, ratio={:.3}x",
        zen_webp.len(),
        lib_webp.len(),
        zen_webp.len() as f64 / lib_webp.len() as f64
    );
}

/// Test method 6 (both have trellis) on real image
#[test]
fn benchmark_image_m6_vs_libwebp_m6() {
    println!("\n=== Benchmark Image: zenwebp m6 vs libwebp m6 (both with trellis) ===");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    println!("Image size: {}x{}", width, height);

    let quality = 75.0;

    // Both at method 6 - both should have trellis
    let zen_webp = encode_zenwebp(&rgb, width, height, quality, 6);
    let lib_webp = encode_libwebp(&rgb, width, height, quality, 6);

    println!(
        "File sizes: zenwebp={} bytes, libwebp={} bytes, ratio={:.3}x",
        zen_webp.len(),
        lib_webp.len(),
        zen_webp.len() as f64 / lib_webp.len() as f64
    );

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8 from zenwebp");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8 from libwebp");

    let (_, zen_diag) =
        Vp8Decoder::decode_diagnostic(zen_vp8).expect("Failed to decode zenwebp with diagnostics");
    let (_, lib_diag) =
        Vp8Decoder::decode_diagnostic(lib_vp8).expect("Failed to decode libwebp with diagnostics");

    let stats = compare_diagnostics(&zen_diag, &lib_diag);
    stats.report();
}

/// Compare coefficient level distributions between encoders
#[test]
fn coefficient_distribution_analysis() {
    println!("\n=== Coefficient Level Distribution Analysis (792079.png m4) ===");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    let zen_webp = encode_zenwebp(&rgb, width, height, 75.0, 4);
    let lib_webp = encode_libwebp(&rgb, width, height, 75.0, 4);

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8");

    let (_, zen_diag) = Vp8Decoder::decode_diagnostic(zen_vp8).expect("decode");
    let (_, lib_diag) = Vp8Decoder::decode_diagnostic(lib_vp8).expect("decode");

    // Count coefficient level distribution for both
    let mut zen_level_counts = [0usize; 128];
    let mut lib_level_counts = [0usize; 128];

    for mb in &zen_diag.macroblocks {
        for blk in &mb.y_blocks {
            for &level in &blk.levels {
                let idx = level.unsigned_abs().min(127) as usize;
                zen_level_counts[idx] += 1;
            }
        }
    }

    for mb in &lib_diag.macroblocks {
        for blk in &mb.y_blocks {
            for &level in &blk.levels {
                let idx = level.unsigned_abs().min(127) as usize;
                lib_level_counts[idx] += 1;
            }
        }
    }

    println!("Level distribution (non-zero levels only):");
    println!("Level | zenwebp | libwebp | delta");
    for i in 1..32 {
        if zen_level_counts[i] > 0 || lib_level_counts[i] > 0 {
            let delta = zen_level_counts[i] as i64 - lib_level_counts[i] as i64;
            let delta_sign = if delta > 0 { "+" } else { "" };
            println!(
                "{:5} | {:7} | {:7} | {}{}",
                i, zen_level_counts[i], lib_level_counts[i], delta_sign, delta
            );
        }
    }

    // Total nonzero
    let zen_total: usize = zen_level_counts[1..].iter().sum();
    let lib_total: usize = lib_level_counts[1..].iter().sum();
    println!("\nTotal nonzero: zenwebp={}, libwebp={}", zen_total, lib_total);
}

/// Compare probability tables on real benchmark image
#[test]
fn probability_table_comparison_real_image() {
    println!("\n=== Probability Table Comparison (792079.png, m4) ===");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    let zen_webp = encode_zenwebp(&rgb, width, height, 75.0, 4);
    let lib_webp = encode_libwebp(&rgb, width, height, 75.0, 4);

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8");

    let (_, zen_diag) = Vp8Decoder::decode_diagnostic(zen_vp8).expect("decode");
    let (_, lib_diag) = Vp8Decoder::decode_diagnostic(lib_vp8).expect("decode");

    // Compare token probability tables
    let mut total_entries = 0usize;
    let mut matching_entries = 0usize;
    let mut total_diff = 0u64;
    let mut max_diff = 0u8;

    println!("Probability differences by plane/band (showing divergent entries):");
    for plane in 0..4 {
        for band in 0..8 {
            for ctx in 0..3 {
                for tok in 0..11 {
                    total_entries += 1;
                    let z_prob = zen_diag.token_probs[plane][band][ctx][tok].prob;
                    let l_prob = lib_diag.token_probs[plane][band][ctx][tok].prob;

                    if z_prob == l_prob {
                        matching_entries += 1;
                    } else {
                        let diff = z_prob.abs_diff(l_prob);
                        total_diff += diff as u64;
                        max_diff = max_diff.max(diff);
                        if diff >= 30 {
                            println!(
                                "  [P{} B{} C{} T{}] zen={} lib={} diff={}",
                                plane, band, ctx, tok, z_prob, l_prob, diff
                            );
                        }
                    }
                }
            }
        }
    }

    println!(
        "\nToken probabilities: {}/{} match ({:.1}%)",
        matching_entries,
        total_entries,
        100.0 * matching_entries as f64 / total_entries as f64
    );
    println!("Total |prob_diff|: {}, max: {}", total_diff, max_diff);
}

#[test]
fn probability_table_comparison() {
    println!("\n=== Probability Table Comparison (64x64, method 2, Q75) ===");

    let rgb = create_test_image(64, 64);
    let quality = 75.0;
    let method = 2;

    let zen_webp = encode_zenwebp(&rgb, 64, 64, quality, method);
    let lib_webp = encode_libwebp(&rgb, 64, 64, quality, method);

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8 from zenwebp");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8 from libwebp");

    let (_, zen_diag) =
        Vp8Decoder::decode_diagnostic(zen_vp8).expect("Failed to decode zenwebp with diagnostics");
    let (_, lib_diag) =
        Vp8Decoder::decode_diagnostic(lib_vp8).expect("Failed to decode libwebp with diagnostics");

    // Compare token probability tables
    let mut total_entries = 0usize;
    let mut matching_entries = 0usize;
    let mut total_diff = 0u64;
    let mut max_diff = 0u8;

    for plane in 0..4 {
        for band in 0..8 {
            for ctx in 0..3 {
                for tok in 0..11 {
                    total_entries += 1;
                    let z_prob = zen_diag.token_probs[plane][band][ctx][tok].prob;
                    let l_prob = lib_diag.token_probs[plane][band][ctx][tok].prob;
                    if z_prob == l_prob {
                        matching_entries += 1;
                    } else {
                        let diff = z_prob.abs_diff(l_prob);
                        total_diff += diff as u64;
                        max_diff = max_diff.max(diff);
                    }
                }
            }
        }
    }

    println!(
        "Token probabilities: {}/{} match ({:.1}%)",
        matching_entries,
        total_entries,
        100.0 * matching_entries as f64 / total_entries as f64
    );
    println!(
        "Total |prob_diff|: {}, max: {}",
        total_diff, max_diff
    );
}
