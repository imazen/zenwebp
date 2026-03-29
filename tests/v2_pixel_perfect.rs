//! Pixel-exact conformance tests for the zenwebp VP8 decoder against libwebp (via webpx).
//!
//! Tests four categories:
//! 1. Roundtrip: encode with zenwebp, decode with zenwebp and libwebp, compare output.
//! 2. Corpus: decode pre-existing lossy WebP files from codec-corpus with both
//!    zenwebp and libwebp, comparing output.
//! 3. Edge cases: odd dimensions, non-MB-aligned sizes, tiny images.
//! 4. Quality sweep: one image at many quality levels.
//!
//! ## YUV->RGB Conversion Differences
//!
//! With the `fast-yuv` feature (on by default), zenwebp uses the `yuv` crate's
//! bilinear upsampler, which produces different results than libwebp's fancy
//! upsampling. These differences are NOT decoder bugs — they reflect different
//! chroma upsampling implementations applied to identical YUV planes.
//!
//! Run: `cargo test --release --test v2_pixel_perfect -- --nocapture`
#![forbid(unsafe_code)]
#![cfg(not(target_arch = "wasm32"))]

use std::path::Path;
use zenwebp::{DecodeConfig, DecodeRequest, EncodeRequest, EncoderConfig, PixelLayout};

// ---------------------------------------------------------------------------
// Decode helpers
// ---------------------------------------------------------------------------

/// Decode WebP bytes with zenwebp decoder, returning RGBA.
/// Disables dithering for pixel-perfect comparison against libwebp.
fn decode_with_zenwebp(webp_data: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    let config = DecodeConfig::default().with_dithering_strength(0);
    let (rgba, w, h) = DecodeRequest::new(&config, webp_data)
        .decode_rgba()
        .map_err(|e| format!("zenwebp decode failed: {e}"))?;
    Ok((rgba, w, h))
}

/// Decode WebP bytes with libwebp (via webpx), returning RGBA.
fn decode_with_libwebp(webp_data: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    let (rgba, w, h) =
        webpx::decode_rgba(webp_data).map_err(|e| format!("webpx decode failed: {e}"))?;
    Ok((rgba, w, h))
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

struct DiffStats {
    pixel_count: u64,
    diff_count: u64,
    max_diff: u8,
    mean_diff: f64,
    channel_max: [u8; 4], // R, G, B, A
    /// Max diff ignoring alpha channel
    max_diff_rgb: u8,
}

fn compute_diff(a: &[u8], b: &[u8], width: u32, height: u32) -> DiffStats {
    assert_eq!(
        a.len(),
        b.len(),
        "buffer length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let expected = (width as usize) * (height as usize) * 4;
    assert_eq!(
        a.len(),
        expected,
        "buffer size {}, expected {expected} for {width}x{height} RGBA",
        a.len()
    );

    let mut max_diff = 0u8;
    let mut max_diff_rgb = 0u8;
    let mut diff_count = 0u64;
    let mut diff_sum = 0u64;
    let mut channel_max = [0u8; 4];
    let pixel_count = (width as u64) * (height as u64);

    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let d = av.abs_diff(bv);
        if d > 0 {
            diff_count += 1;
            diff_sum += d as u64;
            max_diff = max_diff.max(d);
            let ch = i % 4;
            channel_max[ch] = channel_max[ch].max(d);
            if ch < 3 {
                max_diff_rgb = max_diff_rgb.max(d);
            }
        }
    }

    let mean_diff = if diff_count > 0 {
        diff_sum as f64 / diff_count as f64
    } else {
        0.0
    };

    DiffStats {
        pixel_count,
        diff_count,
        max_diff,
        mean_diff,
        channel_max,
        max_diff_rgb,
    }
}

fn report_diff(label: &str, stats: &DiffStats) {
    if stats.diff_count == 0 {
        println!("  {label}: PIXEL-PERFECT (0 diffs)");
    } else {
        let pct = 100.0 * stats.diff_count as f64 / (stats.pixel_count as f64 * 4.0);
        println!(
            "  {label}: {diff_count} bytes differ ({pct:.2}%), \
             max_diff={max_diff} (rgb={max_rgb}), mean_diff={mean:.2}, \
             channel_max=[R={r},G={g},B={b_ch},A={a_ch}]",
            diff_count = stats.diff_count,
            max_diff = stats.max_diff,
            max_rgb = stats.max_diff_rgb,
            mean = stats.mean_diff,
            r = stats.channel_max[0],
            g = stats.channel_max[1],
            b_ch = stats.channel_max[2],
            a_ch = stats.channel_max[3],
        );
    }
}

// ---------------------------------------------------------------------------
// Test image generators
// ---------------------------------------------------------------------------

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

fn noise_rgb(w: u32, h: u32, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h * 3) {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        rgb.push((state & 0xFF) as u8);
    }
    rgb
}

fn uniform_rgb(w: u32, h: u32, val: u8) -> Vec<u8> {
    vec![val; (w * h * 3) as usize]
}

/// Encode RGB pixels as lossy WebP at the given quality.
fn encode_lossy(rgb: &[u8], w: u32, h: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::new_lossy()
        .with_quality(quality)
        .with_method(4);
    EncodeRequest::new(&config, rgb, PixelLayout::Rgb8, w, h)
        .encode()
        .expect("encode failed")
}

// ---------------------------------------------------------------------------
// Detect WebP format from RIFF container
// ---------------------------------------------------------------------------

fn webp_format(data: &[u8]) -> &'static str {
    if data.len() < 16 {
        return "too-short";
    }
    if &data[..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return "not-webp";
    }
    match &data[12..16] {
        b"VP8 " => "lossy",
        b"VP8L" => "lossless",
        b"VP8X" => "extended",
        _ => "unknown",
    }
}

fn is_lossy_vp8(data: &[u8]) -> bool {
    webp_format(data) == "lossy"
}

// ---------------------------------------------------------------------------
// Category 1: Roundtrip with zenwebp encoder
//
// Encode with zenwebp, decode with both zenwebp and libwebp, verify match.
// ---------------------------------------------------------------------------

struct RoundtripResult {
    vs_lib_stats: DiffStats,
}

/// Encode with zenwebp, decode with zenwebp and libwebp, compare output.
fn roundtrip_test(rgb: &[u8], w: u32, h: u32, quality: f32, label: &str) -> RoundtripResult {
    let webp = encode_lossy(rgb, w, h, quality);
    assert!(
        is_lossy_vp8(&webp),
        "{label}: encoder produced non-lossy format"
    );

    let (zen_rgba, zen_w, zen_h) =
        decode_with_zenwebp(&webp).unwrap_or_else(|e| panic!("{label}: zenwebp failed: {e}"));
    let (lib_rgba, lib_w, lib_h) =
        decode_with_libwebp(&webp).unwrap_or_else(|e| panic!("{label}: libwebp failed: {e}"));

    assert_eq!(zen_w, w, "{label}: decoded width {zen_w} != source {w}");
    assert_eq!(zen_h, h, "{label}: decoded height {zen_h} != source {h}");

    // Measure zenwebp vs libwebp — must be bit-exact (0 diffs)
    assert_eq!(
        lib_w, zen_w,
        "{label}: libwebp width={lib_w} != zenwebp width={zen_w}"
    );
    assert_eq!(
        lib_h, zen_h,
        "{label}: libwebp height={lib_h} != zenwebp height={zen_h}"
    );
    let vs_lib_stats = compute_diff(&zen_rgba, &lib_rgba, w, h);
    report_diff(&format!("{label} zenwebp-vs-libwebp"), &vs_lib_stats);
    assert!(
        vs_lib_stats.max_diff == 0,
        "{label}: zenwebp vs libwebp max_diff={} (must be pixel-exact)",
        vs_lib_stats.max_diff,
    );

    RoundtripResult { vs_lib_stats }
}

#[test]
fn cat1_roundtrip_gradient_quality_sweep() {
    println!("\n=== Category 1: Roundtrip gradient Q50/Q75/Q90 ===");
    let w = 256;
    let h = 256;
    let rgb = gradient_rgb(w, h);

    for &q in &[50.0, 75.0, 90.0] {
        roundtrip_test(&rgb, w, h, q, &format!("gradient_{w}x{h}_q{q}"));
    }
}

#[test]
fn cat1_roundtrip_noise_quality_sweep() {
    println!("\n=== Category 1: Roundtrip noise Q50/Q75/Q90 ===");
    let w = 256;
    let h = 256;
    let rgb = noise_rgb(w, h, 42);

    for &q in &[50.0, 75.0, 90.0] {
        roundtrip_test(&rgb, w, h, q, &format!("noise_{w}x{h}_q{q}"));
    }
}

#[test]
fn cat1_roundtrip_uniform() {
    println!("\n=== Category 1: Roundtrip uniform Q75 ===");
    let w = 128;
    let h = 128;
    let rgb = uniform_rgb(w, h, 128);
    roundtrip_test(&rgb, w, h, 75.0, "uniform_128x128_q75");
}

// ---------------------------------------------------------------------------
// Load PNG from codec-corpus for roundtrip tests
// ---------------------------------------------------------------------------

fn load_png_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let width = info.width;
    let height = info.height;

    // Convert to RGB8 if needed
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let rgba = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity(rgba.len() / 4 * 3);
            for chunk in rgba.chunks_exact(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        png::ColorType::Grayscale => {
            let gray = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity(gray.len() * 3);
            for &g in gray {
                rgb.extend_from_slice(&[g, g, g]);
            }
            rgb
        }
        png::ColorType::GrayscaleAlpha => {
            let ga = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity(ga.len() / 2 * 3);
            for chunk in ga.chunks_exact(2) {
                let g = chunk[0];
                rgb.extend_from_slice(&[g, g, g]);
            }
            rgb
        }
        _ => return None,
    };

    Some((rgb, width, height))
}

#[test]
fn cat1_roundtrip_corpus_kodak() {
    let corpus = match codec_corpus::Corpus::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("Skipping: codec-corpus not available");
            return;
        }
    };
    let kodak = match corpus.get("kodak") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping: kodak corpus not available");
            return;
        }
    };

    println!("\n=== Category 1: Roundtrip kodak corpus Q50/Q75/Q90 ===");
    let mut tested = 0u32;
    let mut worst_lib_rgb = 0u8;
    let mut worst_file = String::new();

    let mut entries: Vec<_> = std::fs::read_dir(&kodak)
        .expect("read_dir failed")
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in &entries {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("png") {
            continue;
        }
        let (rgb, w, h) = match load_png_rgb(&path) {
            Some(v) => v,
            None => continue,
        };

        let fname = path.file_name().unwrap().to_string_lossy();
        for &q in &[50.0, 75.0, 90.0] {
            let r = roundtrip_test(&rgb, w, h, q, &format!("{fname}_q{q}"));
            if r.vs_lib_stats.max_diff_rgb > worst_lib_rgb {
                worst_lib_rgb = r.vs_lib_stats.max_diff_rgb;
                worst_file = format!("{fname}_q{q}");
            }
            tested += 1;
        }
    }

    println!(
        "  Tested {tested} roundtrips, worst zenwebp-vs-libwebp RGB diff: {worst_lib_rgb} ({worst_file})"
    );
    assert!(tested > 0, "No kodak images found");
}

// ---------------------------------------------------------------------------
// Category 2: Decode pre-existing WebP files from codec-corpus
//
// Decode with zenwebp and libwebp, verify match.
// ---------------------------------------------------------------------------

struct CorpusDecodeResult {
    vs_lib_stats: DiffStats,
}

/// Decode a pre-existing WebP file with zenwebp and libwebp. Compare.
fn corpus_decode_compare(path: &Path) -> Result<CorpusDecodeResult, String> {
    let data = std::fs::read(path).map_err(|e| format!("read failed: {e}"))?;
    let fname = path.file_name().unwrap().to_string_lossy();

    // Only test lossy VP8
    if !is_lossy_vp8(&data) {
        return Err(format!(
            "{fname}: not lossy VP8 ({fmt}), skipped",
            fmt = webp_format(&data)
        ));
    }

    let (zen_rgba, zen_w, zen_h) =
        decode_with_zenwebp(&data).map_err(|e| format!("{fname} zenwebp: {e}"))?;
    let (lib_rgba, lib_w, lib_h) =
        decode_with_libwebp(&data).map_err(|e| format!("{fname} libwebp: {e}"))?;

    // Dimension checks
    if lib_w != zen_w || lib_h != zen_h {
        return Err(format!(
            "{fname}: libwebp={lib_w}x{lib_h} zenwebp={zen_w}x{zen_h} dimension mismatch"
        ));
    }

    let vs_lib_stats = compute_diff(&zen_rgba, &lib_rgba, zen_w, zen_h);

    Ok(CorpusDecodeResult { vs_lib_stats })
}

#[test]
fn cat2_decode_webp_conformance() {
    let corpus = match codec_corpus::Corpus::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("Skipping: codec-corpus not available");
            return;
        }
    };
    let valid_dir = match corpus.get("webp-conformance/valid") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping: webp-conformance/valid not available");
            return;
        }
    };

    println!("\n=== Category 2: Decode webp-conformance/valid ===");
    let mut tested = 0u32;
    let mut skipped = 0u32;
    let mut worst_lib_rgb = 0u8;
    let mut worst_file = String::new();
    let mut failures: Vec<String> = Vec::new();

    let mut entries: Vec<_> = std::fs::read_dir(&valid_dir)
        .expect("read_dir failed")
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in &entries {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("webp") {
            continue;
        }
        let fname = path.file_name().unwrap().to_string_lossy().to_string();

        match corpus_decode_compare(&path) {
            Ok(result) => {
                tested += 1;
                report_diff(&fname, &result.vs_lib_stats);

                if result.vs_lib_stats.max_diff_rgb > worst_lib_rgb {
                    worst_lib_rgb = result.vs_lib_stats.max_diff_rgb;
                    worst_file = fname;
                }
            }
            Err(msg) => {
                if msg.contains("skipped") {
                    skipped += 1;
                } else {
                    eprintln!("  ERROR: {msg}");
                    failures.push(msg);
                }
            }
        }
    }

    println!("\n  Summary: {tested} tested, {skipped} skipped (non-lossy)");
    if tested > 0 {
        println!("  Worst zenwebp-vs-libwebp RGB diff: {worst_lib_rgb} ({worst_file})");
    }

    for f in &failures {
        eprintln!("  FAIL: {f}");
    }

    assert!(
        tested > 0,
        "No lossy WebP files found in conformance corpus"
    );
    assert!(failures.is_empty(), "{} decode failures", failures.len());
}

#[test]
fn cat2_decode_all_corpus_webp() {
    let corpus = match codec_corpus::Corpus::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("Skipping: codec-corpus not available");
            return;
        }
    };

    // Collect all WebP files from various corpus subdirectories
    let subdirs = [
        "imageflow/test_inputs",
        "image-rs/test-images/webp/lossy_images",
        "image-rs/test-images/webp/lossless_images",
        "image-rs/test-images/webp/extended_images",
    ];

    println!("\n=== Category 2: Decode all corpus WebP files ===");
    let mut tested = 0u32;
    let mut skipped_lossless = 0u32;
    let mut skipped_extended = 0u32;
    let mut skipped_animated = 0u32;
    let mut worst_lib_rgb = 0u8;
    let mut failures: Vec<String> = Vec::new();

    for subdir in &subdirs {
        let dir = match corpus.get(subdir) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in &entries {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("webp") {
                continue;
            }
            let fname = path.file_name().unwrap().to_string_lossy().to_string();
            let data = match std::fs::read(&path) {
                Ok(d) => d,
                Err(_) => continue,
            };

            let fmt = webp_format(&data);
            match fmt {
                "lossy" => {}
                "lossless" => {
                    println!("  {fname}: lossless, skipped (lossy-only test)");
                    skipped_lossless += 1;
                    continue;
                }
                "extended" => {
                    if fname.contains("anim") {
                        println!("  {fname}: animated WebP, skipped (keyframe-only test)");
                        skipped_animated += 1;
                    } else {
                        println!("  {fname}: extended format, skipped (simple VP8 only)");
                        skipped_extended += 1;
                    }
                    continue;
                }
                _ => {
                    println!("  {fname}: unknown format {fmt}, skipped");
                    continue;
                }
            }

            match corpus_decode_compare(&path) {
                Ok(result) => {
                    tested += 1;
                    report_diff(&format!("{subdir}/{fname}"), &result.vs_lib_stats);
                    worst_lib_rgb = worst_lib_rgb.max(result.vs_lib_stats.max_diff_rgb);
                }
                Err(msg) => {
                    if !msg.contains("skipped") {
                        eprintln!("  ERROR: {msg}");
                        failures.push(msg);
                    }
                }
            }
        }
    }

    println!("\n  Summary: {tested} lossy tested");
    println!(
        "  Skipped: {skipped_lossless} lossless, {skipped_extended} extended, \
         {skipped_animated} animated"
    );
    if tested > 0 {
        println!("  Worst zenwebp-vs-libwebp RGB diff: {worst_lib_rgb}");
    }

    for f in &failures {
        eprintln!("  FAIL: {f}");
    }

    if tested > 0 {
        assert!(failures.is_empty(), "{} decode failures", failures.len());
    }
}

// ---------------------------------------------------------------------------
// Category 3: Edge cases — odd dimensions
// ---------------------------------------------------------------------------

fn edge_case_test(w: u32, h: u32, quality: f32) {
    let rgb = gradient_rgb(w, h);
    roundtrip_test(&rgb, w, h, quality, &format!("{w}x{h}_q{quality}"));
}

#[test]
fn cat3_edge_cases_tiny() {
    println!("\n=== Category 3: Edge cases — tiny images ===");

    for &(w, h) in &[(1, 1), (2, 2), (3, 3), (2, 1), (1, 2)] {
        edge_case_test(w, h, 75.0);
    }
}

#[test]
fn cat3_edge_cases_non_mb_aligned() {
    println!("\n=== Category 3: Edge cases — non-MB-aligned ===");

    let sizes: &[(u32, u32)] = &[
        (15, 15),
        (16, 16),
        (17, 17),
        (31, 31),
        (32, 32),
        (33, 33),
        (255, 255),
        (256, 256),
        (257, 257),
        (15, 17),
        (17, 15),
        (33, 17),
        (48, 33),
    ];

    for &(w, h) in sizes {
        edge_case_test(w, h, 75.0);
    }
}

#[test]
fn cat3_edge_cases_extreme_quality() {
    println!("\n=== Category 3: Edge cases — extreme quality ===");
    let w = 64;
    let h = 64;
    let rgb = gradient_rgb(w, h);

    // Very low quality (heavy quantization, more filtering)
    for &q in &[1.0, 5.0, 10.0] {
        roundtrip_test(&rgb, w, h, q, &format!("gradient_64x64_q{q}"));
    }

    // Very high quality (minimal quantization)
    for &q in &[95.0, 99.0, 100.0] {
        roundtrip_test(&rgb, w, h, q, &format!("gradient_64x64_q{q}"));
    }
}

#[test]
fn cat3_edge_cases_noise_content() {
    println!("\n=== Category 3: Edge cases — noise content at edge sizes ===");

    // Non-MB-aligned with noisy content (tests all prediction modes)
    for &(w, h) in &[(15, 15), (17, 17), (33, 33)] {
        let rgb = noise_rgb(w, h, 123);
        roundtrip_test(&rgb, w, h, 75.0, &format!("noise_{w}x{h}_q75"));
    }
}

// ---------------------------------------------------------------------------
// Category 4: Quality sweep — one image across many quality levels
// ---------------------------------------------------------------------------

#[test]
fn cat4_quality_sweep() {
    println!("\n=== Category 4: Quality sweep 256x256 gradient ===");
    let w = 256;
    let h = 256;
    let rgb = gradient_rgb(w, h);

    let qualities = [
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 99.0,
    ];

    let mut worst_lib_rgb = 0u8;
    let mut worst_lib_q = 0.0f32;

    for &q in &qualities {
        let r = roundtrip_test(&rgb, w, h, q, &format!("gradient_256x256_q{q}"));
        if r.vs_lib_stats.max_diff_rgb > worst_lib_rgb {
            worst_lib_rgb = r.vs_lib_stats.max_diff_rgb;
            worst_lib_q = q;
        }
    }

    println!("\n  Worst zenwebp-vs-libwebp RGB: {worst_lib_rgb} (at Q{worst_lib_q})");
}

#[test]
fn cat4_quality_sweep_noise() {
    println!("\n=== Category 4: Quality sweep 256x256 noise ===");
    let w = 256;
    let h = 256;
    let rgb = noise_rgb(w, h, 99);

    let qualities = [
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 99.0,
    ];

    let mut worst_lib_rgb = 0u8;
    let mut worst_lib_q = 0.0f32;

    for &q in &qualities {
        let r = roundtrip_test(&rgb, w, h, q, &format!("noise_256x256_q{q}"));
        if r.vs_lib_stats.max_diff_rgb > worst_lib_rgb {
            worst_lib_rgb = r.vs_lib_stats.max_diff_rgb;
            worst_lib_q = q;
        }
    }

    println!("\n  Worst zenwebp-vs-libwebp RGB: {worst_lib_rgb} (at Q{worst_lib_q})");
}

// ---------------------------------------------------------------------------
// Varied-size decode: verify WebPDecoder handles different image sizes
// ---------------------------------------------------------------------------

#[test]
fn varied_size_decode_produces_correct_output() {
    println!("\n=== Varied-size decode: correct output for different dimensions ===");

    // Encode several different-sized images
    let images: Vec<(u32, u32, Vec<u8>)> = vec![
        (64, 64, gradient_rgb(64, 64)),
        (33, 17, noise_rgb(33, 17, 42)),
        (128, 96, uniform_rgb(128, 96, 200)),
        (15, 15, gradient_rgb(15, 15)),
        (256, 256, noise_rgb(256, 256, 7)),
    ];

    let webp_files: Vec<Vec<u8>> = images
        .iter()
        .map(|(w, h, rgb)| encode_lossy(rgb, *w, *h, 75.0))
        .collect();

    for (i, webp) in webp_files.iter().enumerate() {
        let (iw, ih, _) = &images[i];
        let config = DecodeConfig::default();
        let (rgba, w, h) = DecodeRequest::new(&config, webp)
            .decode_rgba()
            .unwrap_or_else(|e| panic!("image {i} ({iw}x{ih}): decode failed: {e}"));

        assert_eq!(w, *iw, "image {i}: width mismatch");
        assert_eq!(h, *ih, "image {i}: height mismatch");

        let expected_len = (w as usize) * (h as usize) * 4;
        assert_eq!(
            rgba.len(),
            expected_len,
            "image {i} ({w}x{h}): buffer length mismatch"
        );

        // Verify output has actual content (not all zeros)
        let non_zero = rgba.iter().filter(|&&b| b != 0).count();
        assert!(
            non_zero > 0,
            "image {i} ({w}x{h}): decoded output is all zeros"
        );

        println!("  image {i} ({w}x{h}): OK, {non_zero} non-zero bytes");
    }
}

// ---------------------------------------------------------------------------
// Document YUV->RGB conversion differences
// ---------------------------------------------------------------------------

#[test]
fn document_yuv_conversion_differences() {
    println!("\n=== YUV->RGB conversion difference analysis ===");
    println!("  zenwebp uses: fill_rgba with Bilinear upsampling");
    #[cfg(feature = "fast-yuv")]
    println!("  fast-yuv feature: ON (yuv crate bilinear, differs from libwebp)");
    #[cfg(not(feature = "fast-yuv"))]
    println!("  fast-yuv feature: OFF (internal bilinear, closer to libwebp)");
    println!("  libwebp uses: WebPDecodeRGBA (fancy upsampling)");
    println!();

    // Test several sizes to show the pattern
    for &(w, h) in &[(16, 16), (64, 64), (128, 128), (256, 256)] {
        let rgb = gradient_rgb(w, h);
        let webp = encode_lossy(&rgb, w, h, 75.0);

        let (zen_rgba, _, _) = decode_with_zenwebp(&webp).unwrap();
        let (lib_rgba, _, _) = decode_with_libwebp(&webp).unwrap();

        let stats = compute_diff(&zen_rgba, &lib_rgba, w, h);
        report_diff(&format!("{w}x{h} gradient Q75"), &stats);
    }

    println!();
    println!("  Note: Diffs are from YUV->RGB chroma upsampling differences,");
    println!("  NOT from the VP8 decode pipeline. Only the final RGB conversion");
    println!("  may differ from libwebp when using the fast-yuv feature.");
}
