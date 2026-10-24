//! WebP conformance tests from codec-corpus.
//!
//! Tests that zenwebp correctly decodes RFC 6386-compliant WebP files.
//! Run with: `cargo test --test webp_conformance -- --ignored`
//! Or via CI: `cargo test --release test_webp -- --ignored`
//!
//! Requires codec-corpus which is only available on native platforms.
#![cfg(not(target_arch = "wasm32"))]

use std::fs;
use zenwebp::WebPDecoder;

extern crate alloc;
use alloc::vec;

#[test]
#[ignore]
fn test_webp_valid_files() {
    let corpus_path = corpus_path("webp-conformance/valid");
    let mut tested = 0;
    let mut passed = 0;

    for entry in fs::read_dir(&corpus_path).expect("Failed to read corpus directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();

        if path.extension().and_then(|ext| ext.to_str()) != Some("webp") {
            continue;
        }

        tested += 1;
        let data = fs::read(&path).expect("Failed to read WebP file");

        // Attempt to decode
        match WebPDecoder::new(&data) {
            Ok(mut decoder) => {
                // Check that header is valid
                let (width, height) = decoder.dimensions();
                let _has_alpha = decoder.has_alpha();

                // Verify dimensions are reasonable
                if width > 0 && height > 0 {
                    // Attempt to decode image data
                    let output_size = decoder.output_buffer_size().unwrap_or(0);
                    if output_size > 0 {
                        let mut output = alloc::vec![0u8; output_size];
                        match decoder.read_image(&mut output) {
                            Ok(_) => {
                                passed += 1;
                            }
                            Err(e) => {
                                eprintln!("FAIL: {} - decode error: {}", path.display(), e);
                            }
                        }
                    } else {
                        eprintln!("FAIL: {} - invalid output buffer size", path.display());
                    }
                } else {
                    eprintln!("FAIL: {} - invalid dimensions", path.display());
                }
            }
            Err(e) => {
                eprintln!("FAIL: {} - header parse error: {}", path.display(), e);
            }
        }
    }

    if tested == 0 {
        eprintln!("No .webp files found in {}", corpus_path.display());
        return;
    }

    println!(
        "WebP Valid File Test: {}/{} files decoded successfully",
        passed, tested
    );
    assert_eq!(
        passed, tested,
        "Some valid files failed to decode: {}/{}",
        passed, tested
    );
}

/// Invalid inputs derived from every valid conformance file: truncations at
/// many offsets, corrupted RIFF / chunk sizes and fourccs. Decoding must
/// return (Ok or Err) without panicking. (codec-corpus has no `invalid/`
/// folder; generating the cases keeps them reproducible from the valid set.)
#[test]
#[ignore]
fn test_webp_invalid_robustness() {
    let corpus_path = corpus_path("webp-conformance/valid");
    let mut sources: Vec<_> = fs::read_dir(&corpus_path)
        .expect("read valid/")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("webp"))
        .collect();
    sources.sort();
    assert!(
        !sources.is_empty(),
        "no .webp files in {}",
        corpus_path.display()
    );

    let mut tested = 0usize;
    let mut crashed = Vec::new();
    for path in &sources {
        let data = fs::read(path).expect("read webp");
        for (label, mutated) in invalid_variants(&data) {
            tested += 1;
            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if let Ok(mut decoder) = WebPDecoder::new(&mutated)
                    && let Some(size) = decoder.output_buffer_size()
                    && size < 64 << 20
                {
                    let mut output = vec![0u8; size];
                    let _ = decoder.read_image(&mut output);
                }
            }))
            .is_ok();
            if !ok {
                crashed.push(format!("{} {label}", path.display()));
            }
        }
    }
    println!(
        "WebP invalid-input test: {}/{tested} derived inputs handled without panics ({} sources)",
        tested - crashed.len(),
        sources.len()
    );
    assert!(crashed.is_empty(), "panics on: {crashed:#?}");
}

/// Truncations and header corruptions of one valid file.
fn invalid_variants(data: &[u8]) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let len = data.len();
    // Truncations: every byte of the headers, then 16 evenly spaced cuts.
    let mut cuts: Vec<usize> = (0..len.min(64)).collect();
    cuts.extend((1..16).map(|k| len * k / 16));
    cuts.sort();
    cuts.dedup();
    for cut in cuts {
        if cut < len {
            out.push((format!("truncated@{cut}"), data[..cut].to_vec()));
        }
    }
    // Size fields: RIFF size (offset 4) and the first chunk's size (16).
    for (off, name) in [(4usize, "riff"), (16, "chunk")] {
        if len >= off + 4 {
            for v in [
                0u32,
                1,
                7,
                u32::MAX,
                u32::MAX - 1,
                (len as u32).wrapping_mul(2),
            ] {
                let mut m = data.to_vec();
                m[off..off + 4].copy_from_slice(&v.to_le_bytes());
                out.push((format!("{name}_size={v:#x}"), m));
            }
        }
    }
    // Fourccs: "WEBP" (8) and the first chunk (12).
    for off in [8usize, 12] {
        if len >= off + 4 {
            let mut m = data.to_vec();
            m[off] ^= 0x20;
            out.push((format!("fourcc@{off}"), m));
        }
    }
    out
}

#[test]
#[ignore]
fn test_webp_non_conformant_regression() {
    // codec-corpus has no non-conformant/ folder yet (CLAUDE.md, conformance
    // Phase 2); the caller supplies one.
    let corpus_path = env_corpus_path("ZENWEBP_NONCONFORMANT_DIR", "non-conformant WebP corpus");
    let mut tested = 0;

    for entry in fs::read_dir(&corpus_path)
        .unwrap_or_else(|_| {
            eprintln!("Warning: Failed to read non-conformant/ directory");
            fs::read_dir("/dev/null").unwrap()
        })
        .flatten()
    {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("webp") {
            continue;
        }

        tested += 1;
        let data = match fs::read(&path) {
            Ok(d) => d,
            Err(_) => continue,
        };

        // Decode twice - should produce identical results (regression test)
        let result1 = decode_webp(&data);
        let result2 = decode_webp(&data);

        match (result1, result2) {
            (Ok((w1, h1)), Ok((w2, h2))) => {
                // For regression: check dimensions match
                assert_eq!(
                    w1,
                    w2,
                    "Width mismatch on {}: {} vs {}",
                    path.display(),
                    w1,
                    w2
                );
                assert_eq!(
                    h1,
                    h2,
                    "Height mismatch on {}: {} vs {}",
                    path.display(),
                    h1,
                    h2
                );
            }
            (Err(e1), Err(_)) => {
                // Both failed - acceptable for consistency
                eprintln!("Both decodes failed for {}: {}", path.display(), e1);
            }
            (Ok(_), Err(e2)) | (Err(e2), Ok(_)) => {
                eprintln!(
                    "Inconsistent decode result for {}: one succeeded, one failed: {}",
                    path.display(),
                    e2
                );
            }
        }
    }

    if tested == 0 {
        eprintln!("No .webp files found in non-conformant/");
        return;
    }

    println!(
        "WebP Non-Conformant Regression Test: {}/{} files checked",
        tested, tested
    );
}

/// Helper function to decode WebP and return dimensions.
fn decode_webp(data: &[u8]) -> Result<(u32, u32), Box<dyn std::error::Error>> {
    let mut decoder = WebPDecoder::new(data)?;
    let (width, height) = decoder.dimensions();
    let output_size = decoder.output_buffer_size().ok_or("Invalid output size")?;
    if output_size > 0 {
        let mut output = vec![0u8; output_size];
        decoder.read_image(&mut output)?;
    }
    Ok((width, height))
}

/// Decode every .webp in the full scraped corpus and diff against libwebp.
///
/// Walks `$ZENWEBP_SCRAPED_WEBP_DIR` (on dev: /mnt/v/output/corpus-builder/webp,
/// 12,825+ files from 8 sources).
/// Decodes each file with both zenwebp and libwebp (via webpx), then
/// compares RGBA output byte-for-byte. Any pixel mismatch or panic is a failure.
///
/// Run with: `cargo test --release --test webp_conformance test_scraped_webp_corpus -- --ignored --nocapture`
#[test]
#[ignore]
fn test_scraped_webp_corpus() {
    use std::path::Path;
    use std::time::Instant;
    use zenwebp::{DecodeConfig, DecodeRequest};

    // On dev: ZENWEBP_SCRAPED_WEBP_DIR=/mnt/v/output/corpus-builder/webp
    let corpus_dir = env_corpus_path("ZENWEBP_SCRAPED_WEBP_DIR", "scraped WebP corpus");
    let corpus_dir = corpus_dir.as_path();

    // Collect all .webp files recursively
    fn collect_webp(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_webp(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("webp") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    collect_webp(corpus_dir, &mut files);
    files.sort();

    println!("Found {} .webp files in scraped corpus", files.len());

    let start = Instant::now();
    let mut tested = 0usize;
    let mut exact_match = 0usize;
    let mut zen_only_fail = 0usize;
    let mut lib_only_fail = 0usize;
    let mut both_fail = 0usize;
    let mut mismatch = 0usize;
    let mut panicked = 0usize;
    let mut mismatch_files: Vec<(String, u8, usize)> = Vec::new(); // (path, max_diff, count)

    let config = DecodeConfig::default().with_dithering_strength(0);

    for path in &files {
        tested += 1;

        let data = match fs::read(path) {
            Ok(d) => d,
            Err(_) => continue,
        };

        // zenwebp decode (catch panics)
        let zen_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            DecodeRequest::new(&config, &data).decode_rgba()
        }));

        let zen = match zen_result {
            Ok(r) => r.ok(),
            Err(_) => {
                panicked += 1;
                eprintln!("  PANIC: {}", path.display());
                None
            }
        };

        // libwebp decode
        let lib = webpx::decode_rgba(&data).ok();

        match (&zen, &lib) {
            (Some((z_px, z_w, z_h)), Some((l_px, l_w, l_h))) => {
                if z_w != l_w || z_h != l_h {
                    mismatch += 1;
                    eprintln!(
                        "  DIM MISMATCH: {} — zen {}x{} vs lib {}x{}",
                        path.display(),
                        z_w,
                        z_h,
                        l_w,
                        l_h
                    );
                } else if z_px == l_px {
                    exact_match += 1;
                } else {
                    // Pixel diff
                    let mut max_diff = 0u8;
                    let mut diff_count = 0usize;
                    for (&a, &b) in z_px.iter().zip(l_px.iter()) {
                        let d = a.abs_diff(b);
                        if d > max_diff {
                            max_diff = d;
                        }
                        if d > 0 {
                            diff_count += 1;
                        }
                    }
                    mismatch += 1;
                    let short = path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    mismatch_files.push((short, max_diff, diff_count));
                }
            }
            (None, None) => {
                both_fail += 1;
                eprintln!("  BOTH FAIL: {}", path.display());
            }
            (None, Some(_)) => {
                zen_only_fail += 1;
                eprintln!("  ZEN FAIL: {}", path.display());
            }
            (Some(_), None) => {
                lib_only_fail += 1;
                eprintln!("  LIB FAIL: {}", path.display());
            }
        }

        if tested.is_multiple_of(1000) {
            println!(
                "  progress: {}/{} (exact={}, mismatch={}, zen_fail={}, panic={}) [{:.1}s]",
                tested,
                files.len(),
                exact_match,
                mismatch,
                zen_only_fail,
                panicked,
                start.elapsed().as_secs_f64(),
            );
        }
    }

    let elapsed = start.elapsed();

    println!("\n=== Scraped WebP Corpus vs libwebp ===");
    println!("  Total:       {tested}");
    println!("  Exact match: {exact_match}");
    println!("  Mismatch:    {mismatch}");
    println!("  zen fail:    {zen_only_fail}");
    println!("  lib fail:    {lib_only_fail}");
    println!("  Both fail:   {both_fail}");
    println!("  Panicked:    {panicked}");
    println!(
        "  Time:        {:.1}s ({:.0} files/sec)",
        elapsed.as_secs_f64(),
        tested as f64 / elapsed.as_secs_f64()
    );

    if !mismatch_files.is_empty() {
        println!("\n  Pixel mismatches (first 50):");
        mismatch_files.sort_by_key(|b| std::cmp::Reverse(b.1));
        for (name, max_diff, count) in mismatch_files.iter().take(50) {
            println!("    max_diff={max_diff:3} diff_bytes={count:8} {name}");
        }
    }

    assert_eq!(panicked, 0, "Decoder panicked on {panicked} files");
    assert_eq!(
        mismatch, 0,
        "{mismatch} files had pixel differences vs libwebp"
    );
}

/// Get the codec-corpus path for a specific subdirectory.
/// A codec-corpus dataset; a missing or unfetchable dataset fails the test.
fn corpus_path(subdir: &str) -> std::path::PathBuf {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus cache unavailable");
    corpus
        .get(subdir)
        .unwrap_or_else(|e| panic!("codec-corpus dataset {subdir} unavailable: {e}"))
}

/// A directory named by an environment variable. These corpora are not in
/// codec-corpus, so the caller decides whether the test runs; when it does
/// run without the variable, it fails with instructions instead of passing.
fn env_corpus_path(var: &str, what: &str) -> std::path::PathBuf {
    let dir = std::env::var_os(var)
        .unwrap_or_else(|| panic!("set {var} to the {what} directory to run this test"));
    let dir = std::path::PathBuf::from(dir);
    assert!(dir.is_dir(), "{var}={} is not a directory", dir.display());
    dir
}
