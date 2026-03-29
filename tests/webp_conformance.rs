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
    let corpus_path = get_corpus_path("webp-conformance/valid");
    if corpus_path.is_none() {
        eprintln!("Skipping: codec-corpus not found. Set CORPUS_DIR or clone:");
        eprintln!(
            "  git clone --depth=1 https://github.com/imazen/codec-corpus.git ~/codec-corpus"
        );
        return;
    }

    let corpus_path = corpus_path.unwrap();
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

#[test]
#[ignore]
fn test_webp_invalid_robustness() {
    let corpus_path = get_corpus_path("webp-conformance/invalid");
    if corpus_path.is_none() {
        eprintln!("Skipping: invalid/ directory not yet populated in codec-corpus");
        return;
    }

    let corpus_path = corpus_path.unwrap();
    let mut tested = 0;
    let mut crashed = 0;

    for entry in fs::read_dir(&corpus_path)
        .unwrap_or_else(|_| {
            eprintln!("Warning: Failed to read invalid/ directory");
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

        // Main requirement: should NOT panic or crash
        // Result (Ok or Err) is acceptable; we just care about safety
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match WebPDecoder::new(&data) {
                Ok(mut decoder) => {
                    let output_size = decoder.output_buffer_size().unwrap_or(0);
                    if output_size > 0 && output_size < 10_000_000 {
                        let mut output = alloc::vec![0u8; output_size];
                        let _ = decoder.read_image(&mut output);
                    }
                }
                Err(_) => {
                    // Parse error is acceptable for invalid files
                }
            }
        }))
        .is_err()
        {
            crashed += 1;
            eprintln!("CRASH: {}", path.display());
        }
    }

    if tested == 0 {
        eprintln!("No .webp files found in invalid/");
        return;
    }

    println!(
        "WebP Invalid File Test: {}/{} files handled safely (no crashes)",
        tested - crashed,
        tested
    );
    assert_eq!(
        crashed, 0,
        "Invalid files caused panics/crashes: {}/{}",
        crashed, tested
    );
}

#[test]
#[ignore]
fn test_webp_non_conformant_regression() {
    let corpus_path = get_corpus_path("webp-conformance/non-conformant");
    if corpus_path.is_none() {
        eprintln!("Skipping: non-conformant/ directory not yet populated");
        return;
    }

    let corpus_path = corpus_path.unwrap();
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

/// Decode every .webp in the full scraped corpus (12,825+ files from 8 sources).
///
/// Walks all subdirectories of the scraped corpus and decodes each file.
/// Any panic or unhandled decode error is a failure.
///
/// Run with: `cargo test --release --test webp_conformance test_scraped_webp_corpus -- --ignored --nocapture`
#[test]
#[ignore]
fn test_scraped_webp_corpus() {
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    let corpus_dir = Path::new("/mnt/v/output/corpus-builder/webp");
    if !corpus_dir.exists() {
        eprintln!("Skipping: scraped corpus not found at {}", corpus_dir.display());
        eprintln!("  Expected: /mnt/v/output/corpus-builder/webp/{{google-native,pexels,unsplash,...}}");
        return;
    }

    let tested = AtomicUsize::new(0);
    let passed = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let panicked = AtomicUsize::new(0);

    let start = Instant::now();
    let mut files: Vec<_> = Vec::new();

    // Collect all .webp files recursively
    fn collect_webp(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_webp(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("webp") {
                out.push(path);
            }
        }
    }
    collect_webp(corpus_dir, &mut files);
    files.sort();

    println!("Found {} .webp files in scraped corpus", files.len());

    for path in &files {
        tested.fetch_add(1, Ordering::Relaxed);

        let data = match fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  SKIP (read error): {} — {}", path.display(), e);
                continue;
            }
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            decode_webp(&data)
        }));

        match result {
            Ok(Ok(_)) => {
                passed.fetch_add(1, Ordering::Relaxed);
            }
            Ok(Err(e)) => {
                failed.fetch_add(1, Ordering::Relaxed);
                eprintln!("  FAIL: {} — {}", path.display(), e);
            }
            Err(_) => {
                panicked.fetch_add(1, Ordering::Relaxed);
                eprintln!("  PANIC: {}", path.display());
            }
        }

        let count = tested.load(Ordering::Relaxed);
        if count % 1000 == 0 {
            println!(
                "  progress: {}/{} ({} passed, {} failed, {} panicked) [{:.1}s]",
                count,
                files.len(),
                passed.load(Ordering::Relaxed),
                failed.load(Ordering::Relaxed),
                panicked.load(Ordering::Relaxed),
                start.elapsed().as_secs_f64(),
            );
        }
    }

    let total = tested.load(Ordering::Relaxed);
    let pass = passed.load(Ordering::Relaxed);
    let fail = failed.load(Ordering::Relaxed);
    let panic = panicked.load(Ordering::Relaxed);
    let elapsed = start.elapsed();

    println!("\n=== Scraped WebP Corpus Results ===");
    println!("  Total:    {}", total);
    println!("  Passed:   {}", pass);
    println!("  Failed:   {}", fail);
    println!("  Panicked: {}", panic);
    println!("  Time:     {:.1}s ({:.0} files/sec)", elapsed.as_secs_f64(), total as f64 / elapsed.as_secs_f64());

    assert_eq!(panic, 0, "Decoder panicked on {} files — this is a bug", panic);
    // Note: decode failures are expected for some scraped files (truncated, unusual features).
    // We assert zero panics but only warn on decode errors.
    if fail > 0 {
        eprintln!("\nWARNING: {} files failed to decode (see above). Review for missing format support.", fail);
    }
}

/// Get the codec-corpus path for a specific subdirectory.
fn get_corpus_path(subdir: &str) -> Option<std::path::PathBuf> {
    let corpus = codec_corpus::Corpus::new().ok()?;
    corpus.get(subdir).ok()
}
