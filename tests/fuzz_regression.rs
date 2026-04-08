//! Fuzz crash regression suite.
//!
//! Runs every file in `fuzz/regression/` through every decoder entry point that
//! has a fuzz target. Each seed file is a previously-found crash that has been
//! fixed; this test ensures none of them re-introduce a panic.
//!
//! Reproduces what the `decode_still`, `decode_animated`, and `decode_v2` fuzz
//! targets do, but as a regular `cargo test` — no nightly toolchain needed.
//! Failures here mean a regression of a previously-fixed bug.
//!
//! To add a new seed: drop the (preferably minimized) crash file into
//! `fuzz/regression/` with a `crash-<sha>` name, no other action required.

use std::fs;
use std::path::PathBuf;

/// Cap the buffer the decoder is allowed to allocate, mirroring the fuzz targets.
const MAX_PIXEL_BYTES: usize = 1024 * 1024 * 1024;

fn regression_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/regression")
}

fn run_decode_still(input: &[u8]) {
    if let Ok(mut decoder) = zenwebp::WebPDecoder::new(input) {
        let (width, height) = decoder.dimensions();
        let bytes_per_pixel = if decoder.has_alpha() { 4 } else { 3 };
        let total_bytes = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(bytes_per_pixel);
        if total_bytes <= MAX_PIXEL_BYTES {
            let mut data = vec![0u8; total_bytes];
            let _ = decoder.read_image(&mut data);
        }
    }
}

fn run_decode_animated(input: &[u8]) {
    if let Ok(mut decoder) = zenwebp::WebPDecoder::new(input) {
        let (width, height) = decoder.dimensions();
        let bytes_per_pixel = if decoder.has_alpha() { 4 } else { 3 };
        let total_bytes = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(bytes_per_pixel);
        if total_bytes <= MAX_PIXEL_BYTES && decoder.is_animated() {
            let mut data = vec![0u8; total_bytes];
            while let Ok(_delay) = decoder.read_frame(&mut data) {}
        }
    }
}

fn run_decode_v2(input: &[u8]) {
    let _ = zenwebp::oneshot::decode_rgba(input);
    let _ = zenwebp::oneshot::decode_rgb(input);
    if let Ok(mut decoder) = zenwebp::mux::AnimationDecoder::new(input) {
        while let Ok(Some(_frame)) = decoder.next_frame() {}
    }
}

#[test]
fn fuzz_regression_seeds_do_not_panic() {
    let dir = regression_dir();
    let entries: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .collect();

    assert!(
        !entries.is_empty(),
        "fuzz/regression/ is empty — at least the consume_unchecked seeds should be present"
    );

    for entry in entries {
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<unnamed>");
        let input = fs::read(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));

        // Each entry point may return Err but must not panic. If any panics,
        // the test fails with the seed name in the unwind message.
        run_decode_still(&input);
        run_decode_animated(&input);
        run_decode_v2(&input);

        eprintln!("ok: {name} ({} bytes)", input.len());
    }
}
