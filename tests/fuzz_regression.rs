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

/// Cap the buffer the `still`/`animated` regression helpers allocate.
///
/// The `decode_still`/`decode_animated` fuzz targets cap at 1 GB, but this is a
/// plain `cargo test` (no instrumentation) that runs every seed on every entry
/// point — so we keep it tight (64 MB) to stay fast. The #68 decompression-bomb
/// seed declares a 210 MB canvas; the still/animated helpers skip it here, while
/// `bomb_seed_rejected_quickly` proves the `decode_v2` path rejects it in microseconds.
const MAX_PIXEL_BYTES: usize = 64 * 1024 * 1024;

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

/// Restrictive limits for fuzzing — must match `fuzz/fuzz_targets/decode_v2.rs`.
///
/// The library's production default (`Limits::default()`, 120 MP / 1 GB) is the
/// right ceiling for real decoding, but a malformed header can declare a huge
/// canvas from a handful of bytes (a decompression bomb). Under libFuzzer's
/// sanitizer-coverage instrumentation, decoding tens of megapixels blows past the
/// 25s timeout even though the work is fully bounded and linear (issue #68: a
/// 104-byte input declaring 12801x4097 = 52 MP). The harness tightens the limits
/// so the fuzzer explores decode logic, not canvas size.
fn fuzz_config() -> zenwebp::DecodeConfig {
    let limits = zenwebp::Limits::default()
        .max_dimensions(4096, 4096)
        .max_total_pixels(4_000_000)
        .max_memory(64 * 1024 * 1024);
    zenwebp::DecodeConfig::default().limits(limits)
}

fn run_decode_v2(input: &[u8]) {
    let config = fuzz_config();
    let _ = zenwebp::DecodeRequest::new(&config, input).decode_rgba();
    let _ = zenwebp::DecodeRequest::new(&config, input).decode_rgb();
    if let Ok(mut decoder) = zenwebp::mux::AnimationDecoder::new_with_config(input, &config) {
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

/// Regression for #68: a tiny input declaring a giant canvas (a decompression
/// bomb) must not be decoded under the fuzzing limits — it must be rejected at
/// the dimension/memory check, fast, before any per-pixel work or big allocation.
///
/// The seed is `crash-...-issue68-bomb-52mp` (104 bytes, declares 12801x4097 =
/// 52 MP / 210 MB). With `Limits::default()` the library decodes it correctly in
/// ~0.5s on bare metal (that is the production contract and is intentionally not
/// changed); under libFuzzer instrumentation that is ~21s, over the 25s timeout.
/// Under the harness `fuzz_config()` limits it returns `Err` in microseconds.
#[test]
fn bomb_seed_rejected_quickly() {
    let dir = regression_dir();
    // Find the #68 bomb seed by name suffix so a future rename of the sha prefix
    // (e.g. after `cargo fuzz tmin`) doesn't silently drop the assertion.
    let seed = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.contains("issue68-bomb"))
                .unwrap_or(false)
        })
        .expect("issue #68 bomb regression seed must be present in fuzz/regression/");

    let input = fs::read(&seed).expect("read #68 bomb seed");

    let config = fuzz_config();
    let start = std::time::Instant::now();
    let result = zenwebp::DecodeRequest::new(&config, &input).decode_rgba();
    let elapsed = start.elapsed();

    // Must be rejected (oversized canvas) — not decoded into a 210 MB buffer.
    assert!(
        result.is_err(),
        "#68 bomb seed must be rejected under fuzzing limits, but it decoded"
    );

    // Must be fast. The rejection is a header check (~tens of µs on bare metal);
    // 2s is an enormous margin that still catches a regression to full decode
    // (which would be ~0.5s bare metal and far longer under instrumentation) or
    // any reintroduced unbounded loop.
    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "#68 bomb seed rejection took {elapsed:?} — expected microseconds; \
         a bound regressed (full decode or unbounded work)"
    );
}
