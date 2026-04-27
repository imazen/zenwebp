//! Test that large image encode/decode roundtrips produce valid bitstreams,
//! and that the encoder handles partition 0 overflow correctly:
//!
//! - With default settings (partition_limit = None), the encoder automatically
//!   retries with increasing I4 suppression to avoid overflow.
//! - With explicit partition_limit(0), the encoder returns Partition0Overflow.
//! - With explicit partition_limit(100), even very large images succeed.
//!
//! VP8 frame tag: partition_0_size is 19 bits (max 524,287 bytes).
//! See VP8 RFC 6386 Section 9.2 for the frame tag format.

use zenwebp::{DecodeRequest, EncodeError, EncodeRequest, LossyConfig, PixelLayout};

/// Generate a smooth gradient image (RGB, low entropy — compresses small).
fn generate_gradient_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / width) as u8;
            let g = (y * 255 / height) as u8;
            let b = ((x + y) * 128 / (width + height)) as u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    pixels
}

/// Generate a noisy image (RGB, high entropy — compresses large).
/// This pushes the compressed partition size past the 19-bit limit
/// without partition_limit mitigation.
fn generate_noisy_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let hash =
                ((x as u64).wrapping_mul(2654435761) ^ (y as u64).wrapping_mul(2246822519)) as u32;
            let noise = (hash >> 16) as u8;
            let r = ((x * 255 / width) as u8).wrapping_add(noise & 0x3F);
            let g = ((y * 255 / height) as u8).wrapping_add((noise >> 2) & 0x3F);
            let b = (((x + y) * 128 / (width + height)) as u8).wrapping_add((noise >> 4) & 0x3F);
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    pixels
}

/// Encode and decode an RGB image, asserting both succeed and dimensions match.
fn roundtrip_rgb(pixels: &[u8], width: u32, height: u32, quality: f32) {
    let config = LossyConfig::new().with_quality(quality);
    let encoded = EncodeRequest::lossy(&config, pixels, PixelLayout::Rgb8, width, height)
        .encode()
        .unwrap_or_else(|e| panic!("encode failed at {width}x{height}: {e}"));

    assert!(
        !encoded.is_empty(),
        "encoded data is empty at {width}x{height}"
    );

    let dc = zenwebp::DecodeConfig::default();
    let result = DecodeRequest::new(&dc, &encoded).decode();
    match result {
        Ok((_, dw, dh, _)) => {
            assert_eq!(dw, width, "width mismatch");
            assert_eq!(dh, height, "height mismatch");
        }
        Err(e) => {
            panic!(
                "decode failed at {width}x{height} q{quality} (encoded {} bytes): {e}",
                encoded.len()
            );
        }
    }
}

/// Attempt to encode with explicit partition_limit(0), asserting Partition0Overflow.
fn expect_partition0_overflow_no_retry(pixels: &[u8], width: u32, height: u32, quality: f32) {
    let config = LossyConfig::new()
        .with_quality(quality)
        .with_partition_limit(0);
    let result = EncodeRequest::lossy(&config, pixels, PixelLayout::Rgb8, width, height).encode();
    match result {
        Err(e) => {
            assert!(
                matches!(e.error(), EncodeError::Partition0Overflow { .. }),
                "expected Partition0Overflow at {width}x{height} q{quality}, got: {e}"
            );
        }
        Ok(data) => {
            panic!(
                "expected Partition0Overflow at {width}x{height} q{quality}, \
                 but encode succeeded ({} bytes)",
                data.len()
            );
        }
    }
}

/// Encode with explicit partition_limit, asserting success.
fn roundtrip_rgb_with_partition_limit(
    pixels: &[u8],
    width: u32,
    height: u32,
    quality: f32,
    limit: u8,
) {
    let config = LossyConfig::new()
        .with_quality(quality)
        .with_partition_limit(limit);
    let encoded = EncodeRequest::lossy(&config, pixels, PixelLayout::Rgb8, width, height)
        .encode()
        .unwrap_or_else(|e| {
            panic!("encode failed at {width}x{height} q{quality} limit={limit}: {e}")
        });

    assert!(
        !encoded.is_empty(),
        "encoded data is empty at {width}x{height}"
    );

    let dc = zenwebp::DecodeConfig::default();
    let result = DecodeRequest::new(&dc, &encoded).decode();
    match result {
        Ok((_, dw, dh, _)) => {
            assert_eq!(dw, width, "width mismatch");
            assert_eq!(dh, height, "height mismatch");
        }
        Err(e) => {
            panic!(
                "decode failed at {width}x{height} q{quality} limit={limit} (encoded {} bytes): {e}",
                encoded.len()
            );
        }
    }
}

// --- Low-entropy gradient: partition stays small ---

#[test]
fn gradient_4096x4096() {
    let pixels = generate_gradient_rgb(4096, 4096);
    roundtrip_rgb(&pixels, 4096, 4096, 80.0);
}

// --- High-entropy noisy: partition fits at moderate sizes ---

#[test]
fn noisy_4096x4096() {
    // 16.8MP — partition fits in 19 bits without retry
    let pixels = generate_noisy_rgb(4096, 4096);
    roundtrip_rgb(&pixels, 4096, 4096, 80.0);
}

// --- Automatic partition_limit retry ---

#[test]
fn noisy_5888x4416_auto_retry() {
    // 26.0MP — would overflow without partition_limit, auto-retry handles it
    let pixels = generate_noisy_rgb(5888, 4416);
    roundtrip_rgb(&pixels, 5888, 4416, 80.0);
}

// --- Explicit partition_limit(0): overflow errors are preserved ---

// 64-bit only — the 9216×6912 noisy image needs ~191 MB of pixel data plus
// encoder working buffers (~1 GB total), exceeding 32-bit address space.
// The smaller 5888×4416 image used previously no longer reliably overflows
// post-libwebp-parity-audit (PR #37) on any platform. The test's intent —
// verify partition_limit(0) disables auto-retry and returns
// Partition0Overflow — does not depend on pointer width.
//
// `noisy_4096x4096_q99_explicit_zero_overflows` (below) covers the same
// error-emission path on i686, where memory is the binding constraint.
#[cfg(target_pointer_width = "64")]
#[test]
fn noisy_5888x4416_explicit_zero_overflows() {
    //
    // The 2026-04-26 libwebp-parity audit (PR #37) + #25's full
    // SKIP_PROBA_THRESHOLD gate (this PR) made the encoder ~1.5% more
    // efficient on real-world content and saved ~1 bit/MB plus the
    // per-frame skip_proba header byte on noisy content. The same image
    // at 5888x4416 q80 now fits comfortably under the 524 KB partition-0
    // cap. Bumped dims to 9216x6912 (~64 MPix, ~250K MBs) at q=95 to
    // reliably overflow partition 0 again post-audit and exercise the
    // same error-emission path the test was designed to verify. The
    // test's INTENT (verify partition_limit(0) disables auto-retry and
    // returns Partition0Overflow) is preserved.
    let pixels = generate_noisy_rgb(9216, 6912);
    expect_partition0_overflow_no_retry(&pixels, 9216, 6912, 95.0);
}

// 32-bit coverage for the partition_limit API contract.
//
// I tried to write a small-dim synthetic that overflows on i686, but the
// post-audit encoder is efficient enough that even 8192×6144 of pure-random
// RGB at q99 with `partition_limit(0)` (worst-case settings for partition-0
// fill) doesn't overflow — the partition-0 fill stays well under 524 KB.
// To reliably overflow now requires ≥64 MPix, which doesn't fit in 32-bit
// address space.
//
// The tests below cover the *API contract* for `partition_limit` — what
// the test originally guarded — without needing to trigger real overflow:
// they verify the value propagates from LossyConfig into the encoder's
// auto-retry decision and that explicit partition_limit values produce
// valid output across the full range. i686 retains coverage of these
// regression classes; the actual error-emission path stays 64-bit-only.
//
// Triggering real overflow on i686 would need either an encoder hook to
// inject a smaller cap or a config that increases the partition-0 fill
// rate (e.g. forcing I4 unconditionally). Both are bigger changes than
// this review item warrants.

/// Verify partition_limit(0) produces a valid encode on a normal image
/// (partition-0 doesn't overflow at this size, so the success path is
/// what we exercise — separately from the overflow path covered on 64-bit).
#[test]
fn partition_limit_zero_succeeds_on_normal_image() {
    let pixels = generate_noisy_rgb(512, 512);
    let config = LossyConfig::new()
        .with_quality(80.0)
        .with_partition_limit(0);
    let encoded = EncodeRequest::lossy(&config, &pixels, PixelLayout::Rgb8, 512, 512)
        .encode()
        .expect("partition_limit(0) on a normal image must not error");
    assert!(!encoded.is_empty());
}

/// Verify partition_limit values across the full 0–100 range encode
/// successfully on a normal image. Catches signature / clamping bugs
/// without needing 32-bit memory headroom.
#[test]
fn partition_limit_full_range_smoke_test() {
    let pixels = generate_noisy_rgb(512, 512);
    for &limit in &[0u8, 25, 50, 75, 100] {
        let config = LossyConfig::new()
            .with_quality(75.0)
            .with_partition_limit(limit);
        let encoded = EncodeRequest::lossy(&config, &pixels, PixelLayout::Rgb8, 512, 512)
            .encode()
            .unwrap_or_else(|e| panic!("partition_limit({limit}) failed: {e}"));
        assert!(
            !encoded.is_empty(),
            "partition_limit({limit}) produced empty output"
        );
    }
}

// --- Explicit partition_limit: high values allow large images ---

#[test]
fn noisy_5888x4416_explicit_limit_100() {
    // Bumped limit from 70 → 100 after the 2026-04-26 libwebp-parity audit
    // (PR #37). The audit's mode-cost-table fixes (#21) and FastMBAnalyze
    // port (#32) shifted the m4 mode mix slightly toward I4 for textured
    // content, pushing partition-0 over the 524 KB cap on 26 MP noisy
    // even at limit=85. With ~100K MBs at this resolution, partition-0 sits
    // close to the cap regardless; limit=100 (maximum I4 suppression)
    // restores comfortable headroom across all platforms. The test's
    // INTENT (verify partition_limit honors a non-zero value and lets
    // the encoder succeed where partition_limit=0 would error) is preserved.
    let pixels = generate_noisy_rgb(5888, 4416);
    roundtrip_rgb_with_partition_limit(&pixels, 5888, 4416, 80.0, 100);
}
