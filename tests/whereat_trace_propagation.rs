//! Regression test for whereat trace preservation through the converted
//! internal decode path.
//!
//! Background: zenwebp previously stripped `whereat::At<_>` traces at several
//! type boundaries via `From<At<X>> for Y` impls and explicit `.decompose().0`
//! calls. Those droppers were removed and the internal propagation functions
//! (`WebpDecoder::do_decode` / `do_decode_lossy` / `do_decode_general`, etc.)
//! now return `Result<_, At<DecodeError>>` and propagate dependency errors with
//! a plain `?`.
//!
//! This test asserts that a decode error *originating inside a dependency*
//! (the container/VP8L decoder, reached through the converted `do_decode`
//! chain) still carries its `file:line` trace when it surfaces — i.e.
//! `err.frame_count() >= 1`. If a future change reintroduces a trace-dropping
//! boundary on this path, the frame count collapses to 0 and this test fails.

#![cfg(all(feature = "std", feature = "zencodec", not(target_arch = "wasm32")))]

use zenwebp::zencodec::WebpDecoderConfig;

/// A WebP whose RIFF/WEBP/VP8L container header is well-formed enough to clear
/// `WebpDecoder`'s early local checks, but whose VP8L payload declares an
/// invalid bitstream version. The error therefore originates deep in the
/// container/VP8L decode dependency (`VersionNumberInvalid`), not in one of
/// `do_decode`'s own `at!(...)` early returns — exactly the dependency-trace
/// propagation this test guards.
fn vp8l_with_invalid_version() -> Vec<u8> {
    let mut d: Vec<u8> = Vec::new();
    d.extend_from_slice(b"RIFF");
    // File size = "WEBP" (4) + chunk header (8) + claimed payload.
    d.extend_from_slice(&(4u32 + 8 + 16).to_le_bytes());
    d.extend_from_slice(b"WEBP");
    d.extend_from_slice(b"VP8L");
    d.extend_from_slice(&16u32.to_le_bytes()); // declared VP8L chunk size
    // VP8L payload: 0x2f signature, then bits encoding a non-zero (invalid)
    // version number — rejected by the VP8L bitstream reader.
    d.extend_from_slice(&[0x2f, 0x00, 0x00, 0x88, 0x88, 0x00, 0x00, 0x00]);
    d
}

#[test]
fn dependency_decode_error_keeps_whereat_trace() {
    let data = vp8l_with_invalid_version();

    // `WebpDecoderConfig::decode` runs the converted `do_decode` →
    // `do_decode_general` → dependency `decode()?` path.
    let err = WebpDecoderConfig::new()
        .decode(&data)
        .expect_err("invalid VP8L version must fail to decode");

    // The trace must survive propagation through the converted functions.
    // A dropped trace would leave `frame_count() == 0`.
    assert!(
        err.frame_count() >= 1,
        "expected the dependency decode error to carry a whereat trace \
         (frame_count >= 1), got frame_count = {} for error {:?}",
        err.frame_count(),
        err.error(),
    );
}
