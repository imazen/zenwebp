//! `StreamingDecoder` behavioral tests.
//!
//! This public API had ZERO executed tests until 2026-08-21 (0/91 lines in
//! the coverage run — only a `no_run` doc example). Found alongside a live
//! 32-bit bug: `riff_size as usize + 8` wrapped for a near-u32::MAX declared
//! RIFF size, reporting `Complete` after a handful of bytes and handing
//! `finish_*` truncated garbage. The `huge_declared_riff_size` test is the
//! gate for that fix; it exercises the wrap on the i686/wasm32 CI targets.

use zenwebp::decoder::{StreamStatus, StreamingDecoder};
use zenwebp::{DecodeConfig, DecodeRequest, EncodeRequest, EncoderConfig, PixelLayout};

/// Odd-dimensioned RGBA test image (exercises non-MB-aligned paths).
fn test_webp() -> (Vec<u8>, Vec<u8>, u32, u32) {
    let (w, h) = (33u32, 17u32);
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            rgba.extend_from_slice(&[(x * 7) as u8, (y * 13) as u8, ((x ^ y) & 0xff) as u8, 255]);
        }
    }
    let config = EncoderConfig::new_lossy().with_quality(75.0);
    let webp = EncodeRequest::new(&config, &rgba, PixelLayout::Rgba8, w, h)
        .encode()
        .unwrap();
    (webp, rgba, w, h)
}

#[test]
fn chunked_delivery_matches_one_shot() {
    let (webp, _rgba, w, h) = test_webp();

    // One-shot reference decode.
    let config = DecodeConfig::default();
    let (reference, rw, rh) = DecodeRequest::new(&config, &webp).decode_rgba().unwrap();
    assert_eq!((rw, rh), (w, h));

    // Feed in awkward 7-byte chunks; statuses must progress monotonically
    // NeedMoreData -> HeaderReady -> Complete.
    let mut dec = StreamingDecoder::new();
    let mut seen_header = false;
    let mut seen_complete = false;
    for chunk in webp.chunks(7) {
        let status = dec.append(chunk).unwrap();
        match status {
            StreamStatus::NeedMoreData => {
                assert!(!seen_header, "status regressed from HeaderReady");
                assert!(!seen_complete, "status regressed from Complete");
            }
            StreamStatus::HeaderReady => {
                assert!(!seen_complete, "status regressed from Complete");
                seen_header = true;
                let info = dec.info().unwrap();
                assert_eq!((info.width, info.height), (w, h));
            }
            StreamStatus::Complete => seen_complete = true,
            _ => {}
        }
    }
    assert!(seen_complete, "full file must reach Complete");
    assert!(dec.is_complete());
    assert_eq!(dec.bytes_buffered(), webp.len());
    assert_eq!(dec.total_size(), Some(webp.len()));

    let (pixels, fw, fh) = dec.finish_rgba().unwrap();
    assert_eq!((fw, fh), (w, h));
    assert_eq!(
        pixels, reference,
        "chunked decode must match one-shot decode exactly"
    );
}

#[test]
fn single_append_completes_and_decodes() {
    let (webp, _rgba, w, h) = test_webp();
    let mut dec = StreamingDecoder::new();
    assert_eq!(dec.append(&webp).unwrap(), StreamStatus::Complete);

    let config = DecodeConfig::default();
    let (reference, ..) = DecodeRequest::new(&config, &webp).decode_rgb().unwrap();
    let (pixels, fw, fh) = dec.finish_rgb().unwrap();
    assert_eq!((fw, fh), (w, h));
    assert_eq!(pixels, reference);
}

#[test]
fn finish_rgba_into_matches() {
    let (webp, _rgba, w, h) = test_webp();
    let mut dec = StreamingDecoder::new();
    dec.append(&webp).unwrap();

    let config = DecodeConfig::default();
    let (reference, ..) = DecodeRequest::new(&config, &webp).decode_rgba().unwrap();

    let mut out = vec![0u8; (w * h * 4) as usize];
    let (fw, fh) = dec.finish_rgba_into(&mut out).unwrap();
    assert_eq!((fw, fh), (w, h));
    assert_eq!(out, reference);
}

#[test]
fn bad_signatures_error() {
    let mut dec = StreamingDecoder::new();
    let err = dec.append(b"XIFF\x00\x00\x00\x00WEBPVP8 ").unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("Riff"), "expected RIFF signature error: {msg}");

    let mut dec = StreamingDecoder::new();
    let err = dec.append(b"RIFF\x00\x00\x00\x00XEBPVP8 ").unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("Webp"), "expected WEBP signature error: {msg}");
}

#[test]
fn premature_finish_errors() {
    let (webp, ..) = test_webp();
    let mut dec = StreamingDecoder::new();
    // Header + a bit of payload, but not the whole file.
    dec.append(&webp[..40]).unwrap();
    assert!(!dec.is_complete());
    assert!(
        dec.finish_rgba().is_err(),
        "incomplete data must not decode"
    );

    // info() before headers are parseable must error, not panic.
    let dec = StreamingDecoder::new();
    assert!(dec.info().is_err());
}

/// 32-bit wrap gate: a file declaring a near-u32::MAX RIFF size must NOT
/// report Complete after a handful of bytes. Before the fix,
/// `riff_size as usize + 8` wrapped to 4 on a 32-bit target, so
/// `buf.len() >= total` was true immediately. Trivially passes on 64-bit;
/// gates for real on the i686 and wasm32 CI targets.
#[test]
fn huge_declared_riff_size_does_not_complete_early() {
    let mut header = Vec::new();
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&0xFFFF_FFF4u32.to_le_bytes());
    header.extend_from_slice(b"WEBP");
    header.extend_from_slice(&[0u8; 100]);

    let mut dec = StreamingDecoder::new();
    let status = dec.append(&header).unwrap();
    assert_ne!(
        status,
        StreamStatus::Complete,
        "112 bytes of a 4 GB-declared file must not be Complete"
    );
    assert!(!dec.is_complete());
    assert!(dec.finish_rgba().is_err());
}

#[test]
fn into_inner_returns_exact_buffer() {
    let (webp, ..) = test_webp();
    let mut dec = StreamingDecoder::new();
    for chunk in webp.chunks(11) {
        dec.append(chunk).unwrap();
    }
    assert_eq!(dec.into_inner(), webp);
}
