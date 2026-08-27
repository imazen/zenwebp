//! The VP8L (lossless) decoder's cold paths — header, transform list,
//! Huffman-code reading, color-cache header — and the transform entry points
//! must tag their error ORIGINS with `at!()` (#60). Before this, every error
//! from `LosslessDecoder` surfaced as a bare `InternalDecodeError`, was
//! converted to `DecodeError` at `decode_frame`, and only picked up a
//! location at whatever `?` the caller happened to use (or none: the
//! `From<E> for At<E>` bridge records no frame), so a corrupt lossless file
//! reported `BitStreamError` with no idea which check failed.
//!
//! Each case here is a minimal VP8L stream whose first failing check lives in
//! `src/decoder/lossless.rs`, and asserts the FIRST trace frame is in that
//! file. Watched to FAIL (first frame in `src/decoder/api.rs`, at the
//! caller's `?`) before the origins were tagged.

use zenwebp::{DecodeConfig, DecodeRequest};

/// A 1x1 VP8L still whose payload after the 5-byte header is `tail`.
fn vp8l_stream(tail: &[u8]) -> Vec<u8> {
    let mut payload = vec![0x2f, 0, 0, 0, 0]; // signature; w-1 = h-1 = 0; no alpha; v0
    payload.extend_from_slice(tail);
    let mut d = Vec::new();
    d.extend_from_slice(b"RIFF");
    let chunk = 8 + payload.len() + (payload.len() & 1);
    d.extend_from_slice(&((4 + chunk) as u32).to_le_bytes());
    d.extend_from_slice(b"WEBP");
    d.extend_from_slice(b"VP8L");
    d.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    d.extend_from_slice(&payload);
    if payload.len() & 1 == 1 {
        d.push(0);
    }
    d
}

fn origin_file(data: &[u8]) -> (String, String, usize) {
    let err = DecodeRequest::new(&DecodeConfig::default(), data)
        .decode_rgba()
        .expect_err("crafted stream must fail to decode");
    let first = err
        .frames()
        .next()
        .and_then(|f| f.location())
        .map(|l| l.file().replace('\\', "/"))
        .unwrap_or_default();
    (format!("{:?}", err.error()), first, err.frame_count())
}

#[test]
fn duplicate_transform_error_originates_in_lossless_rs() {
    // Transform present (1) + type 2 (subtract green), then present (1) +
    // type 2 again: the "one of each transform" check in `read_transforms`.
    let data = vp8l_stream(&[0b0010_1101, 0, 0, 0]);
    let (kind, file, frames) = origin_file(&data);
    assert_eq!(kind, "TransformError");
    assert!(frames >= 1, "no trace frames at all");
    assert!(
        file.ends_with("src/decoder/lossless.rs"),
        "first frame must be the read_transforms origin, got {file:?} ({frames} frames)"
    );
}

#[test]
fn invalid_color_cache_bits_error_originates_in_lossless_rs() {
    // No transforms (0), color cache present (1), code_bits = 0 (invalid,
    // must be 1..=11): the check in `read_color_cache`.
    let data = vp8l_stream(&[0b0000_0010, 0, 0, 0]);
    let (kind, file, frames) = origin_file(&data);
    assert_eq!(
        kind, "BitStreamError",
        "InvalidColorCacheBits maps to BitStreamError"
    );
    assert!(frames >= 1, "no trace frames at all");
    assert!(
        file.ends_with("src/decoder/lossless.rs"),
        "first frame must be the read_color_cache origin, got {file:?} ({frames} frames)"
    );
}

#[test]
fn bad_simple_huffman_symbol_error_originates_in_lossless_rs() {
    // No transforms (0), no color cache (0), no meta-Huffman (0), then the
    // first (green) tree: simple (1), num_symbols-1 = 0, is_first_8bits = 1,
    // symbol = 0xFF... the 8-bit symbol 255 is below the 280-entry green
    // alphabet, so instead target the RED tree: make the green tree a valid
    // single-symbol code, then give red (alphabet 256) an 8-bit symbol that
    // is fine too — the reliable out-of-range case is `zero_symbol >=
    // alphabet_size` for the DISTANCE tree (alphabet 40) with symbol 255.
    // Bits (LSB first): [0][0][0] | green: 1,0,0,(1 bit)0 | red: 1,0,0,0 |
    // blue: 1,0,0,0 | alpha: 1,0,0,0 | dist: 1,0,1,(8 bits)0xFF
    let mut bits: Vec<u8> = Vec::new();
    let mut push = |v: u32, n: u8| {
        for i in 0..n {
            bits.push(((v >> i) & 1) as u8);
        }
    };
    push(0, 3); // no transform, no color cache, no meta
    for _ in 0..4 {
        push(1, 1); // simple
        push(0, 1); // one symbol
        push(0, 1); // 1-bit symbol
        push(0, 1); // symbol 0
    }
    push(1, 1); // dist: simple
    push(0, 1); // one symbol
    push(1, 1); // 8-bit symbol
    push(0xFF, 8); // 255 >= 40
    let mut tail = vec![0u8; bits.len().div_ceil(8) + 4];
    for (i, b) in bits.iter().enumerate() {
        tail[i / 8] |= b << (i % 8);
    }
    let data = vp8l_stream(&tail);
    let (kind, file, frames) = origin_file(&data);
    assert_eq!(kind, "BitStreamError");
    assert!(frames >= 1, "no trace frames at all");
    assert!(
        file.ends_with("src/decoder/lossless.rs"),
        "first frame must be the read_huffman_code origin, got {file:?} ({frames} frames)"
    );
}
