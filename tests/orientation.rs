//! zencodec adapter orientation-hint coverage.
//!
//! Mirrors heic's `tests/cov_zencodec.rs::orientation_*`. WebP carries
//! orientation only as the EXIF `Orientation` tag (TIFF tag 274); the bitstream
//! dimensions are always the stored (coded) dimensions. The adapter honors
//! [`zencodec::OrientationHint`]:
//!   - `Preserve` (default): pixels stay in stored orientation; `ImageInfo`
//!     reports the stored (coded) dims + the intrinsic EXIF `Orientation` tag,
//!     and `display_width()/display_height()` yield the upright dims.
//!   - `Correct`: the decoder bakes the image upright; `ImageInfo` reports the
//!     display dims + `Orientation::Identity`, and the pixels are physically
//!     rotated.

#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use std::borrow::Cow;

use zencodec::decode::{Decode, DecodeJob, DecoderConfig};
use zencodec::{Orientation, OrientationHint};
use zenpixels::PixelDescriptor;
use zenwebp::zencodec::WebpDecoderConfig;
use zenwebp::{EncodeRequest, LosslessConfig, PixelLayout};

/// Stored (coded) dimensions of the synthesized fixture.
const STORED: (u32, u32) = (4, 2);
/// Display dimensions after applying the EXIF Rotate90 (axes swap).
const DISPLAY: (u32, u32) = (2, 4);

/// Build a minimal raw-TIFF EXIF blob carrying a single `Orientation` tag.
///
/// WebP EXIF is raw TIFF bytes (no `Exif\0\0` prefix). Little-endian IFD0 with
/// exactly the orientation entry — the same shape the in-crate
/// `exif_orientation` parser reads.
fn exif_with_orientation(orientation: u16) -> Vec<u8> {
    const TAG_ORIENTATION: u16 = 0x0112;
    const TIFF_TYPE_SHORT: u16 = 3;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II"); // little-endian
    buf.extend_from_slice(&42u16.to_le_bytes()); // TIFF magic
    buf.extend_from_slice(&8u32.to_le_bytes()); // IFD0 offset
    buf.extend_from_slice(&1u16.to_le_bytes()); // entry count
    buf.extend_from_slice(&TAG_ORIENTATION.to_le_bytes());
    buf.extend_from_slice(&TIFF_TYPE_SHORT.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes()); // count
    buf.extend_from_slice(&orientation.to_le_bytes()); // value (inline)
    buf.extend_from_slice(&0u16.to_le_bytes()); // padding
    buf
}

/// A unique color per `(x, y)` so a rotation can be verified pixel-exactly.
fn color_at(x: u32, y: u32) -> [u8; 3] {
    [10 + x as u8 * 20, 100 + y as u8 * 40, 200]
}

/// Encode a `STORED`-sized RGB image (asymmetric content) with the given EXIF
/// orientation tag, losslessly + exact so the decode round-trips bit-for-bit.
fn fixture_with_orientation(orientation: u16) -> Vec<u8> {
    let (w, h) = STORED;
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.extend_from_slice(&color_at(x, y));
        }
    }
    let exif = exif_with_orientation(orientation);
    let cfg = LosslessConfig::new().with_exact(true);
    EncodeRequest::lossless(&cfg, &pixels, PixelLayout::Rgb8, w, h)
        .with_exif(&exif)
        .encode()
        .expect("encode fixture")
}

/// Read an RGB pixel from a decoded RGB8 output.
fn px(out: &zencodec::decode::DecodeOutput, x: u32, y: u32) -> [u8; 3] {
    let ps = out.pixels();
    assert_eq!(ps.descriptor(), PixelDescriptor::RGB8_SRGB);
    let row = ps.row(y);
    let off = x as usize * 3;
    [row[off], row[off + 1], row[off + 2]]
}

// ── Preserve (default) ──────────────────────────────────────────────────────

#[test]
fn orientation_preserve_default_reports_stored_dims_and_tag() {
    // EXIF Rotate90 (tag 6). Default config == OrientationHint::Preserve.
    let data = fixture_with_orientation(6);
    let info = WebpDecoderConfig::new().job().probe(&data).expect("probe");
    assert_eq!(
        (info.width, info.height),
        STORED,
        "Preserve must report stored (coded, pre-rotation) dims"
    );
    assert_eq!(
        info.orientation,
        Orientation::Rotate90,
        "Preserve must report the intrinsic EXIF orientation tag"
    );
    assert_eq!(
        (info.display_width(), info.display_height()),
        DISPLAY,
        "display_width/height must yield the upright dims under Preserve"
    );
}

#[test]
fn orientation_preserve_decode_keeps_stored_pixels() {
    let data = fixture_with_orientation(6);
    let out = WebpDecoderConfig::new()
        .job()
        .decoder(Cow::Borrowed(&data), &[PixelDescriptor::RGB8_SRGB])
        .expect("decoder")
        .decode()
        .expect("decode");
    assert_eq!(
        (out.width(), out.height()),
        STORED,
        "Preserve decode must output stored-orientation pixels"
    );
    assert_eq!(
        (out.info().width, out.info().height),
        STORED,
        "Preserve decode ImageInfo dims must match the decoded pixels"
    );
    assert_eq!(
        out.info().orientation,
        Orientation::Rotate90,
        "Preserve decode must tag the intrinsic orientation"
    );
    // Pixels are untouched: every source position holds its original color.
    for y in 0..STORED.1 {
        for x in 0..STORED.0 {
            assert_eq!(px(&out, x, y), color_at(x, y), "preserved pixel ({x},{y})");
        }
    }
}

// ── Correct ─────────────────────────────────────────────────────────────────

#[test]
fn orientation_correct_reports_display_dims_and_identity() {
    let data = fixture_with_orientation(6);
    let info = WebpDecoderConfig::new()
        .job()
        .with_orientation(OrientationHint::Correct)
        .probe(&data)
        .expect("probe");
    assert_eq!(
        (info.width, info.height),
        DISPLAY,
        "Correct must report display (post-rotation) dims"
    );
    assert_eq!(
        info.orientation,
        Orientation::Identity,
        "Correct must report Identity — orientation is baked into the pixels"
    );
    assert_eq!((info.display_width(), info.display_height()), DISPLAY);
}

#[test]
fn orientation_correct_decode_bakes_upright_pixels() {
    let data = fixture_with_orientation(6);
    let out = WebpDecoderConfig::new()
        .job()
        .with_orientation(OrientationHint::Correct)
        .decoder(Cow::Borrowed(&data), &[PixelDescriptor::RGB8_SRGB])
        .expect("decoder")
        .decode()
        .expect("decode");
    assert_eq!(
        (out.width(), out.height()),
        DISPLAY,
        "Correct decode must output display-orientation (upright) pixels"
    );
    assert_eq!(
        out.info().orientation,
        Orientation::Identity,
        "Correct decode must report Identity"
    );
    assert_eq!((out.info().width, out.info().height), DISPLAY);

    // The pixels must be physically rotated. Orientation::Rotate90 forward-maps
    // source (sx, sy) with source dims (w, h) to (h - 1 - sy, sx). Verify every
    // source pixel landed at its rotated destination, proving the bake ran.
    let (sw, sh) = STORED;
    for sy in 0..sh {
        for sx in 0..sw {
            let (dx, dy) = Orientation::Rotate90.forward_map(sx, sy, sw, sh);
            assert_eq!(
                px(&out, dx, dy),
                color_at(sx, sy),
                "rotated pixel src({sx},{sy}) -> dst({dx},{dy})"
            );
        }
    }
}

// ── Correct on an already-upright image (no EXIF) ───────────────────────────

#[test]
fn orientation_correct_on_upright_image_is_noop_but_reports_identity() {
    // No EXIF orientation → intrinsic Identity. Correct resolves to a no-op bake
    // but must still report Identity + the unchanged dims.
    let (w, h) = STORED;
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.extend_from_slice(&color_at(x, y));
        }
    }
    let cfg = LosslessConfig::new().with_exact(true);
    let data = EncodeRequest::lossless(&cfg, &pixels, PixelLayout::Rgb8, w, h)
        .encode()
        .expect("encode");

    let out = WebpDecoderConfig::new()
        .job()
        .with_orientation(OrientationHint::Correct)
        .decoder(Cow::Borrowed(&data), &[PixelDescriptor::RGB8_SRGB])
        .expect("decoder")
        .decode()
        .expect("decode");
    assert_eq!((out.width(), out.height()), STORED);
    assert_eq!(out.info().orientation, Orientation::Identity);
    for y in 0..h {
        for x in 0..w {
            assert_eq!(px(&out, x, y), color_at(x, y));
        }
    }
}

// ── ExactTransform: ignore EXIF, apply literally ────────────────────────────

#[test]
fn orientation_exact_transform_ignores_exif() {
    // EXIF says Rotate90, but ExactTransform(FlipH) ignores it and flips
    // horizontally (no axis swap → dims unchanged).
    let data = fixture_with_orientation(6);
    let out = WebpDecoderConfig::new()
        .job()
        .with_orientation(OrientationHint::ExactTransform(Orientation::FlipH))
        .decoder(Cow::Borrowed(&data), &[PixelDescriptor::RGB8_SRGB])
        .expect("decoder")
        .decode()
        .expect("decode");
    assert_eq!(
        (out.width(), out.height()),
        STORED,
        "FlipH does not swap axes → stored dims"
    );
    assert_eq!(out.info().orientation, Orientation::Identity);
    let (sw, sh) = STORED;
    for sy in 0..sh {
        for sx in 0..sw {
            let (dx, dy) = Orientation::FlipH.forward_map(sx, sy, sw, sh);
            assert_eq!(px(&out, dx, dy), color_at(sx, sy), "flipped ({sx},{sy})");
        }
    }
}
