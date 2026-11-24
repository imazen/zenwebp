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
use zenpixels::ImageSpec;
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
    assert_eq!(ps.spec(), &ImageSpec::rgb8_srgb());
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
        .decoder(Cow::Borrowed(&data), &[ImageSpec::rgb8_srgb()])
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
        .decoder(Cow::Borrowed(&data), &[ImageSpec::rgb8_srgb()])
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
        .decoder(Cow::Borrowed(&data), &[ImageSpec::rgb8_srgb()])
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
        .decoder(Cow::Borrowed(&data), &[ImageSpec::rgb8_srgb()])
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

#[test]
fn baked_orientation_retains_icc_origin_and_source_encoding() {
    let profile = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    let data = fixture_with_orientation(6);
    let data = zenwebp::metadata::embed_icc(&data, profile).expect("attach fixture ICC");
    let out = WebpDecoderConfig::new()
        .job()
        .with_orientation(OrientationHint::CorrectAndTransform(Orientation::Rotate180))
        .decoder(Cow::Borrowed(&data), &[ImageSpec::rgb8_srgb()])
        .unwrap()
        .decode()
        .unwrap();

    assert_eq!((out.width(), out.height()), DISPLAY);
    assert_eq!(out.info().orientation, Orientation::Identity);
    assert_eq!(out.buffer().origin().icc.as_deref(), Some(profile));
    assert!(out.source_encoding_details().is_some());
    let view = out.pixels();
    let zenpixels::ColorSpace::Icc {
        profile: active, ..
    } = &view.spec().color
    else {
        panic!("missing active RGB ICC");
    };
    assert_eq!(active.as_ref(), profile);
    assert_eq!(view.spec().layout, zenpixels::SampleLayout::RGBA8);
    let (sw, sh) = STORED;
    for y in 0..sh {
        for x in 0..sw {
            let (dx, dy) = Orientation::Rotate270.forward_map(x, y, sw, sh);
            let start = dx as usize * 4;
            let [r, g, b] = color_at(x, y);
            assert_eq!(&view.row(dy)[start..start + 4], &[r, g, b, 255]);
        }
    }
}

#[test]
#[allow(deprecated)] // The legacy structured metadata setter is still supported.
fn animation_encode_serializes_structured_orientation() {
    use zencodec::encode::{AnimationFrameEncoder, EncodeJob, EncoderConfig};
    use zenpixels::PixelView;
    use zenwebp::zencodec::WebpEncoderConfig;

    let mut encoder = WebpEncoderConfig::lossless()
        .job()
        .with_metadata_policy(
            zencodec::Metadata::none().with_orientation(Orientation::Rotate90),
            zencodec::MetadataPolicy::PreserveExact,
        )
        .animation_frame_encoder()
        .unwrap();
    for bytes in [[20, 40, 60, 80, 100, 120], [121, 101, 81, 61, 41, 21]] {
        let view = PixelView::from_packed(&bytes, 2, 1, ImageSpec::rgb8_srgb()).unwrap();
        encoder.push_frame(view, 15, None).unwrap();
    }
    let output = encoder.finish(None).unwrap();
    let demux = zenwebp::mux::WebPDemuxer::new(output.data()).unwrap();
    let exif = demux
        .exif()
        .expect("structured orientation needs an EXIF chunk");
    assert_eq!(
        zencodec::helpers::parse_exif_orientation(exif),
        Some(Orientation::Rotate90)
    );
}

const FIRST: [[u8; 4]; 6] = [
    [11, 21, 31, 255],
    [42, 52, 62, 255],
    [73, 83, 93, 255],
    [104, 114, 124, 255],
    [135, 145, 155, 255],
    [166, 176, 186, 255],
];

fn rotate_clockwise(pixels: &[[u8; 4]; 6]) -> Vec<u8> {
    [3, 0, 4, 1, 5, 2]
        .into_iter()
        .flat_map(|i| pixels[i])
        .collect()
}

fn assert_frame_pixels(
    view: &zenpixels::PixelView<'_>,
    dims: (u32, u32),
    expected: &[u8],
    profile: &[u8],
) {
    assert_eq!((view.width(), view.rows()), dims);
    assert_eq!(view.spec().layout, zenpixels::SampleLayout::RGBA8);
    assert_eq!(view.origin().unwrap().icc.as_deref(), Some(profile));
    let zenpixels::ColorSpace::Icc {
        profile: active, ..
    } = &view.spec().color
    else {
        panic!("missing active ICC");
    };
    assert_eq!(active.as_ref(), profile);
    for y in 0..dims.1 {
        let stride = dims.0 as usize * 4;
        assert_eq!(
            view.row(y),
            &expected[y as usize * stride..(y as usize + 1) * stride]
        );
    }
}

#[test]
#[allow(deprecated)] // Exercise metadata retention on animation encode.
fn true_animation_applies_orientation_at_all_delivery_boundaries() {
    use zencodec::animation::FrameDuration;
    use zencodec::decode::AnimationFrameDecoder;
    use zencodec::encode::{AnimationFrameEncoder, EncodeJob, EncoderConfig};
    use zencodec::{Metadata, MetadataPolicy};
    use zenwebp::zencodec::WebpEncoderConfig;

    let profile = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    let second = FIRST.map(|[r, g, b, _]| [g, b, r, 181]);
    let sources = [FIRST, second];
    let mut encoder = WebpEncoderConfig::lossless()
        .job()
        .with_metadata_policy(
            Metadata::none()
                .with_exif(complete_exif(6))
                .with_icc(profile),
            MetadataPolicy::PreserveExact,
        )
        .animation_frame_encoder()
        .unwrap();
    for (pixels, duration) in sources.iter().zip([17, 29]) {
        let view =
            zenpixels::PixelView::from_packed(pixels.as_flattened(), 3, 2, ImageSpec::rgba8_srgb())
                .unwrap();
        encoder
            .push_frame_timed(view, FrameDuration::from_millis(duration), None)
            .unwrap();
    }
    let encoded = encoder.finish(None).unwrap();
    let data = encoded.data();
    let oracle = webpx::AnimationDecoder::new(data)
        .unwrap()
        .decode_all()
        .unwrap();
    assert_eq!(oracle.len(), 2);
    assert_eq!(oracle[0].data, FIRST.as_flattened());
    assert_eq!(oracle[1].data, second.as_flattened());

    for (hint, dims, pending, applied) in [
        (
            OrientationHint::Preserve,
            (3, 2),
            Orientation::Rotate90,
            Orientation::Identity,
        ),
        (
            OrientationHint::Correct,
            (2, 3),
            Orientation::Identity,
            Orientation::Rotate90,
        ),
    ] {
        let job = WebpDecoderConfig::new().job().with_orientation(hint);
        let info = job.probe(data).unwrap();
        assert_eq!(
            ((info.width, info.height), info.orientation),
            (dims, pending)
        );
        let predicted = job.output_info(data).unwrap();
        assert_eq!(
            (
                (predicted.width, predicted.height),
                predicted.orientation_applied
            ),
            (dims, applied)
        );
        let mut borrowed = job
            .clone()
            .animation_frame_decoder(Cow::Borrowed(data), &[])
            .unwrap();
        let mut owned = job
            .clone()
            .animation_frame_decoder(Cow::Borrowed(data), &[])
            .unwrap();
        let mut sink_decoder = job
            .animation_frame_decoder(Cow::Borrowed(data), &[])
            .unwrap();
        let info = borrowed.info();
        assert_eq!(
            ((info.width, info.height), info.orientation),
            (dims, pending)
        );

        for (index, (source, millis)) in sources.iter().zip([17, 29]).enumerate() {
            let duration = FrameDuration::from_millis(millis);
            let expected = if applied == Orientation::Identity {
                source.as_flattened().to_vec()
            } else {
                rotate_clockwise(source)
            };
            let frame = borrowed.render_next_frame(None).unwrap().unwrap();
            assert_frame_pixels(frame.pixels(), dims, &expected, profile);
            assert_eq!(
                (frame.frame_index(), frame.duration(), frame.pts()),
                (index as u32, duration, None)
            );
            let copy = frame.try_to_owned_frame().unwrap();
            assert_frame_pixels(&copy.pixels(), dims, &expected, profile);
            let frame = owned.render_next_frame_owned(None).unwrap().unwrap();
            assert_frame_pixels(&frame.pixels(), dims, &expected, profile);
            assert_eq!(
                (frame.frame_index(), frame.duration(), frame.pts()),
                (index as u32, duration, None)
            );
            let mut sink = zencodec_io::CollectSink::new();
            let completed = sink_decoder
                .render_next_frame_to_sink(None, &mut sink)
                .unwrap()
                .unwrap();
            let timing = completed.frame.unwrap();
            assert_eq!(
                (timing.frame_index, timing.duration, timing.pts),
                (index as u32, duration, None)
            );
            assert_eq!(
                (
                    (completed.width, completed.height),
                    completed.orientation_applied
                ),
                (dims, applied)
            );
            assert_frame_pixels(
                &sink.into_buffer().unwrap().as_view(),
                dims,
                &expected,
                profile,
            );
        }
        assert!(borrowed.render_next_frame(None).unwrap().is_none());
        assert!(owned.render_next_frame_owned(None).unwrap().is_none());
        assert!(
            sink_decoder
                .render_next_frame_to_sink(None, &mut zencodec_io::CollectSink::new())
                .unwrap()
                .is_none()
        );
    }
}

fn complete_exif(orientation: u16) -> Vec<u8> {
    let mut exif = exif_with_orientation(orientation);
    exif.extend_from_slice(&0u32.to_le_bytes()); // end of IFD chain
    exif
}

// Read orientation directly from the little-endian IFD0 fixture extracted
// by libwebp. The codec's EXIF orientation parser is independent of this check.
fn fixture_exif_orientation(exif: &[u8]) -> Option<u16> {
    let exif = exif.strip_prefix(b"Exif\0\0").unwrap_or(exif);
    assert_eq!(&exif[..4], b"II*\0");
    let offset = u32::from_le_bytes(exif[4..8].try_into().unwrap()) as usize;
    let count = u16::from_le_bytes(exif[offset..offset + 2].try_into().unwrap()) as usize;
    exif[offset + 2..offset + 2 + 12 * count]
        .as_chunks::<12>()
        .0
        .iter()
        .find_map(|entry| {
            (entry[..2] == [0x12, 1]).then(|| {
                assert_eq!(&entry[2..8], &[3, 0, 1, 0, 0, 0]);
                u16::from_le_bytes(entry[8..10].try_into().unwrap())
            })
        })
}

#[test]
#[allow(deprecated)] // The documented PreserveExact metadata path.
fn correct_decode_reencode_reconciles_exif_and_keeps_icc_and_display_pixels() {
    use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
    use zenwebp::zencodec::WebpEncoderConfig;

    let profile = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    let data = EncodeRequest::lossless(
        &LosslessConfig::new().with_exact(true),
        FIRST.as_flattened(),
        PixelLayout::Rgba8,
        3,
        2,
    )
    .with_exif(&complete_exif(6))
    .with_icc_profile(profile)
    .encode()
    .unwrap();
    let decoder = || {
        WebpDecoderConfig::new()
            .job()
            .with_orientation(OrientationHint::Correct)
    };
    let decoded = decoder()
        .decoder(Cow::Borrowed(&data), &[])
        .unwrap()
        .decode()
        .unwrap();
    assert_eq!(decoded.info().orientation, Orientation::Identity);
    assert_eq!(
        fixture_exif_orientation(decoded.metadata().exif.as_deref().unwrap()),
        Some(6)
    );
    assert_eq!(
        fixture_exif_orientation(&webpx::get_exif(&data).unwrap().unwrap()),
        Some(6)
    );
    let expected = rotate_clockwise(&FIRST);
    assert_frame_pixels(&decoded.pixels(), (2, 3), &expected, profile);
    let output = WebpEncoderConfig::lossless()
        .job()
        .with_metadata_policy(decoded.metadata(), zencodec::MetadataPolicy::PreserveExact)
        .encoder()
        .unwrap()
        .encode(decoded.pixels())
        .unwrap();
    let reencoded = output.data();
    let emitted = webpx::get_exif(reencoded).unwrap();
    assert!(matches!(
        emitted.as_deref().and_then(fixture_exif_orientation),
        None | Some(1)
    ));
    assert_eq!(
        webpx::get_icc_profile(reencoded).unwrap().as_deref(),
        Some(profile)
    );
    assert_eq!(webpx::decode_rgba(reencoded).unwrap().0, expected);
    let decoded_again = decoder()
        .decoder(Cow::Borrowed(reencoded), &[])
        .unwrap()
        .decode()
        .unwrap();
    assert_frame_pixels(&decoded_again.pixels(), (2, 3), &expected, profile);
}
