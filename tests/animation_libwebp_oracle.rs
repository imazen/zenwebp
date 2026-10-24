//! The zencodec animation encoder and decoder, read back by libwebp (webpx)
//! instead of our own demuxer: frame count, timing, loop count, pixels, and
//! the ICC/EXIF/XMP the job carries.
#![forbid(unsafe_code)]
#![cfg(not(target_arch = "wasm32"))]
//!
//! Durations come from libwebp's per-frame end timestamps. webpx's
//! `Frame::duration_ms` is not used: `decode_all` overwrites the last frame's
//! duration with the previous frame's (or 100 ms for a single frame).

use std::borrow::Cow;
use zencodec::animation::FrameDuration;
use zencodec::decode::{AnimationFrameDecoder as _, DecodeJob as _, DecoderConfig as _};
use zencodec::encode::{AnimationFrameEncoder as _, EncodeJob as _, EncoderConfig as _};
use zenpixels::{PixelBuffer, PixelDescriptor};
use zenwebp::zencodec::{WebpDecoderConfig, WebpEncoderConfig};

const EXIF: &[u8] = b"Exif\0\0oracle-exif";
const XMP: &[u8] = b"<x:xmpmeta>oracle</x:xmpmeta>";

fn frame(seed: u8) -> PixelBuffer {
    let px: Vec<u8> = (0..8 * 8)
        .flat_map(|i| {
            let i = i as u8;
            [i.wrapping_mul(7) ^ seed, seed.wrapping_add(i), 255 - i, 255]
        })
        .collect();
    PixelBuffer::from_vec(px, 8, 8, PixelDescriptor::RGBA8_SRGB).unwrap()
}

/// libwebp frame durations: differences of `WebPAnimDecoderGetNext`'s end
/// timestamps.
fn libwebp_durations(frames: &[webpx::Frame]) -> Vec<u32> {
    let mut prev = 0;
    frames
        .iter()
        .map(|f| {
            let d = (f.timestamp_ms - prev) as u32;
            prev = f.timestamp_ms;
            d
        })
        .collect()
}

fn encode(durations: &[u32], plays: u32, metadata: zencodec::Metadata) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut enc = WebpEncoderConfig::lossless()
        .job()
        .with_loop_count(Some(plays))
        .with_metadata_policy(metadata, zencodec::MetadataPolicy::PreserveExact)
        .animation_frame_encoder()
        .unwrap();
    let mut sources = Vec::new();
    for (i, &ms) in durations.iter().enumerate() {
        let f = frame(i as u8 * 40 + 1);
        sources.push(f.as_slice().as_strided_bytes().to_vec());
        enc.push_frame_timed(f.as_slice(), FrameDuration::from_millis(ms), None)
            .unwrap();
    }
    (enc.finish(None).unwrap().data().to_vec(), sources)
}

/// Every frame, its duration, the loop count and the metadata survive, as
/// libwebp reads them.
#[test]
fn animation_encode_round_trips_through_libwebp() {
    let icc = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    let durations = [40, 0, 130];
    let metadata = zencodec::Metadata::none()
        .with_icc(icc)
        .with_exif(EXIF)
        .with_xmp(XMP);
    let (data, sources) = encode(&durations, 3, metadata);

    let mut lib = webpx::AnimationDecoder::new(&data).unwrap();
    let info = lib.info().clone();
    assert_eq!(info.frame_count, 3);
    assert_eq!(info.loop_count, 3);
    let frames = lib.decode_all().unwrap();
    assert_eq!(libwebp_durations(&frames), durations);
    for (i, (f, src)) in frames.iter().zip(&sources).enumerate() {
        assert!(&f.data == src, "frame {i} pixels");
    }
    assert_eq!(webpx::get_icc_profile(&data).unwrap().as_deref(), Some(icc));
    assert_eq!(webpx::get_exif(&data).unwrap().as_deref(), Some(EXIF));
    assert_eq!(webpx::get_xmp(&data).unwrap().as_deref(), Some(XMP));
}

/// A one-frame animation stays an animation, with its duration and loop
/// count, for libwebp too.
#[test]
fn single_frame_animation_stays_animated_for_libwebp() {
    let (data, sources) = encode(&[250], 2, zencodec::Metadata::none());
    let mut lib = webpx::AnimationDecoder::new(&data).unwrap();
    assert_eq!(lib.info().frame_count, 1);
    assert_eq!(lib.info().loop_count, 2);
    let frames = lib.decode_all().unwrap();
    assert_eq!(libwebp_durations(&frames), [250]);
    assert!(frames[0].data == sources[0]);
    assert!(webpx::get_xmp(&data).unwrap().is_none());
}

/// Decoded animation frames match libwebp's pixels and carry the file's ICC
/// profile instead of an sRGB label.
#[test]
fn animation_decode_matches_libwebp_and_keeps_icc() {
    let icc = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    let (data, _) = encode(&[40, 60], 0, zencodec::Metadata::none().with_icc(icc));
    let lib = webpx::AnimationDecoder::new(&data)
        .unwrap()
        .decode_all()
        .unwrap();
    let mut dec = WebpDecoderConfig::new()
        .job()
        .animation_frame_decoder(Cow::Borrowed(&data), &[PixelDescriptor::RGBA8_SRGB])
        .unwrap();
    let durations = libwebp_durations(&lib);
    assert_eq!(durations, [40, 60]);
    for (i, l) in lib.iter().enumerate() {
        let f = dec.render_next_frame(None).unwrap().unwrap();
        let px = f.pixels();
        assert_eq!(f.duration(), FrameDuration::from_millis(durations[i]));
        let ctx = px.color_context().expect("frame carries a color context");
        assert_eq!(ctx.icc.as_deref(), Some(icc), "frame {i} ICC");
        let mut got = Vec::new();
        for y in 0..px.rows() {
            got.extend_from_slice(px.row(y));
        }
        assert!(got == l.data, "frame {i} pixels");
    }
    assert!(dec.render_next_frame(None).unwrap().is_none());
}
