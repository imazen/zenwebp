//! Native still-image decodes refuse animated files, as libwebp's
//! `WebPDecode` does (user decision 2026-10-25). Frames come from
//! `AnimationDecoder`; the zencodec still decode keeps its contract of
//! returning the first frame.
#![forbid(unsafe_code)]
#![cfg(not(target_arch = "wasm32"))]

use std::borrow::Cow;
use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
use zenwebp::decoder::{self, DecodeError};
use zenwebp::mux::{AnimationConfig, AnimationDecoder, AnimationEncoder};
use zenwebp::zencodec::WebpDecoderConfig;
use zenwebp::{DecodeConfig, DecodeRequest, EncoderConfig, PixelLayout};

fn animation() -> Vec<u8> {
    let (w, h) = (8, 6);
    let mut enc = AnimationEncoder::new(w, h, AnimationConfig::default()).unwrap();
    for (i, ms) in [0u32, 50].into_iter().enumerate() {
        let px: Vec<u8> = (0..w * h)
            .flat_map(|p| [(p * 9) as u8, (i * 120) as u8, 77, 255])
            .collect();
        enc.add_frame(&px, PixelLayout::Rgba8, ms, &EncoderConfig::new_lossless())
            .unwrap();
    }
    enc.finalize(50).unwrap()
}

fn refused<T: std::fmt::Debug>(label: &str, r: Result<T, whereat::At<DecodeError>>) {
    match r {
        Err(e) if matches!(e.error(), DecodeError::UnsupportedFeature(_)) => {}
        other => panic!("{label}: expected UnsupportedFeature, got {other:?}"),
    }
}

#[test]
fn still_decodes_refuse_animated_files_like_libwebp() {
    let data = animation();
    assert!(
        webpx::decode_rgba(&data).is_err(),
        "libwebp oracle decodes it"
    );
    assert!(
        webpx::decode_yuv(&data).is_err(),
        "libwebp oracle decodes it"
    );

    let mut buf = vec![0u8; 8 * 6 * 4];
    refused("decode_rgba", decoder::decode_rgba(&data));
    refused("decode_rgb", decoder::decode_rgb(&data));
    refused("decode_bgra", decoder::decode_bgra(&data));
    refused("decode_bgr", decoder::decode_bgr(&data));
    refused("decode_argb", decoder::decode_argb(&data));
    refused(
        "decode_rgba_premultiplied",
        decoder::decode_rgba_premultiplied(&data),
    );
    refused("decode_rgb565", decoder::decode_rgb565(&data));
    refused("decode_rgba4444", decoder::decode_rgba4444(&data));
    refused("decode_yuv420", decoder::decode_yuv420(&data));
    refused(
        "decode_rgba_into",
        decoder::decode_rgba_into(&data, &mut buf, 8),
    );
    let config = DecodeConfig::default();
    refused(
        "DecodeRequest::decode",
        DecodeRequest::new(&config, &data).decode(),
    );
    refused(
        "DecodeRequest::decode_yuv420",
        DecodeRequest::new(&config, &data).decode_yuv420(),
    );

    // The animation decoder and the zencodec still decode (first frame) work.
    let first = AnimationDecoder::new(&data)
        .unwrap()
        .next_frame()
        .unwrap()
        .unwrap();
    let out = WebpDecoderConfig::new()
        .job()
        .decoder(
            Cow::Borrowed(&data),
            &[zenpixels::PixelDescriptor::RGBA8_SRGB],
        )
        .unwrap()
        .decode()
        .unwrap();
    let px = out.pixels();
    let mut got = Vec::new();
    for y in 0..px.rows() {
        got.extend_from_slice(px.row(y));
    }
    assert!(
        got == first.data,
        "zencodec still decode returns the first frame"
    );
}
