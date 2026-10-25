//! Still decodes of an animated file (user decision 2026-10-25): the pixel
//! decodes return the first frame, and `decode_yuv420` refuses the file as
//! libwebp's `WebPDecode` does (its planes would come from a composited
//! canvas, not the frame's own YUV).
#![forbid(unsafe_code)]
#![cfg(not(target_arch = "wasm32"))]

use std::borrow::Cow;
use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
use zenwebp::decoder::{self, DecodeError};
use zenwebp::mux::{AnimationConfig, AnimationDecoder, AnimationEncoder};
use zenwebp::zencodec::WebpDecoderConfig;
use zenwebp::{DecodeConfig, DecodeRequest, EncoderConfig, PixelLayout};

const W: u32 = 8;
const H: u32 = 6;

fn animation() -> Vec<u8> {
    let mut enc = AnimationEncoder::new(W, H, AnimationConfig::default()).unwrap();
    for (i, ms) in [0u32, 50].into_iter().enumerate() {
        let px: Vec<u8> = (0..W * H)
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
fn yuv_decode_refuses_animated_files_like_libwebp() {
    let data = animation();
    assert!(
        webpx::decode_yuv(&data).is_err(),
        "libwebp oracle decodes it"
    );
    refused("decode_yuv420", decoder::decode_yuv420(&data));
    let config = DecodeConfig::default();
    refused(
        "DecodeRequest::decode_yuv420",
        DecodeRequest::new(&config, &data).decode_yuv420(),
    );
}

#[test]
fn pixel_decodes_of_animated_files_return_the_first_frame() {
    let data = animation();
    let first = AnimationDecoder::new(&data)
        .unwrap()
        .next_frame()
        .unwrap()
        .unwrap()
        .data;
    let rgb: Vec<u8> = first
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();

    assert_eq!(decoder::decode_rgba(&data).unwrap(), (first.clone(), W, H));
    assert_eq!(decoder::decode_rgb(&data).unwrap(), (rgb.clone(), W, H));
    let config = DecodeConfig::default();
    let (px, w, h, layout) = DecodeRequest::new(&config, &data).decode().unwrap();
    assert_eq!((w, h), (W, H));
    match layout {
        PixelLayout::Rgba8 => assert!(px == first),
        PixelLayout::Rgb8 => assert!(px == rgb),
        other => panic!("unexpected layout {other:?}"),
    }
    let mut buf = vec![0u8; (W * H * 4) as usize];
    decoder::decode_rgba_into(&data, &mut buf, W).unwrap();
    assert!(buf == first);

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
        got == first,
        "zencodec still decode returns the first frame"
    );
}
