//! VP8L at the maximum dimension (16384) must round-trip.
//!
//! Regression for a review finding (2026-08-21): both the demuxer and the
//! decoder parsed VP8L dimensions as `(1 + header) & 0x3FFF` instead of
//! `(header & 0x3FFF) + 1`. For the max 14-bit field (0x3FFF → dimension
//! 16384) the `+1` carried into bit 14 and masked to 0, so a legal
//! 16384-wide/‑tall lossless image was reported as 0 pixels — and in the
//! decoder path the VP8L stream's own dimension check then rejected the
//! whole file. 16384×1 keeps the pixel count tiny while hitting the exact
//! edge.

use zenwebp::{
    DecodeConfig, DecodeRequest, EncodeRequest, EncoderConfig, LosslessConfig, PixelLayout,
};

fn roundtrip_max_dim(w: u32, h: u32) {
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for i in 0..(w * h) {
        rgba.extend_from_slice(&[(i * 7) as u8, (i * 13) as u8, (i * 29) as u8, 255]);
    }
    let cfg = LosslessConfig::new().with_method(4).with_exact(true);
    let webp = EncodeRequest::new(
        &EncoderConfig::Lossless(cfg),
        &rgba,
        PixelLayout::Rgba8,
        w,
        h,
    )
    .encode()
    .expect("encode 16384-edge VP8L");

    // Decode path: must not reject the max-dimension file.
    let (pixels, dw, dh, _) = DecodeRequest::new(&DecodeConfig::default(), &webp)
        .decode()
        .expect("decode 16384-edge VP8L (was rejected by the dimension-mask bug)");
    assert_eq!((dw, dh), (w, h), "decoded dimensions must match");
    let rgb: Vec<u8> = rgba
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    match pixels.len() {
        n if n == (w * h * 4) as usize => assert_eq!(pixels, rgba),
        _ => assert_eq!(pixels, rgb),
    }

    // Demuxer path: canvas dimensions must report 16384, not 0.
    let demux = zenwebp::mux::WebPDemuxer::new(&webp).expect("demux 16384-edge VP8L");
    assert_eq!(
        demux.canvas_width(),
        w,
        "demuxer canvas_width wrong at the edge"
    );
    assert_eq!(
        demux.canvas_height(),
        h,
        "demuxer canvas_height wrong at the edge"
    );
}

#[test]
fn vp8l_width_16384_roundtrips() {
    roundtrip_max_dim(16384, 1);
}

#[test]
fn vp8l_height_16384_roundtrips() {
    roundtrip_max_dim(1, 16384);
}
