use codec::{
    EncoderConfig, PixelLayout,
    mux::{AnimationConfig, AnimationDecoder, AnimationEncoder},
    zencodec::{WebpDecoderConfig, WebpEncoderConfig},
};
use std::borrow::Cow;
use zenbench::black_box;
use zencodec::{
    decode::{AnimationFrameDecoder, Decode, DecodeJob, DecoderConfig},
    encode::{AnimationFrameEncoder, EncodeJob, EncoderConfig as _},
};
use zenpixels::PixelBuffer;
fn config(lossy: bool) -> EncoderConfig {
    if lossy {
        EncoderConfig::new_lossy().with_quality(75.0).with_method(4)
    } else {
        EncoderConfig::new_lossless().with_method(4)
    }
}
pub fn native_encode(frames: &[PixelBuffer], lossy: bool) -> usize {
    let cfg = config(lossy);
    let mut enc = AnimationEncoder::new(256, 256, AnimationConfig::default()).unwrap();
    for (i, frame) in frames.iter().enumerate() {
        enc.add_frame(
            frame.as_contiguous_bytes().unwrap(),
            PixelLayout::Rgb8,
            i as u32 * 20,
            &cfg,
        )
        .unwrap();
    }
    enc.finalize(20).unwrap().len()
}
pub fn shared_encode(frames: &[PixelBuffer], lossy: bool) -> usize {
    let cfg = if lossy {
        WebpEncoderConfig::lossy().with_quality(75.0).with_method(4)
    } else {
        WebpEncoderConfig::lossless().with_method(4)
    };
    let mut enc = cfg.job().animation_frame_encoder().unwrap();
    for frame in frames {
        enc.push_frame(frame.as_slice(), 20, None).unwrap();
    }
    enc.finish(None).unwrap().len()
}
pub fn native_decode(bytes: &[u8]) -> usize {
    let mut dec = AnimationDecoder::new(bytes).unwrap();
    let mut count = 0;
    while dec.decode_next().unwrap().is_some() {
        black_box(dec.current_frame_data());
        count += 1;
    }
    count
}
pub fn shared_decode(bytes: &[u8]) -> usize {
    let mut dec = WebpDecoderConfig::new()
        .job()
        .animation_frame_decoder(Cow::Borrowed(bytes), &[])
        .unwrap();
    let mut count = 0;
    while let Some(frame) = dec.render_next_frame(None).unwrap() {
        black_box(frame.pixels().row(0));
        count += 1;
    }
    count
}

pub fn still_decode(bytes: &[u8]) -> usize {
    let output = WebpDecoderConfig::new()
        .job()
        .decoder(Cow::Borrowed(bytes), &[])
        .unwrap()
        .decode()
        .unwrap();
    black_box(output.pixels().row(0));
    output.pixels().rows() as usize
}
