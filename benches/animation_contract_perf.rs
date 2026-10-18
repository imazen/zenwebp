#![forbid(unsafe_code)]
//! Main-vs-PR105 performance witness. All timed work is in-memory native Rust.
//! Run the same source and settings on each revision; see benchmarks/pr105/.
use std::{borrow::Cow, path::PathBuf, sync::Arc, time::Duration};
use zenbench::prelude::*;
use zencodec::{
    decode::{AnimationFrameDecoder, Decode, DecodeJob, DecoderConfig},
    encode::{AnimationFrameEncoder, EncodeJob, EncoderConfig as _},
};
use zenpixels::{PixelBuffer, PixelDescriptor};
use zenwebp::{
    EncoderConfig, PixelLayout,
    mux::{AnimationConfig, AnimationDecoder, AnimationEncoder},
    zencodec::{WebpDecoderConfig, WebpEncoderConfig},
};

fn config(lossy: bool) -> EncoderConfig {
    if lossy {
        EncoderConfig::new_lossy().with_quality(75.0).with_method(4)
    } else {
        EncoderConfig::new_lossless().with_method(4)
    }
}
fn native_encode(frames: &[PixelBuffer], lossy: bool) -> usize {
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
fn shared_encode(frames: &[PixelBuffer], lossy: bool) -> usize {
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
fn native_decode(bytes: &[u8]) -> usize {
    let mut dec = AnimationDecoder::new(bytes).unwrap();
    let mut count = 0;
    while dec.decode_next().unwrap().is_some() {
        black_box(dec.current_frame_data());
        count += 1;
    }
    count
}
fn shared_decode(bytes: &[u8]) -> usize {
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
fn bench(suite: &mut Suite) {
    let root = PathBuf::from(
        std::env::var_os("ZENWEBP_PERF_FIXTURES")
            .expect("set ZENWEBP_PERF_FIXTURES to the common repository tests directory"),
    );
    let png = std::fs::read(root.join("reference/gallery1/1.png")).unwrap();
    let mut reader = png::Decoder::new(std::io::Cursor::new(png))
        .read_info()
        .unwrap();
    let mut photo = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut photo).unwrap();
    assert_eq!(info.color_type, png::ColorType::Rgb);
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    let frames: Arc<Vec<_>> = Arc::new(
        (0..3usize)
            .map(|i| {
                let mut data = Vec::with_capacity(256 * 256 * 3);
                for row in 0..256 {
                    let offset = ((row + i * 8) * info.width as usize + i * 16) * 3;
                    data.extend_from_slice(&photo[offset..offset + 256 * 3]);
                }
                PixelBuffer::from_vec(data, 256, 256, PixelDescriptor::RGB8_SRGB).unwrap()
            })
            .collect(),
    );
    for lossy in [false, true] {
        for shared in [false, true] {
            let name = format!(
                "{}_anim_encode_{}",
                if shared { "shared" } else { "native" },
                if lossy { "lossy" } else { "lossless" }
            );
            let frames = frames.clone();
            suite.compare(name, move |g| {
                g.config()
                    .warmup_time(Duration::from_millis(200))
                    .min_rounds(20)
                    .max_rounds(80)
                    .max_time(Duration::from_secs(3));
                g.bench("zenwebp", move |b| {
                    let frames = frames.clone();
                    b.iter(move || {
                        black_box(if shared {
                            shared_encode(&frames, lossy)
                        } else {
                            native_encode(&frames, lossy)
                        })
                    })
                });
            });
        }
        let path = root.join(if lossy {
            "images/animated/random_lossy.webp"
        } else {
            "images/animated/random_lossless.webp"
        });
        let bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
        for shared in [false, true] {
            let name = format!(
                "{}_anim_decode_{}",
                if shared { "shared" } else { "native" },
                if lossy { "lossy" } else { "lossless" }
            );
            let bytes = bytes.clone();
            suite.compare(name, move |g| {
                g.config()
                    .warmup_time(Duration::from_millis(200))
                    .min_rounds(20)
                    .max_rounds(80)
                    .max_time(Duration::from_secs(3));
                g.bench("zenwebp", move |b| {
                    let bytes = bytes.clone();
                    b.iter(move || {
                        black_box(if shared {
                            shared_decode(&bytes)
                        } else {
                            native_decode(&bytes)
                        })
                    })
                });
            });
        }
    }
    let bytes: Arc<[u8]> = std::fs::read(root.join("images/gallery1/1.webp"))
        .unwrap()
        .into();
    suite.compare("shared_still_decode_lossy", move |g| {
        g.config()
            .warmup_time(Duration::from_millis(200))
            .min_rounds(20)
            .max_rounds(80)
            .max_time(Duration::from_secs(3));
        g.bench("zenwebp", move |b| {
            let bytes = bytes.clone();
            b.iter(move || {
                black_box(
                    WebpDecoderConfig::new()
                        .job()
                        .decoder(Cow::Borrowed(&bytes), &[])
                        .unwrap()
                        .decode()
                        .unwrap(),
                );
            })
        });
    });
}
zenbench::main!(bench);
