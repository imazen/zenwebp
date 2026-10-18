use std::{
    borrow::Cow,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use zencodec::decode::{AnimationFrameDecoder, DecodeJob, DecoderConfig};
use zencodec::encode::{AnimationFrameEncoder, EncodeJob, EncoderConfig};
use zenpixels::{PixelBuffer, PixelDescriptor};
use zenwebp::zencodec::{WebpDecoderConfig, WebpEncoderConfig};

fn encode(lossy: bool, icc: bool) -> Vec<u8> {
    let data: Vec<_> = (0..128 * 256 * 4)
        .map(|i| {
            if i % 4 == 3 {
                255
            } else {
                ((i * 73 + i / 257 * 31) % 256) as u8
            }
        })
        .collect();
    let pixels = PixelBuffer::from_vec(data, 128, 256, PixelDescriptor::RGBA8_SRGB).unwrap();
    let config = if lossy {
        WebpEncoderConfig::lossy()
    } else {
        WebpEncoderConfig::lossless()
    };
    let mut job = config.job();
    if icc {
        job = job.with_metadata_policy(
            zencodec::Metadata::none().with_icc(zenpixels_convert::icc_profiles::DISPLAY_P3_V4),
            zencodec::MetadataPolicy::PreserveExact,
        );
    }
    let mut encoder = job.animation_frame_encoder().unwrap();
    for _ in 0..3 {
        encoder.push_frame(pixels.as_slice(), 10, None).unwrap();
    }
    encoder.finish(None).unwrap().into_vec()
}

#[test]
fn decoded_animation_frames_retain_icc_context_in_borrowed_and_owned_output() {
    let encoded = encode(false, true);
    for owned in [false, true] {
        let mut decoder = WebpDecoderConfig::new()
            .job()
            .animation_frame_decoder(Cow::Borrowed(&encoded), &[])
            .unwrap();
        for _ in 0..3 {
            if owned {
                let frame = decoder.render_next_frame_owned(None).unwrap().unwrap();
                assert_eq!(
                    frame
                        .pixels()
                        .color_context()
                        .and_then(|c| c.icc.as_deref()),
                    Some(zenpixels_convert::icc_profiles::DISPLAY_P3_V4)
                );
            } else {
                let frame = decoder.render_next_frame(None).unwrap().unwrap();
                assert_eq!(
                    frame
                        .pixels()
                        .color_context()
                        .and_then(|c| c.icc.as_deref()),
                    Some(zenpixels_convert::icc_profiles::DISPLAY_P3_V4)
                );
            }
        }
    }
}

#[test]
fn decoded_color_context_can_be_reencoded_without_relabeling_samples() {
    let encoded = encode(false, true);
    let mut decoder = WebpDecoderConfig::new()
        .job()
        .animation_frame_decoder(Cow::Borrowed(&encoded), &[])
        .unwrap();
    let mut encoder = WebpEncoderConfig::lossless()
        .job()
        .animation_frame_encoder()
        .unwrap();
    while let Some(frame) = decoder.render_next_frame(None).unwrap() {
        assert_eq!(
            frame.pixels().descriptor().primaries,
            zenpixels::ColorPrimaries::DisplayP3
        );
        encoder
            .push_frame_timed(frame.pixels().clone(), frame.duration(), None)
            .unwrap();
    }
    let output = encoder.finish(None).unwrap();
    let demux = zenwebp::mux::WebPDemuxer::new(output.data()).unwrap();
    assert_eq!(
        demux.icc_profile(),
        Some(zenpixels_convert::icc_profiles::DISPLAY_P3_V4)
    );
    let mut a = zenwebp::mux::AnimationDecoder::new(&encoded).unwrap();
    let mut b = zenwebp::mux::AnimationDecoder::new(output.data()).unwrap();
    for _ in 0..3 {
        assert_eq!(
            a.next_frame().unwrap().unwrap().data,
            b.next_frame().unwrap().unwrap().data
        );
    }
}

#[test]
fn animation_decode_cancellation_reaches_lossy_and_lossless_kernels() {
    struct PollLimit(Arc<AtomicUsize>);
    impl enough::Stop for PollLimit {
        fn check(&self) -> Result<(), enough::StopReason> {
            if self.0.fetch_add(1, Ordering::Relaxed) >= 8 {
                Err(enough::StopReason::Cancelled)
            } else {
                Ok(())
            }
        }
    }
    for lossy in [false, true] {
        let encoded = encode(lossy, false);
        for job_stop in [false, true] {
            let calls = Arc::new(AtomicUsize::new(0));
            let mut job = WebpDecoderConfig::new().job();
            if job_stop {
                job = job.with_stop(zencodec::StopToken::new(PollLimit(calls.clone())));
            }
            let mut decoder = job
                .animation_frame_decoder(Cow::Borrowed(&encoded), &[])
                .unwrap();
            calls.store(0, Ordering::Relaxed);
            let per_call = PollLimit(calls.clone());
            let result = decoder.render_next_frame(if job_stop { None } else { Some(&per_call) });
            assert!(
                result.is_err(),
                "lossy={lossy}, job_stop={job_stop}: ignored internal stop"
            );
            assert!(calls.load(Ordering::Relaxed) > 8);
            assert!(
                decoder.render_next_frame(None).is_err(),
                "a partially decoded frame poisons subsequent output"
            );
        }
    }
}

#[test]
fn still_decoder_views_keep_icc_context_and_bgra_pixels() {
    use zencodec::decode::{Decode, StreamingDecode};
    use zencodec::encode::Encoder;
    let data: Vec<_> = (0..17 * 19 * 4)
        .map(|i| ((i * 73 + i / 11) % 255) as u8)
        .collect();
    let pixels = PixelBuffer::from_vec(data, 17, 19, PixelDescriptor::RGBA8_SRGB).unwrap();
    let icc = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    for lossy in [false, true] {
        let config = if lossy {
            WebpEncoderConfig::lossy()
        } else {
            WebpEncoderConfig::lossless()
        };
        let output = config
            .job()
            .with_metadata_policy(
                zencodec::Metadata::none().with_icc(icc),
                zencodec::MetadataPolicy::PreserveExact,
            )
            .encoder()
            .unwrap()
            .encode(pixels.as_slice())
            .unwrap();
        let rgba = WebpDecoderConfig::new()
            .job()
            .decoder(Cow::Borrowed(output.data()), &[])
            .unwrap()
            .decode()
            .unwrap()
            .pixels()
            .contiguous_bytes()
            .into_owned();
        for allow_icc in [None, Some(true), Some(false)] {
            let deny_icc = allow_icc == Some(false);
            let job = || {
                let job = WebpDecoderConfig::new().job();
                if let Some(allow) = allow_icc {
                    job.with_policy(zencodec::decode::DecodePolicy::default().with_allow_icc(allow))
                } else {
                    job
                }
            };
            for bgra in [false, true] {
                let preferred: &[PixelDescriptor] = if bgra {
                    &[PixelDescriptor::BGRA8_SRGB]
                } else {
                    &[]
                };
                let decoded = job()
                    .decoder(Cow::Borrowed(output.data()), preferred)
                    .unwrap()
                    .decode()
                    .unwrap();
                let mut expected = rgba.clone();
                if bgra {
                    for pixel in expected.as_chunks_mut::<4>().0 {
                        pixel.swap(0, 2);
                    }
                }
                assert_eq!(decoded.pixels().contiguous_bytes().as_ref(), expected);
                let descriptor = decoded.pixels().descriptor();
                assert_eq!(descriptor.primaries, zenpixels::ColorPrimaries::DisplayP3);
                assert_eq!(
                    descriptor.layout(),
                    if bgra {
                        zenpixels::ChannelLayout::Bgra
                    } else {
                        zenpixels::ChannelLayout::Rgba
                    }
                );
                if !bgra {
                    assert_eq!(
                        job().output_info(output.data()).unwrap().native_format,
                        descriptor
                    );
                }
                let check = |view: &zenpixels::PixelSlice<'_>| {
                    assert_eq!(view.descriptor(), descriptor);
                    assert_eq!(
                        view.color_context().and_then(|c| c.icc.as_deref()),
                        (!deny_icc).then_some(icc)
                    );
                };
                check(&decoded.pixels());
                let mut streaming = job()
                    .streaming_decoder(Cow::Borrowed(output.data()), preferred)
                    .unwrap();
                let mut streamed = Vec::new();
                while let Some((_, batch)) = streaming.next_batch().unwrap() {
                    check(&batch);
                    streamed.extend_from_slice(&batch.contiguous_bytes());
                }
                assert_eq!(streamed, expected);
                let mut animation = job()
                    .animation_frame_decoder(Cow::Borrowed(output.data()), preferred)
                    .unwrap();
                let frame = animation.render_next_frame(None).unwrap().unwrap();
                check(frame.pixels());
                assert_eq!(frame.pixels().contiguous_bytes().as_ref(), expected);
                assert!(animation.render_next_frame(None).unwrap().is_none());
            }
        }
    }
}
