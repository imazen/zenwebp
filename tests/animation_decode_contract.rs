use std::{
    borrow::Cow,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use zencodec::decode::{AnimationFrameDecoder, DecodeJob, DecoderConfig};
use zencodec::encode::{AnimationFrameEncoder, EncodeJob, EncoderConfig};
use zenpixels::{PixelBuf, PixelFormat};
use zenwebp::zencodec::{WebpDecoderConfig, WebpEncoderConfig};

fn view_icc<'a>(view: &'a zenpixels::PixelView<'_>) -> Option<&'a [u8]> {
    match &view.format().color {
        zenpixels::ColorSpace::Icc { profile, .. } => Some(profile),
        _ => None,
    }
}

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
    let pixels = PixelBuf::from_bytes(data, 128, 256, PixelFormat::RGBA8_SRGB).unwrap();
    let config = if lossy {
        WebpEncoderConfig::lossy()
    } else {
        WebpEncoderConfig::lossless()
    };
    let mut job = config.job();
    if icc {
        job = job.with_unfiltered_metadata(
            zencodec::Metadata::none()
                .with_icc(zenpixels_convert::icc_profiles::DISPLAY_P3_V4)
                .filtered(&zencodec::MetadataPolicy::PreserveExact),
        );
    }
    let mut encoder = job.animation_frame_encoder().unwrap();
    for _ in 0..3 {
        encoder.push_frame(pixels.as_view(), 10, None).unwrap();
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
                    view_icc(&frame.pixels()),
                    Some(zenpixels_convert::icc_profiles::DISPLAY_P3_V4)
                );
            } else {
                let frame = decoder.render_next_frame(None).unwrap().unwrap();
                assert_eq!(
                    view_icc(frame.pixels()),
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
            zenpixels::icc::IccIdentification::from_icc(view_icc(frame.pixels()).unwrap())
                .unwrap()
                .primaries,
            zenpixels::Primaries::DisplayP3
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
                job = job.with_pulse(zencodec::SharedPulse::new(
                    zencodec::how_far::StopOnly::new(PollLimit(calls.clone())),
                ));
            }
            let mut decoder = job
                .animation_frame_decoder(Cow::Borrowed(&encoded), &[])
                .unwrap();
            calls.store(0, Ordering::Relaxed);
            let per_call = PollLimit(calls.clone());
            let per_call_pulse = zencodec::how_far::StopOnly::borrowed(&per_call);
            let result = decoder.render_next_frame(if job_stop {
                None
            } else {
                Some(&per_call_pulse)
            });
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
    let pixels = PixelBuf::from_bytes(data, 17, 19, PixelFormat::RGBA8_SRGB).unwrap();
    let icc = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    for lossy in [false, true] {
        let config = if lossy {
            WebpEncoderConfig::lossy()
        } else {
            WebpEncoderConfig::lossless()
        };
        let output = config
            .job()
            .with_unfiltered_metadata(
                zencodec::Metadata::none()
                    .with_icc(icc)
                    .filtered(&zencodec::MetadataPolicy::PreserveExact),
            )
            .encoder()
            .unwrap()
            .encode(pixels.as_view())
            .unwrap();
        let rgba = WebpDecoderConfig::new()
            .job()
            .decoder(Cow::Borrowed(output.data()), &[])
            .unwrap()
            .decode()
            .unwrap()
            .pixels()
            .to_contiguous_bytes()
            .unwrap()
            .into_owned();
        for allow_icc in [None, Some(true), Some(false)] {
            let deny_icc = allow_icc == Some(false);
            let job = || {
                let job = WebpDecoderConfig::new().job();
                if let Some(allow) = allow_icc {
                    job.try_with_policy(
                        zencodec::decode::DecodePolicy::default().with_allow_icc(allow),
                    )
                    .unwrap()
                } else {
                    job
                }
            };
            for bgra in [false, true] {
                let preferred: &[PixelFormat] = if bgra {
                    &[PixelFormat::BGRA8_SRGB]
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
                assert_eq!(
                    decoded.pixels().to_contiguous_bytes().unwrap().as_ref(),
                    expected
                );
                let view = decoded.pixels();
                let descriptor = view.format();
                let primaries = match &descriptor.color {
                    zenpixels::ColorSpace::Icc { profile, .. } => {
                        zenpixels::icc::IccIdentification::from_icc(profile)
                            .unwrap()
                            .primaries
                    }
                    zenpixels::ColorSpace::Rgb { primaries, .. } => *primaries,
                    color => panic!("unexpected WebP color spec {color:?}"),
                };
                assert_eq!(primaries, zenpixels::Primaries::DisplayP3);
                assert_eq!(
                    descriptor.layout.order,
                    if bgra {
                        zenpixels::ChannelOrder::Bgra
                    } else {
                        zenpixels::ChannelOrder::Rgba
                    }
                );
                if !bgra {
                    assert_eq!(
                        &job().output_info(output.data()).unwrap().native_format,
                        descriptor
                    );
                }
                let check = |view: &zenpixels::PixelView<'_>| {
                    assert_eq!(view.format(), descriptor);
                    assert_eq!(view_icc(view), (!deny_icc).then_some(icc));
                };
                check(&decoded.pixels());
                let mut streaming = job()
                    .streaming_decoder(Cow::Borrowed(output.data()), preferred)
                    .unwrap();
                let mut streamed = Vec::new();
                while let Some((_, batch)) = streaming.next_batch().unwrap() {
                    check(&batch);
                    streamed.extend_from_slice(&batch.to_contiguous_bytes().unwrap());
                }
                assert_eq!(streamed, expected);
                let mut animation = job()
                    .animation_frame_decoder(Cow::Borrowed(output.data()), preferred)
                    .unwrap();
                let frame = animation.render_next_frame(None).unwrap().unwrap();
                check(frame.pixels());
                assert_eq!(
                    frame.pixels().to_contiguous_bytes().unwrap().as_ref(),
                    expected
                );
                assert!(animation.render_next_frame(None).unwrap().is_none());
            }
        }
    }
}
