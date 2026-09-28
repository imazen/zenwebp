use std::borrow::Cow;
use zencodec::animation::FrameDuration;
use zencodec::decode::{AnimationFrameDecoder, DecodeJob, DecoderConfig};
use zencodec::encode::{AnimationFrameEncoder, EncodeJob, EncoderConfig};
use zenpixels::{PixelBuffer, PixelDescriptor};
use zenwebp::mux::{AnimationConfig, AnimationDecoder, AnimationEncoder, WebPDemuxer};
use zenwebp::zencodec::{WebpDecoderConfig, WebpEncoderConfig};

fn pixels() -> PixelBuffer {
    PixelBuffer::from_vec(
        [12, 34, 56, 255].repeat(4),
        2,
        2,
        PixelDescriptor::RGBA8_SRGB,
    )
    .unwrap()
}

#[test]
fn single_frame_animation_preserves_timing_and_total_plays() {
    let pixels = pixels();
    for plays in [0, 1, 2, 65535] {
        for milliseconds in [0, 1, 0x00FF_FFFF] {
            let mut encoder = WebpEncoderConfig::lossless()
                .job()
                .with_loop_count(Some(plays))
                .animation_frame_encoder()
                .unwrap();
            encoder
                .push_frame_timed(
                    pixels.as_slice(),
                    FrameDuration::from_millis(milliseconds),
                    None,
                )
                .unwrap();
            let encoded = encoder.finish(None).unwrap();
            let demux = WebPDemuxer::new(encoded.data()).unwrap();
            assert!(
                demux.is_animated(),
                "explicit animation must retain its signaling"
            );
            assert_eq!(demux.frames().count(), 1);
            assert_eq!(demux.frame(1).unwrap().duration_ms, milliseconds);
            let mut decoder = WebpDecoderConfig::new()
                .job()
                .with_limits(zencodec::ResourceLimits::none())
                .animation_frame_decoder(Cow::Borrowed(encoded.data()), &[])
                .unwrap();
            assert_eq!(decoder.loop_count(), Some(plays));
            let frame = decoder.render_next_frame(None).unwrap().unwrap();
            assert_eq!(frame.duration(), FrameDuration::from_millis(milliseconds));
            assert_eq!(frame.pixels().row(0), pixels.as_slice().row(0));
            assert!(decoder.render_next_frame(None).unwrap().is_none());
        }
    }
    assert!(
        WebpEncoderConfig::lossless()
            .job()
            .with_loop_count(Some(65536))
            .animation_frame_encoder()
            .is_err()
    );
}

#[test]
fn exact_delays_reject_fractional_and_oversized_values_before_accepting() {
    let pixels = pixels();
    let mut encoder = WebpEncoderConfig::lossless()
        .job()
        .animation_frame_encoder()
        .unwrap();
    for duration in [
        FrameDuration::new(1, 30000).unwrap(),
        FrameDuration::from_millis(0x0100_0000),
        FrameDuration::new(u64::MAX, 1).unwrap(),
    ] {
        assert!(
            encoder
                .push_frame_timed(pixels.as_slice(), duration, None)
                .is_err()
        );
    }
    for milliseconds in [0, 1, 10] {
        encoder
            .push_frame_timed(
                pixels.as_slice(),
                FrameDuration::from_millis(milliseconds),
                None,
            )
            .unwrap();
    }
    let encoded = encoder.finish(None).unwrap();
    let demux = WebPDemuxer::new(encoded.data()).unwrap();
    assert_eq!(
        demux
            .frames()
            .map(|frame| frame.duration_ms)
            .collect::<Vec<_>>(),
        [0, 1, 10]
    );
}

#[test]
fn long_timeline_survives_u32_milliseconds_on_encode_and_native_decode() {
    let pixels = pixels();
    let mut encoder = WebpEncoderConfig::lossless()
        .job()
        .animation_frame_encoder()
        .unwrap();
    let milliseconds = 0x00FF_FFFF;
    for _ in 0..260 {
        encoder
            .push_frame_timed(
                pixels.as_slice(),
                FrameDuration::from_millis(milliseconds),
                None,
            )
            .unwrap();
    }
    let encoded = encoder.finish(None).unwrap();
    let mut decoder = AnimationDecoder::new(encoded.data()).unwrap();
    for index in 0..260_u64 {
        let frame = decoder.next_frame().unwrap().unwrap();
        assert_eq!(frame.timestamp_ms, index * u64::from(milliseconds));
        assert_eq!(frame.duration_ms, milliseconds);
        assert_eq!(frame.data, pixels.copy_to_contiguous_bytes());
    }
    assert!(decoder.next_frame().unwrap().is_none());
}

#[test]
fn native_timestamp_errors_retain_pending_frame() {
    let mut encoder = AnimationEncoder::new(2, 2, AnimationConfig::default()).unwrap();
    let config = zenwebp::EncoderConfig::new_lossless();
    let pixels = pixels().copy_to_contiguous_bytes();
    encoder
        .add_frame(&pixels, zenwebp::PixelLayout::Rgba8, 100, &config)
        .unwrap();
    assert!(
        encoder
            .add_frame(&pixels, zenwebp::PixelLayout::Rgba8, 99, &config)
            .is_err()
    );
    assert!(
        encoder
            .add_frame(&pixels, zenwebp::PixelLayout::Rgba8, u64::MAX, &config)
            .is_err()
    );
    encoder
        .add_frame(&pixels, zenwebp::PixelLayout::Rgba8, 101, &config)
        .unwrap();
    let bytes = encoder.finalize_animation(1).unwrap();
    let demux = WebPDemuxer::new(&bytes).unwrap();
    assert_eq!(
        demux
            .frames()
            .map(|frame| frame.duration_ms)
            .collect::<Vec<_>>(),
        [1, 1]
    );
}

#[test]
fn animation_retains_metadata_and_synthesizes_cicp_color_like_still_images() {
    let pixels = pixels();
    let icc = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    for explicit_icc in [false, true] {
        let mut metadata = zencodec::Metadata::none()
            .with_exif(&b"Exif\0\0animation-exif"[..])
            .with_xmp(&b"<x:xmpmeta>animation</x:xmpmeta>"[..]);
        if explicit_icc {
            metadata = metadata.with_icc(icc);
        } else {
            metadata = metadata.with_cicp(zenpixels::Cicp::DISPLAY_P3);
        }
        let mut encoder = WebpEncoderConfig::lossless()
            .job()
            .with_metadata_policy(metadata, zencodec::MetadataPolicy::PreserveExact)
            .animation_frame_encoder()
            .unwrap();
        encoder
            .push_frame_timed(pixels.as_slice(), FrameDuration::from_millis(10), None)
            .unwrap();
        let output = encoder.finish(None).unwrap();
        let demux = WebPDemuxer::new(output.data()).unwrap();
        assert_eq!(demux.icc_profile(), Some(icc));
        assert_eq!(demux.exif(), Some(&b"Exif\0\0animation-exif"[..]));
        assert_eq!(demux.xmp(), Some(&b"<x:xmpmeta>animation</x:xmpmeta>"[..]));
    }
}

#[test]
fn animation_frame_limits_and_dimensions_reject_before_acceptance() {
    let pixels = pixels();
    let mut encoder = WebpEncoderConfig::lossless()
        .job()
        .with_limits(zencodec::ResourceLimits::none().with_max_frames(1))
        .animation_frame_encoder()
        .unwrap();
    encoder.push_frame(pixels.as_slice(), 10, None).unwrap();
    assert!(encoder.push_frame(pixels.as_slice(), 20, None).is_err());
    let output = encoder.finish(None).unwrap();
    assert_eq!(WebPDemuxer::new(output.data()).unwrap().frames().count(), 1);
    let mut encoder = WebpEncoderConfig::lossless()
        .job()
        .animation_frame_encoder()
        .unwrap();
    encoder.push_frame(pixels.as_slice(), 10, None).unwrap();
    let mismatch = PixelBuffer::from_vec(vec![0; 4], 1, 1, PixelDescriptor::RGBA8_SRGB).unwrap();
    assert!(encoder.push_frame(mismatch.as_slice(), 20, None).is_err());
    let output = encoder.finish(None).unwrap();
    assert_eq!(WebPDemuxer::new(output.data()).unwrap().frames().count(), 1);
}

#[test]
fn cancellation_reaches_native_animation_kernels_from_job_and_per_call_tokens() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
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
    let data: Vec<_> = (0..64 * 64 * 4)
        .map(|i| ((i * 37 + i / 5 * 113) % 256) as u8)
        .collect();
    let pixels = PixelBuffer::from_vec(data, 64, 64, PixelDescriptor::RGBA8_SRGB).unwrap();
    for job_stop in [false, true] {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut job = WebpEncoderConfig::lossless().job();
        if job_stop {
            job = job.with_stop(zencodec::StopToken::new(PollLimit(calls.clone())));
        }
        let mut encoder = job.animation_frame_encoder().unwrap();
        let per_call = PollLimit(calls.clone());
        let stop = if job_stop {
            None
        } else {
            Some(&per_call as &dyn enough::Stop)
        };
        assert!(
            encoder.push_frame(pixels.as_slice(), 10, stop).is_err(),
            "job_stop={job_stop}: encoder ignored cancellation"
        );
        assert!(
            calls.load(Ordering::Relaxed) > 8,
            "must reach internal polls beyond wrapper checks"
        );
    }
}

#[test]
fn native_animation_rejects_short_buffers_and_bad_regions_without_losing_pending_frame() {
    use zenwebp::mux::{BlendMethod, DisposeMethod};
    let config = zenwebp::EncoderConfig::new_lossless();
    for layout in [
        zenwebp::PixelLayout::L8,
        zenwebp::PixelLayout::La8,
        zenwebp::PixelLayout::Rgb8,
        zenwebp::PixelLayout::Bgr8,
        zenwebp::PixelLayout::Rgba8,
        zenwebp::PixelLayout::Bgra8,
        zenwebp::PixelLayout::Argb8,
        zenwebp::PixelLayout::Yuv420,
    ] {
        let mut encoder = AnimationEncoder::new(2, 2, AnimationConfig::default()).unwrap();
        let pixels = pixels().copy_to_contiguous_bytes();
        encoder
            .add_frame(&pixels, zenwebp::PixelLayout::Rgba8, 0, &config)
            .unwrap();
        assert!(encoder.add_frame(&[], layout, 10, &config).is_err());
        for (w, h, x, y) in [
            (1, 1, 1, 0),
            (1, 1, 0, 1),
            (2, 2, 2, 0),
            (2, 2, 0, u32::MAX - 1),
            (0, 1, 0, 0),
        ] {
            assert!(
                encoder
                    .add_frame_advanced(
                        &pixels,
                        zenwebp::PixelLayout::Rgba8,
                        w,
                        h,
                        x,
                        y,
                        10,
                        &config,
                        DisposeMethod::None,
                        BlendMethod::Overwrite
                    )
                    .is_err()
            );
        }
        let output = encoder.finalize_animation(10).unwrap();
        let mut decoder = AnimationDecoder::new(&output).unwrap();
        assert_eq!(decoder.next_frame().unwrap().unwrap().data, pixels);
        assert!(decoder.next_frame().unwrap().is_none());
    }
}
