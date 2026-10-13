use std::borrow::Cow;
use zencodec::animation::FrameDuration;
use zencodec::decode::{AnimationFrameDecoder as _, DecodeJob as _, DecoderConfig as _};
use zencodec::encode::{AnimationFrameEncoder as _, EncodeJob as _, EncoderConfig as _};
use zenwebp::zencodec::{WebpDecoderConfig, WebpEncoderConfig};

#[test]
fn ordinary_timing_and_exact_admission_match_encoded_frames() {
    let mut encoder = WebpEncoderConfig::lossless()
        .job()
        .animation_frame_encoder()
        .unwrap();
    let mut expected = Vec::new();
    for index in 0..30 {
        let bytes = [index * 7, 40, 60].repeat(256);
        let pixels = || {
            zenpixels::PixelSlice::new(&bytes, 16, 16, 48, zenpixels::PixelDescriptor::RGB8_SRGB)
                .unwrap()
        };
        if index == 0 {
            assert!(
                encoder
                    .push_frame_exact(pixels(), FrameDuration::new(1, 30).unwrap(), None)
                    .is_err()
            );
            assert!(
                encoder
                    .push_frame_exact(pixels(), FrameDuration::from_millis(0x0100_0000), None)
                    .is_err()
            );
        }
        expected.push(
            encoder
                .push_frame_timed(pixels(), FrameDuration::new(1, 30).unwrap(), None)
                .unwrap(),
        );
    }
    assert_eq!(expected[0], FrameDuration::from_millis(33));
    assert_eq!(expected[1], FrameDuration::from_millis(34));
    assert_eq!(
        expected
            .iter()
            .map(|d| d.ticks_at(1000).unwrap())
            .sum::<u64>(),
        1000
    );
    let mut decoder = WebpDecoderConfig::new()
        .job()
        .animation_frame_decoder(Cow::Owned(encoder.finish(None).unwrap().into_vec()), &[])
        .unwrap();
    for duration in expected {
        assert_eq!(
            decoder
                .render_next_frame_owned(None)
                .unwrap()
                .unwrap()
                .duration(),
            duration
        );
    }
    assert!(decoder.render_next_frame_owned(None).unwrap().is_none());
}

#[test]
fn cumulative_timestamp_overflow_rejects_before_acceptance() {
    let mut encoder = WebpEncoderConfig::lossless()
        .job()
        .animation_frame_encoder()
        .unwrap();
    let bytes = [20, 40, 60];
    let pixels = || {
        zenpixels::PixelSlice::new(&bytes, 1, 1, 3, zenpixels::PixelDescriptor::RGB8_SRGB).unwrap()
    };
    let maximum = FrameDuration::from_millis(0x00ff_ffff);
    for _ in 0..256 {
        encoder.push_frame_exact(pixels(), maximum, None).unwrap();
    }
    assert!(encoder.push_frame_exact(pixels(), maximum, None).is_err());
    encoder
        .push_frame_exact(pixels(), FrameDuration::from_millis(0), None)
        .unwrap();
    let bytes = encoder.finish(None).unwrap().into_vec();
    let mut decoder = WebpDecoderConfig::new()
        .job()
        .with_limits(zencodec::ResourceLimits::none())
        .animation_frame_decoder(Cow::Owned(bytes), &[])
        .unwrap();
    for _ in 0..256 {
        assert_eq!(
            decoder
                .render_next_frame_owned(None)
                .unwrap()
                .unwrap()
                .duration(),
            maximum
        );
    }
    assert_eq!(
        decoder
            .render_next_frame_owned(None)
            .unwrap()
            .unwrap()
            .duration(),
        FrameDuration::from_millis(0)
    );
    assert!(decoder.render_next_frame_owned(None).unwrap().is_none());
}
