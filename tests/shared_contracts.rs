use std::borrow::Cow;
use zencodec::Metadata;
use zencodec::animation::FrameDuration as Duration;
use zencodec::decode::{Decode, DecodeJob, DecodePolicy, DecoderConfig};
use zencodec::encode::{EncodeJob, EncodePolicy, EncoderConfig};
use zencodec_testkit as tk;
use zenwebp::zencodec::{WebpDecoderConfig as D, WebpEncoderConfig as E};

#[test]
fn shared_timing_boundaries_and_dispatch() {
    let d = |n, den| Duration::new(n, den).unwrap();
    let ms = Duration::from_millis;
    let img = tk::TestImage::rgb8_gradient(16, 16);
    let exact = [
        (d(1, 30), None),
        (ms(17), Some(ms(17))),
        (ms(0x0100_0000), None),
        (ms(0), Some(ms(0))),
        (ms(0x00ff_ffff), Some(ms(0x00ff_ffff))),
        (d(u64::MAX, 1), None),
        (ms(10), Some(ms(10))),
    ];
    tk::check_animation_timing(E::lossless(), D::new(), img.as_slice(), true, &exact).unwrap();
    let ordinary = [
        (d(1, 30), Some(ms(33))),
        (d(u64::MAX, 1), None),
        (d(1, 30), Some(ms(34))),
        (ms(0), Some(ms(0))),
        (d(1, 30), Some(ms(33))),
    ];
    tk::check_animation_timing(E::lossless(), D::new(), img.as_slice(), false, &ordinary).unwrap();
}

#[test]
fn policy_and_inventory_contracts() {
    let img = tk::TestImage::rgb8_gradient(16, 16);
    let enc = E::lossless();
    let bytes = enc.clone().job().encode(img.as_slice()).unwrap();
    let expected = D::new()
        .job()
        .decoder(Cow::Borrowed(bytes.data()), &[])
        .unwrap()
        .decode()
        .unwrap()
        .metadata();
    for (policy, accepted) in [
        (EncodePolicy::none(), true),
        (EncodePolicy::strip_all(), false),
    ] {
        tk::check_encode_policy_admission(
            enc.clone(),
            D::new(),
            img.as_slice(),
            Metadata::none(),
            EncodePolicy::none(),
            policy,
            accepted,
            &expected,
        )
        .unwrap();
    }
    for (policy, accepted) in [
        (DecodePolicy::none(), true),
        (DecodePolicy::strict(), false),
    ] {
        tk::check_decode_policy_admission(
            D::new(),
            bytes.data(),
            DecodePolicy::none(),
            policy,
            accepted,
            &expected,
        )
        .unwrap();
    }
    tk::check_inventory_dispatch(D::new(), bytes.data()).unwrap();
    #[cfg(feature = "inventory")]
    tk::check_inventory_input_limit(D::new(), bytes.data()).unwrap();
}

#[test]
fn unsupported_causes_survive_erasure() {
    use zencodec::encode::AnimationFrameEncoder;
    use zencodec::{CodecErrorExt, UnsupportedOperation};
    let img = tk::TestImage::rgb8_gradient(16, 16);
    let mut encoder = E::lossless().job().animation_frame_encoder().unwrap();
    let err = encoder
        .push_frame_exact(img.as_slice(), Duration::new(1, 3).unwrap(), None)
        .unwrap_err();
    let erased: Box<dyn std::error::Error + Send + Sync> = Box::new(err);
    assert_eq!(
        erased.unsupported_operation(),
        Some(&UnsupportedOperation::AnimationTiming)
    );
    let native: Box<dyn std::error::Error + Send + Sync> = Box::new(
        zenwebp::decoder::DecodeError::from(UnsupportedOperation::DecodePolicy),
    );
    assert_eq!(
        native.unsupported_operation(),
        Some(&UnsupportedOperation::DecodePolicy)
    );
}
