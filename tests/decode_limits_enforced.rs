//! `DecodeConfig::limits(...)` must actually enforce dimension / pixel /
//! frame caps.
//!
//! Regression for a limits-ordering bug found in the 2026-08-21 review:
//! `WebPDecoder::new_with_options` ran `read_data()` (which runs the
//! container `check_dimensions` / `check_frame_count` gates) against the
//! DEFAULT limits, and the caller's `set_limits` was applied afterward — so
//! every caller-supplied dimension/pixel/frame limit was dead on every entry
//! point, and the crate's advertised caps held only at their defaults. The
//! fix threads the limits through the constructor. `max_memory` was already
//! enforced (checked during decode, after limits are set) and is not retested
//! here.

use zenwebp::{DecodeConfig, DecodeRequest, EncodeRequest, EncoderConfig, Limits, PixelLayout};

fn encode(w: u32, h: u32) -> Vec<u8> {
    let rgba = vec![128u8; (w * h * 4) as usize];
    let config = EncoderConfig::new_lossy().with_quality(50.0);
    EncodeRequest::new(&config, &rgba, PixelLayout::Rgba8, w, h)
        .encode()
        .unwrap()
}

#[test]
fn max_dimensions_rejects_oversized_image() {
    let webp = encode(200, 200);

    // A generous default config decodes it fine.
    let ok = DecodeRequest::new(&DecodeConfig::default(), &webp).decode_rgba();
    assert!(ok.is_ok(), "200x200 must decode under default limits");

    // A 100x100 cap must REJECT the 200x200 image — before the fix this
    // silently decoded because the cap arrived after the header check.
    let tight = DecodeConfig::default().limits(Limits::default().max_dimensions(100, 100));
    let err = DecodeRequest::new(&tight, &webp).decode_rgba();
    assert!(
        err.is_err(),
        "max_dimensions(100,100) must reject a 200x200 image"
    );
}

#[test]
fn max_total_pixels_rejects_oversized_image() {
    let webp = encode(200, 200); // 40_000 px

    let tight = DecodeConfig::default().limits(Limits::default().max_total_pixels(10_000));
    let err = DecodeRequest::new(&tight, &webp).decode_rgba();
    assert!(
        err.is_err(),
        "max_total_pixels(10_000) must reject a 40_000-pixel image"
    );

    // Exactly at the limit still decodes.
    let webp_small = encode(100, 100); // 10_000 px
    let at_limit = DecodeConfig::default().limits(Limits::default().max_total_pixels(10_000));
    assert!(
        DecodeRequest::new(&at_limit, &webp_small)
            .decode_rgba()
            .is_ok(),
        "10_000-pixel image must decode under a 10_000-pixel cap"
    );
}

#[test]
fn max_dimensions_enforced_on_decode_rgb_and_into() {
    let webp = encode(200, 200);
    let tight = DecodeConfig::default().limits(Limits::default().max_dimensions(100, 100));

    // Every native entry point must honor the cap, not just decode_rgba.
    assert!(
        DecodeRequest::new(&tight, &webp).decode_rgb().is_err(),
        "decode_rgb must honor max_dimensions"
    );
    let mut buf = vec![0u8; 200 * 200 * 4];
    assert!(
        DecodeRequest::new(&tight, &webp)
            .decode_rgba_into(&mut buf)
            .is_err(),
        "decode_rgba_into must honor max_dimensions"
    );
}

/// Build an N-frame animation (tiny lossless frames) for the frame-count gate.
fn animation_with_frames(n: u32) -> Vec<u8> {
    use zenwebp::mux::{AnimationConfig, AnimationEncoder};
    let mut anim = AnimationEncoder::new(8, 8, AnimationConfig::default()).unwrap();
    let cfg = EncoderConfig::new_lossless();
    for i in 0..n {
        let rgba = vec![(i * 40) as u8; 8 * 8 * 4];
        anim.add_frame(&rgba, PixelLayout::Rgba8, i * 100, &cfg)
            .unwrap();
    }
    anim.finalize(100).unwrap()
}

/// `Limits::max_frame_count(n)` is documented as a MAXIMUM ("Maximum number
/// of frames in an animation"). The gate ran `count >= max` on the
/// post-increment count, so it rejected the n-th frame: `max_frame_count(3)`
/// refused every 3-frame animation, and the default 10,000 cap admitted
/// 9,999. Exactly `max` frames must be admitted; `max + 1` rejected.
#[test]
fn max_frame_count_admits_exactly_the_limit() {
    use zenwebp::mux::AnimationDecoder;
    let three = animation_with_frames(3);

    let at_limit = DecodeConfig::default().limits(Limits::default().max_frame_count(3));
    let mut dec = AnimationDecoder::new_with_config(&three, &at_limit)
        .expect("max_frame_count(3) must admit a 3-frame animation");
    let frames = dec
        .decode_all()
        .expect("all 3 frames decode under the limit");
    assert_eq!(frames.len(), 3);

    let under = DecodeConfig::default().limits(Limits::default().max_frame_count(2));
    assert!(
        AnimationDecoder::new_with_config(&three, &under).is_err(),
        "max_frame_count(2) must reject a 3-frame animation"
    );
}
