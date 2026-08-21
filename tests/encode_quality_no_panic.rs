//! Out-of-range quality must never panic the encoder.
//!
//! Regression for a review finding (2026-08-21): `LossyConfig::with_preset`
//! and the `EncoderConfig` enum `with_quality` assigned `quality` raw (unlike
//! `LossyConfig::with_quality`, which clamps), and the config→params
//! conversion did `roundf(quality) as u8` with no clamp — so a quality > 100
//! reached `panic!("lossy quality must be between 0 and 100")` in the encoder,
//! and a quality > 255 wrapped `as u8` to a bogus low value. All three
//! conversion boundaries now clamp to [0, 100].

use zenwebp::{EncodeRequest, EncoderConfig, LossyConfig, PixelLayout, Preset};

fn img() -> Vec<u8> {
    vec![128u8; 16 * 16 * 4]
}

fn encodes_ok(cfg: &EncoderConfig) -> bool {
    EncodeRequest::new(cfg, &img(), PixelLayout::Rgba8, 16, 16)
        .encode()
        .is_ok()
}

#[test]
fn out_of_range_quality_clamps_not_panics() {
    // Every public way to set an out-of-range quality must clamp, not panic.
    for q in [150.0f32, 300.0, 1000.0, -50.0, f32::INFINITY, f32::NAN] {
        assert!(
            encodes_ok(&EncoderConfig::Lossy(LossyConfig::with_preset(
                Preset::Photo,
                q
            ))),
            "with_preset quality={q} panicked or failed"
        );
        assert!(
            encodes_ok(&EncoderConfig::new_lossy().with_quality(q)),
            "enum with_quality quality={q} panicked or failed"
        );
        assert!(
            encodes_ok(&EncoderConfig::Lossy(LossyConfig::new().with_quality(q))),
            "LossyConfig::with_quality quality={q} panicked or failed"
        );
    }
}

/// A clamped over-range quality must behave like q=100 (best), not like a
/// wrapped-`as u8` low quality — i.e. it must not silently produce a
/// tiny/low-quality file. Compare byte size against an explicit q=100.
#[test]
fn over_range_quality_behaves_like_max() {
    // Use gradient content so quality actually affects size.
    let (w, h) = (64usize, 64usize);
    let mut rgba = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            rgba.extend_from_slice(&[(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8, 255]);
        }
    }
    let enc = |q: f32| {
        EncodeRequest::new(
            &EncoderConfig::new_lossy().with_quality(q),
            &rgba,
            PixelLayout::Rgba8,
            w as u32,
            h as u32,
        )
        .encode()
        .unwrap()
        .len()
    };
    let at_100 = enc(100.0);
    let over = enc(500.0);
    assert_eq!(
        over, at_100,
        "quality=500 must clamp to 100 ({at_100} B), not wrap to a low quality ({over} B)"
    );
}
