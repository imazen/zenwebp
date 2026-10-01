//! Decoder parity against libwebp (via webpx) for paths the reference-PNG tests
//! do not cover: point-sampled ("nofancy") chroma and animation compositing.
//!
//! Every comparison is exact. The animation cases each pin one way zenwebp's
//! compositing used to diverge from libwebp's `WebPAnimDecoder`; the wide-corpus
//! sweep that found them is `dev/decode_parity_sweep.rs`.
#![forbid(unsafe_code)]
#![cfg(not(target_arch = "wasm32"))]

use zenwebp::decoder::UpsamplingMethod;
use zenwebp::mux::{
    AnimationConfig, AnimationDecoder, AnimationEncoder, BlendMethod, DisposeMethod,
};
use zenwebp::{
    DecodeConfig, DecodeRequest, EncodeRequest, EncoderConfig, LosslessConfig, LossyConfig,
    PixelLayout, WebPDecoder,
};

/// Deterministic textured RGBA: gradients plus hashed noise, alpha from `alpha`.
fn pattern(w: u32, h: u32, seed: u32, alpha: impl Fn(u32, u32) -> u8) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let n =
                (x.wrapping_mul(2_654_435_761) ^ y.wrapping_mul(40_503) ^ seed.wrapping_mul(97))
                    >> 9;
            v.extend_from_slice(&[
                ((x * 255 / w.max(1)) as u8).wrapping_add((n & 31) as u8),
                ((y * 255 / h.max(1)) as u8).wrapping_add((n >> 5 & 31) as u8),
                (n >> 10) as u8,
                alpha(x, y),
            ]);
        }
    }
    v
}

fn rgb(rgba: &[u8]) -> Vec<u8> {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect()
}

fn first_diff(a: &[u8], b: &[u8]) -> String {
    match a.iter().zip(b).position(|(x, y)| x != y) {
        Some(i) => format!("first diff at byte {i}: zen {} lib {}", a[i], b[i]),
        None => format!("lengths {} vs {}", a.len(), b.len()),
    }
}

// ---------------------------------------------------------------------------
// No-fancy upsampling
// ---------------------------------------------------------------------------

fn lib_nofancy_rgba(data: &[u8]) -> Vec<u8> {
    webpx::Decoder::new(data)
        .unwrap()
        .config(webpx::DecoderConfig::new().no_fancy_upsampling(true))
        .decode_rgba_raw()
        .unwrap()
        .0
}

#[test]
fn nofancy_matches_libwebp() {
    let simple = DecodeConfig::default().upsampling(UpsamplingMethod::Simple);
    let mut changed_something = false;
    for &(w, h) in &[(1u32, 1u32), (3, 5), (17, 13), (33, 31), (128, 96)] {
        // filter 0 takes the full-frame path, filter > 0 the streaming one.
        for &filter in &[0u8, 60] {
            for &with_alpha in &[false, true] {
                let px = pattern(w, h, w + h, |x, y| {
                    if with_alpha {
                        ((x * 7 + y * 3) % 256) as u8
                    } else {
                        255
                    }
                });
                let cfg = LossyConfig::new()
                    .with_quality(60.0)
                    .with_filter_strength(filter);
                let (input, layout) = if with_alpha {
                    (px.clone(), PixelLayout::Rgba8)
                } else {
                    (rgb(&px), PixelLayout::Rgb8)
                };
                let webp = EncodeRequest::lossy(&cfg, &input, layout, w, h)
                    .encode()
                    .unwrap();
                let label = format!("{w}x{h} filter{filter} alpha{with_alpha}");

                let lib = lib_nofancy_rgba(&webp);
                let (zen, ..) = DecodeRequest::new(&simple, &webp).decode_rgba().unwrap();
                assert!(
                    zen == lib,
                    "DecodeRequest rgba nofancy {label}: {}",
                    first_diff(&zen, &lib)
                );

                let (zen_rgb, ..) = DecodeRequest::new(&simple, &webp).decode_rgb().unwrap();
                assert_eq!(zen_rgb, rgb(&lib), "DecodeRequest rgb nofancy {label}");

                let mut dec = WebPDecoder::new(&webp).unwrap();
                dec.set_lossy_upsampling(UpsamplingMethod::Simple);
                let mut buf = vec![0u8; dec.output_buffer_size().unwrap()];
                dec.read_image(&mut buf).unwrap();
                let expect = if dec.has_alpha() {
                    lib.clone()
                } else {
                    rgb(&lib)
                };
                assert!(
                    buf == expect,
                    "WebPDecoder nofancy {label}: {}",
                    first_diff(&buf, &expect)
                );

                // Liveness: the option must change the output somewhere.
                let (fancy, ..) = DecodeRequest::new(&DecodeConfig::default(), &webp)
                    .decode_rgba()
                    .unwrap();
                changed_something |= fancy != zen;
            }
        }
    }
    assert!(
        changed_something,
        "UpsamplingMethod::Simple produced fancy output on every case"
    );
}

// ---------------------------------------------------------------------------
// Animation compositing
// ---------------------------------------------------------------------------

struct Frame<'a> {
    px: &'a [u8],
    w: u32,
    h: u32,
    x: u32,
    y: u32,
    has_alpha: bool,
    blend: BlendMethod,
    dispose: DisposeMethod,
    lossless: bool,
}

fn build_anim(cw: u32, ch: u32, frames: &[Frame]) -> Vec<u8> {
    let cfg = AnimationConfig {
        minimize_size: false,
        ..Default::default()
    };
    let mut enc = AnimationEncoder::new(cw, ch, cfg).unwrap();
    for (i, f) in frames.iter().enumerate() {
        let ecfg = if f.lossless {
            EncoderConfig::Lossless(LosslessConfig::new().with_exact(true))
        } else {
            EncoderConfig::Lossy(LossyConfig::new().with_quality(70.0))
        };
        let (pixels, layout) = if f.has_alpha {
            (f.px.to_vec(), PixelLayout::Rgba8)
        } else {
            (rgb(f.px), PixelLayout::Rgb8)
        };
        enc.add_frame_advanced(
            &pixels,
            layout,
            f.w,
            f.h,
            f.x,
            f.y,
            i as u32 * 100,
            &ecfg,
            f.dispose,
            f.blend,
        )
        .unwrap();
    }
    enc.finalize(100).unwrap()
}

/// Decode all frames with both decoders and require byte equality (RGBA).
fn assert_anim_matches(label: &str, data: &[u8]) {
    let lib = webpx::AnimationDecoder::new(data)
        .unwrap()
        .decode_all()
        .unwrap();
    let mut dec = AnimationDecoder::new(data).unwrap();
    let has_alpha = dec.info().has_alpha;
    let zen = dec.decode_all().unwrap();
    assert_eq!(zen.len(), lib.len(), "{label}: frame count");
    for (i, (z, l)) in zen.iter().zip(&lib).enumerate() {
        let z4 = if has_alpha {
            z.data.clone()
        } else {
            // Opaque-only animations decode to RGB; libwebp's alpha must then be 255.
            assert!(
                l.data.as_chunks::<4>().0.iter().all(|p| p[3] == 255),
                "{label} frame {i}: libwebp has transparency but zen reported no alpha"
            );
            z.data
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect()
        };
        assert!(
            z4 == l.data,
            "{label} frame {i}: {}",
            first_diff(&z4, &l.data)
        );
    }
}

/// libwebp copies a keyframe raw (no blend). Blending translucent pixels over
/// the zero-filled canvas instead loses up to ~50 levels at low alpha.
#[test]
fn anim_keyframe_after_full_background_dispose_is_not_blended() {
    let (w, h) = (48, 40);
    let opaque = pattern(w, h, 1, |_, _| 255);
    let translucent = pattern(w, h, 2, |x, y| ((x * 5 + y * 3) % 250 + 1) as u8);
    let data = build_anim(
        w,
        h,
        &[
            Frame {
                px: &opaque,
                w,
                h,
                x: 0,
                y: 0,
                has_alpha: false,
                blend: BlendMethod::Overwrite,
                dispose: DisposeMethod::Background,
                lossless: true,
            },
            Frame {
                px: &translucent,
                w,
                h,
                x: 0,
                y: 0,
                has_alpha: true,
                blend: BlendMethod::AlphaBlend,
                dispose: DisposeMethod::None,
                lossless: true,
            },
        ],
    );
    assert_anim_matches("keyframe-after-dispose", &data);
}

/// libwebp blends with `dst_factor_a = (dst_a * (256 - src_a)) >> 8`; a rounded
/// divide-by-255 differs by one level on translucent-over-translucent pixels.
#[test]
fn anim_translucent_over_translucent_uses_libwebp_blend_arithmetic() {
    let (w, h) = (40, 32);
    let a = pattern(w, h, 3, |x, _| (40 + x * 4) as u8);
    let b = pattern(w, h, 4, |_, y| (30 + y * 6) as u8);
    let data = build_anim(
        w,
        h,
        &[
            Frame {
                px: &a,
                w,
                h,
                x: 0,
                y: 0,
                has_alpha: true,
                blend: BlendMethod::Overwrite,
                dispose: DisposeMethod::None,
                lossless: true,
            },
            Frame {
                px: &b,
                w,
                h,
                x: 0,
                y: 0,
                has_alpha: true,
                blend: BlendMethod::AlphaBlend,
                dispose: DisposeMethod::None,
                lossless: true,
            },
        ],
    );
    assert_anim_matches("translucent-blend", &data);
}

/// Frame 1 is a keyframe: its RGB under alpha=0 survives in libwebp's output.
#[test]
fn anim_first_frame_keeps_rgb_under_zero_alpha() {
    let (w, h) = (32, 24);
    let px = pattern(
        w,
        h,
        5,
        |x, y| if (x / 4 + y / 4) % 2 == 0 { 0 } else { 255 },
    );
    let data = build_anim(
        w,
        h,
        &[
            Frame {
                px: &px,
                w,
                h,
                x: 0,
                y: 0,
                has_alpha: true,
                blend: BlendMethod::AlphaBlend,
                dispose: DisposeMethod::None,
                lossless: true,
            },
            Frame {
                px: &px,
                w: 16,
                h: 16,
                x: 8,
                y: 4,
                has_alpha: true,
                blend: BlendMethod::AlphaBlend,
                dispose: DisposeMethod::None,
                lossless: true,
            },
        ],
    );
    assert_anim_matches("rgb-under-zero-alpha", &data);
}

/// No frame carries alpha, but a background dispose leaves a transparent hole.
/// zen used to report the animation as RGB and turn the hole opaque black.
#[test]
fn anim_background_dispose_hole_without_frame_alpha_is_transparent() {
    let (w, h) = (48, 40);
    let full = pattern(w, h, 6, |_, _| 255);
    let sub = pattern(24, 20, 7, |_, _| 255);
    let data = build_anim(
        w,
        h,
        &[
            Frame {
                px: &full,
                w,
                h,
                x: 0,
                y: 0,
                has_alpha: false,
                blend: BlendMethod::Overwrite,
                dispose: DisposeMethod::None,
                lossless: false,
            },
            Frame {
                px: &sub,
                w: 24,
                h: 20,
                x: 10,
                y: 10,
                has_alpha: false,
                blend: BlendMethod::AlphaBlend,
                dispose: DisposeMethod::Background,
                lossless: false,
            },
            Frame {
                px: &sub,
                w: 24,
                h: 20,
                x: 4,
                y: 2,
                has_alpha: false,
                blend: BlendMethod::Overwrite,
                dispose: DisposeMethod::None,
                lossless: false,
            },
        ],
    );
    let dec = AnimationDecoder::new(&data).unwrap();
    assert!(
        dec.info().has_alpha,
        "a background-dispose hole makes the canvas transparent"
    );
    assert_anim_matches("dispose-hole", &data);
}

/// Disposing the previous rect must happen even when the CURRENT frame has no
/// alpha (zen used to skip the clear in that case and keep stale pixels).
#[test]
fn anim_dispose_applies_before_opaque_frame() {
    let (w, h) = (48, 40);
    let full = pattern(w, h, 8, |_, _| 255);
    let big = pattern(32, 32, 9, |_, _| 200);
    let small = pattern(8, 8, 10, |_, _| 255);
    let data = build_anim(
        w,
        h,
        &[
            Frame {
                px: &full,
                w,
                h,
                x: 0,
                y: 0,
                has_alpha: false,
                blend: BlendMethod::Overwrite,
                dispose: DisposeMethod::None,
                lossless: true,
            },
            Frame {
                px: &big,
                w: 32,
                h: 32,
                x: 4,
                y: 4,
                has_alpha: true,
                blend: BlendMethod::AlphaBlend,
                dispose: DisposeMethod::Background,
                lossless: true,
            },
            Frame {
                px: &small,
                w: 8,
                h: 8,
                x: 6,
                y: 6,
                has_alpha: false,
                blend: BlendMethod::AlphaBlend,
                dispose: DisposeMethod::None,
                lossless: false,
            },
        ],
    );
    assert_anim_matches("dispose-before-opaque", &data);
}

/// Translucent sub-frame inside the previous (disposed) rect: libwebp keeps
/// the raw pixel there (blend against transparent is a no-op) and blends
/// against the canvas everywhere else.
#[test]
fn anim_blend_inside_disposed_rect_is_raw() {
    let (w, h) = (48, 40);
    let full = pattern(w, h, 11, |_, _| 255);
    let mid = pattern(24, 24, 12, |_, _| 255);
    let tr = pattern(32, 28, 13, |x, y| ((x * 9 + y * 5) % 254 + 1) as u8);
    let data = build_anim(
        w,
        h,
        &[
            Frame {
                px: &full,
                w,
                h,
                x: 0,
                y: 0,
                has_alpha: false,
                blend: BlendMethod::Overwrite,
                dispose: DisposeMethod::None,
                lossless: true,
            },
            Frame {
                px: &mid,
                w: 24,
                h: 24,
                x: 8,
                y: 8,
                has_alpha: false,
                blend: BlendMethod::AlphaBlend,
                dispose: DisposeMethod::Background,
                lossless: true,
            },
            Frame {
                px: &tr,
                w: 32,
                h: 28,
                x: 2,
                y: 4,
                has_alpha: true,
                blend: BlendMethod::AlphaBlend,
                dispose: DisposeMethod::None,
                lossless: true,
            },
        ],
    );
    assert_anim_matches("blend-in-disposed-rect", &data);
}
