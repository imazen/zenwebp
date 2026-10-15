//! Float-input descriptor advertisement + roundtrip.
//!
//! `pixels_to_webp_input` already converts `RGBF32_LINEAR`,
//! `RGBAF32_LINEAR`, and `GRAYF32_LINEAR` to the encoder's expected u8
//! sRGB layouts (see `src/codec.rs`), but those descriptors weren't in
//! `ENCODE_DESCRIPTORS`, so callers querying capabilities couldn't
//! discover that the encoder accepts them. This test ensures:
//!
//! 1. The descriptors are advertised in `supported_descriptors()`.
//! 2. End-to-end encode + decode through the zencodec adapter works for
//!    each float descriptor in both lossy and lossless modes.
//! 3. Decoded sRGB output matches the linear-to-sRGB conversion of the
//!    input (within lossless tolerance for lossless / lossy quant
//!    tolerance for lossy).

#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use zencodec::encode::{EncodeJob, Encoder, EncoderConfig as _};
use zenpixels::{PixelDescriptor, PixelSlice};
use zenwebp::zencodec::WebpEncoderConfig;

const W: u32 = 64;
const H: u32 = 64;

fn linear_gradient_rgb_f32(channels: usize) -> Vec<f32> {
    // Linear gradient in scene-referred linear-light. Values 0.0..1.0 across
    // the diagonal so the linear→sRGB conversion exercises the curve's mid
    // and bright regions (where rounding bites hardest).
    let mut out = Vec::with_capacity((W * H) as usize * channels);
    for y in 0..H {
        for x in 0..W {
            let r = (x as f32) / (W - 1) as f32;
            let g = (y as f32) / (H - 1) as f32;
            let b = ((x + y) as f32) / (W + H - 2) as f32;
            out.push(r);
            if channels >= 2 {
                out.push(g);
            }
            if channels >= 3 {
                out.push(b);
            }
            if channels == 4 {
                out.push(1.0); // alpha
            }
        }
    }
    out
}

fn gray_gradient_f32() -> Vec<f32> {
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            // Diagonal linear gradient
            let g = ((x + y) as f32) / (W + H - 2) as f32;
            out.push(g);
        }
    }
    out
}

fn floats_as_bytes(f: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(f.len() * 4);
    for v in f {
        out.extend_from_slice(&v.to_ne_bytes());
    }
    out
}

#[test]
fn supported_descriptors_includes_floats() {
    use zencodec::encode::EncoderConfig;
    let descs: &[PixelDescriptor] = WebpEncoderConfig::supported_descriptors();
    assert!(
        descs.contains(&PixelDescriptor::RGBF32_LINEAR),
        "RGBF32_LINEAR not advertised; descriptors = {descs:?}"
    );
    assert!(
        descs.contains(&PixelDescriptor::RGBAF32_LINEAR),
        "RGBAF32_LINEAR not advertised; descriptors = {descs:?}"
    );
    assert!(
        descs.contains(&PixelDescriptor::GRAYF32_LINEAR,),
        "GRAYF32_LINEAR not advertised; descriptors = {descs:?}"
    );
}

fn encode_roundtrip(pixels_bytes: &[u8], descriptor: PixelDescriptor, lossless: bool) -> Vec<u8> {
    let cfg = if lossless {
        WebpEncoderConfig::lossless().with_exact(true)
    } else {
        WebpEncoderConfig::lossy().with_quality(85.0)
    };
    let stride_bytes = W as usize * descriptor.bytes_per_pixel();
    let slice =
        PixelSlice::new(pixels_bytes, W, H, stride_bytes, descriptor).expect("PixelSlice::new");
    let bytes = cfg
        .job()
        .encoder()
        .expect("encoder")
        .encode(slice)
        .expect("encode")
        .data()
        .to_vec();
    let (decoded, dw, dh) = zenwebp::oneshot::decode_rgba(&bytes).expect("decode");
    assert_eq!((dw, dh), (W, H));
    decoded
}

/// Compare against the same scalar linear-to-sRGB conversion the encoder
/// uses internally, so a round-trip through lossless gives byte-exact
/// pixels (modulo decoder upsampling for non-RGB descriptors).
fn linear_f32_to_srgb_u8(v: f32) -> u8 {
    let mut tmp = [0u8; 1];
    linear_srgb::default::linear_to_srgb_u8_slice(&[v], &mut tmp);
    tmp[0]
}

#[test]
fn rgbf32_lossless_roundtrip() {
    let f = linear_gradient_rgb_f32(3);
    let bytes = floats_as_bytes(&f);
    let decoded = encode_roundtrip(&bytes, PixelDescriptor::RGBF32_LINEAR, true);

    for (i, px) in decoded.as_chunks::<4>().0.iter().enumerate() {
        let r_lin = f[i * 3];
        let g_lin = f[i * 3 + 1];
        let b_lin = f[i * 3 + 2];
        assert_eq!(
            (px[0], px[1], px[2], px[3]),
            (
                linear_f32_to_srgb_u8(r_lin),
                linear_f32_to_srgb_u8(g_lin),
                linear_f32_to_srgb_u8(b_lin),
                255
            ),
            "RGBF32_LINEAR lossless pixel {i} mismatch (lin={r_lin:.3},{g_lin:.3},{b_lin:.3})",
        );
    }
}

#[test]
fn rgbaf32_lossless_roundtrip() {
    let f = linear_gradient_rgb_f32(4);
    let bytes = floats_as_bytes(&f);
    let decoded = encode_roundtrip(&bytes, PixelDescriptor::RGBAF32_LINEAR, true);

    for (i, px) in decoded.as_chunks::<4>().0.iter().enumerate() {
        let r_lin = f[i * 4];
        let g_lin = f[i * 4 + 1];
        let b_lin = f[i * 4 + 2];
        let a_lin = f[i * 4 + 3];
        let expected_alpha = (a_lin * 255.0).round().clamp(0.0, 255.0) as u8;
        assert_eq!(
            (px[0], px[1], px[2]),
            (
                linear_f32_to_srgb_u8(r_lin),
                linear_f32_to_srgb_u8(g_lin),
                linear_f32_to_srgb_u8(b_lin),
            ),
            "RGBAF32_LINEAR RGB at pixel {i}: lin={r_lin:.3},{g_lin:.3},{b_lin:.3}",
        );
        // Alpha goes through the linear-to-sRGB-rgba slice converter which
        // preserves alpha as a straight 8-bit value.
        assert_eq!(px[3], expected_alpha, "alpha at pixel {i}");
    }
}

#[test]
fn grayf32_lossless_roundtrip() {
    let f = gray_gradient_f32();
    let bytes = floats_as_bytes(&f);
    let decoded = encode_roundtrip(&bytes, PixelDescriptor::GRAYF32_LINEAR, true);

    for (i, px) in decoded.as_chunks::<4>().0.iter().enumerate() {
        let g_lin = f[i];
        let g_srgb = linear_f32_to_srgb_u8(g_lin);
        assert_eq!(
            (px[0], px[1], px[2], px[3]),
            (g_srgb, g_srgb, g_srgb, 255),
            "GRAYF32_LINEAR lossless pixel {i} mismatch (lin={g_lin:.3})",
        );
    }
}

#[test]
fn rgbf32_lossy_roundtrip_within_tolerance() {
    let f = linear_gradient_rgb_f32(3);
    let bytes = floats_as_bytes(&f);
    let decoded = encode_roundtrip(&bytes, PixelDescriptor::RGBF32_LINEAR, false);

    let mut max_rgb = 0i32;
    for (i, px) in decoded.as_chunks::<4>().0.iter().enumerate() {
        let r_exp = linear_f32_to_srgb_u8(f[i * 3]);
        let g_exp = linear_f32_to_srgb_u8(f[i * 3 + 1]);
        let b_exp = linear_f32_to_srgb_u8(f[i * 3 + 2]);
        for (a, b) in [px[0], px[1], px[2]].iter().zip([r_exp, g_exp, b_exp]) {
            max_rgb = max_rgb.max((*a as i32 - b as i32).abs());
        }
        assert_eq!(px[3], 255, "alpha must be opaque");
    }
    assert!(
        max_rgb <= 24,
        "RGBF32_LINEAR lossy q85 RGB delta exceeded tolerance: {max_rgb}"
    );
}

#[test]
fn rgbaf32_lossy_roundtrip_within_tolerance() {
    let f = linear_gradient_rgb_f32(4);
    let bytes = floats_as_bytes(&f);
    let decoded = encode_roundtrip(&bytes, PixelDescriptor::RGBAF32_LINEAR, false);

    let mut max_rgb = 0i32;
    let mut max_alpha = 0i32;
    for (i, px) in decoded.as_chunks::<4>().0.iter().enumerate() {
        let r_exp = linear_f32_to_srgb_u8(f[i * 4]);
        let g_exp = linear_f32_to_srgb_u8(f[i * 4 + 1]);
        let b_exp = linear_f32_to_srgb_u8(f[i * 4 + 2]);
        let a_exp = (f[i * 4 + 3] * 255.0).round().clamp(0.0, 255.0) as u8;
        for (a, b) in [px[0], px[1], px[2]].iter().zip([r_exp, g_exp, b_exp]) {
            max_rgb = max_rgb.max((*a as i32 - b as i32).abs());
        }
        max_alpha = max_alpha.max((px[3] as i32 - a_exp as i32).abs());
    }
    assert!(
        max_rgb <= 24,
        "RGBAF32_LINEAR lossy q85 RGB delta exceeded tolerance: {max_rgb}"
    );
    assert!(
        max_alpha <= 4,
        "RGBAF32_LINEAR lossy q85 alpha delta exceeded tolerance: {max_alpha}"
    );
}

#[test]
fn grayf32_lossy_roundtrip_within_tolerance() {
    let f = gray_gradient_f32();
    let bytes = floats_as_bytes(&f);
    let decoded = encode_roundtrip(&bytes, PixelDescriptor::GRAYF32_LINEAR, false);

    let mut max_split = 0i32;
    let mut max_y = 0i32;
    for (i, px) in decoded.as_chunks::<4>().0.iter().enumerate() {
        let g_exp = linear_f32_to_srgb_u8(f[i]);
        let split = (px[0] as i32 - px[1] as i32)
            .abs()
            .max((px[1] as i32 - px[2] as i32).abs());
        max_split = max_split.max(split);
        max_y = max_y.max((px[0] as i32 - g_exp as i32).abs());
        assert_eq!(px[3], 255, "alpha must be opaque");
    }
    assert!(
        max_split <= 2,
        "GRAYF32_LINEAR decoded must stay grayscale (R≈G≈B): max split = {max_split}"
    );
    assert!(
        max_y <= 24,
        "GRAYF32_LINEAR lossy q85 luma delta exceeded tolerance: {max_y}"
    );
}

/// f32 input is stored as 8-bit sRGB, so a CICP calling it linear must not
/// turn into a linear ICC profile next to sRGB samples: the profile must say
/// sRGB, with the CICP's primaries.
#[test]
fn a_linear_cicp_on_f32_input_is_written_as_srgb() {
    use std::borrow::Cow;
    use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
    let floats = linear_gradient_rgb_f32(3);
    let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_ne_bytes()).collect();
    let slice = PixelSlice::new(
        &bytes,
        W,
        H,
        W as usize * 12,
        PixelDescriptor::RGBF32_LINEAR,
    )
    .unwrap();
    let out = WebpEncoderConfig::lossless()
        .job()
        .with_metadata_policy(
            zencodec::Metadata::none().with_cicp(zencodec::Cicp::new(12, 8, 0, true)),
            zencodec::MetadataPolicy::PreserveExact,
        )
        .encoder()
        .unwrap()
        .encode(slice)
        .unwrap();
    let decoded = zenwebp::zencodec::WebpDecoderConfig::new()
        .job()
        .decoder(Cow::Borrowed(out.data()), &[])
        .unwrap()
        .decode()
        .unwrap();
    let icc = decoded
        .info()
        .source_color
        .icc_profile
        .clone()
        .expect("a P3 source needs an ICC in WebP");
    let cicp = zenpixels::icc::identify_common(&icc)
        .and_then(|id| id.to_cicp())
        .expect("a known profile");
    assert_eq!(
        (cicp.color_primaries, cicp.transfer_characteristics),
        (12, 13)
    );
}
