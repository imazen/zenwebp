//! Encode via the BGRX8_SRGB / RGBX8_SRGB descriptors.
//!
//! These are 4-byte layouts where byte 3 is undefined padding — the encoder
//! must take the RGB fast path and ignore the padding byte. Confirms that
//! descriptor dispatch accepts these formats and that the padding byte does
//! not leak into encoded output.
#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

use zencodec::encode::{DynEncoderConfig, EncoderConfig};
use zenpixels::{PixelDescriptor, PixelSlice};

fn solid_pattern_4bpp(w: u32, h: u32, rgb_order: bool) -> Vec<u8> {
    // rgb_order=true → [R, G, B, X]; false → [B, G, R, X]
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = (x as u8).wrapping_mul(17);
            let g = (y as u8).wrapping_mul(23);
            let b = ((x ^ y) as u8).wrapping_mul(31);
            // padding byte deliberately non-0xFF to catch any leak into alpha
            let pad: u8 = 0x42;
            if rgb_order {
                out.extend_from_slice(&[r, g, b, pad]);
            } else {
                out.extend_from_slice(&[b, g, r, pad]);
            }
        }
    }
    out
}

#[test]
fn supported_descriptors_includes_rgbx_and_bgrx() {
    let desc =
        <zenwebp::zencodec::WebpEncoderConfig as EncoderConfig>::supported_descriptors();
    assert!(
        desc.contains(&PixelDescriptor::RGBX8_SRGB),
        "RGBX8_SRGB must be in supported_descriptors"
    );
    assert!(
        desc.contains(&PixelDescriptor::BGRX8_SRGB),
        "BGRX8_SRGB must be in supported_descriptors"
    );
}

#[test]
fn encode_rgbx_lossless_roundtrip() {
    let w = 16u32;
    let h = 16u32;
    let rgbx = solid_pattern_4bpp(w, h, true);

    let config = zenwebp::zencodec::WebpEncoderConfig::lossless();
    let job = config.dyn_job();
    let encoder = job.into_encoder().expect("into_encoder");

    let ps = PixelSlice::new(&rgbx, w, h, (w * 4) as usize, PixelDescriptor::RGBX8_SRGB)
        .expect("pixel slice");
    let output = encoder.encode(ps).expect("encode");
    assert!(!output.data().is_empty());

    let (decoded, dw, dh) = zenwebp::oneshot::decode_rgba(output.data()).expect("decode");
    assert_eq!((dw, dh), (w, h));

    // Padding byte must not leak into decoded alpha; RGB must match.
    for i in 0..(w * h) as usize {
        let src = &rgbx[i * 4..i * 4 + 4]; // R, G, B, pad
        let dst = &decoded[i * 4..i * 4 + 4]; // R, G, B, A
        assert_eq!(
            &dst[..3],
            &src[..3],
            "RGB mismatch at pixel {i}: expected {:?} got {:?}",
            &src[..3],
            &dst[..3]
        );
        assert_eq!(dst[3], 255, "decoded alpha must be opaque at pixel {i}");
    }
}

#[test]
fn encode_bgrx_lossless_roundtrip() {
    let w = 16u32;
    let h = 16u32;
    let bgrx = solid_pattern_4bpp(w, h, false);

    let config = zenwebp::zencodec::WebpEncoderConfig::lossless();
    let job = config.dyn_job();
    let encoder = job.into_encoder().expect("into_encoder");

    let ps = PixelSlice::new(&bgrx, w, h, (w * 4) as usize, PixelDescriptor::BGRX8_SRGB)
        .expect("pixel slice");
    let output = encoder.encode(ps).expect("encode");
    assert!(!output.data().is_empty());

    let (decoded, dw, dh) = zenwebp::oneshot::decode_rgba(output.data()).expect("decode");
    assert_eq!((dw, dh), (w, h));

    for i in 0..(w * h) as usize {
        let src = &bgrx[i * 4..i * 4 + 4]; // B, G, R, pad
        let dst = &decoded[i * 4..i * 4 + 4]; // R, G, B, A
        let expect_rgb = [src[2], src[1], src[0]];
        assert_eq!(
            &dst[..3],
            &expect_rgb[..],
            "RGB mismatch at pixel {i}: expected {:?} got {:?}",
            &expect_rgb,
            &dst[..3]
        );
        assert_eq!(dst[3], 255, "decoded alpha must be opaque at pixel {i}");
    }
}
