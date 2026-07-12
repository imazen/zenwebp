//! EOF / truncation conformance for the zencodec dyn-erased decode path.
//!
//! Feeds a known-good WebP through `zencodec_testkit::check_decode_truncation_series`,
//! which truncates it at a deterministic series of byte prefixes, decodes each
//! through the dyn-erased boundary, and asserts every failure categorizes as
//! incomplete-input (never a panic, OOM, or `Internal`).

use zencodec::encode::{EncodeJob, Encoder, EncoderConfig as _};
use zenpixels::{PixelDescriptor, PixelSlice};
use zenwebp::zencodec::{WebpDecoderConfig, WebpEncoderConfig};

/// Produce a valid WebP by encoding a tiny RGB image through the zencodec trait —
/// the same encode path a passing repo test (`tests/gray8_input.rs`) exercises.
fn valid_webp() -> Vec<u8> {
    let (w, h) = (8u32, 8u32);
    let bytes = vec![0x77u8; (w * h * 3) as usize];
    let slice = PixelSlice::new(&bytes, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB)
        .expect("rgb8 slice");
    WebpEncoderConfig::lossless()
        .job()
        .encoder()
        .expect("encoder")
        .encode(slice)
        .expect("encode")
        .data()
        .to_vec()
}

#[test]
fn truncation_series_categorizes_as_incomplete_input() {
    let valid = valid_webp();
    zencodec_testkit::check_decode_truncation_series(WebpDecoderConfig::new(), &valid)
        .expect("truncated input must categorize as incomplete, never panic/OOM/Internal");
}
