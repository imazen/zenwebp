//! Unit and integration tests for `WebpIncrementalDecoder` (Sans-I/O decoding).

use std::path::{Path, PathBuf};
use zencodec::decode::{
    DecodeJob, DynIncrementalDecoder, IncrementalDecode, PushOutcome, PushStatus,
};
use zenpixels::PixelDescriptor;
use zenwebp::decoder::{DecodeConfig, WebpIncrementalDecoder};
use zenwebp::zencodec::{WebpDecoderConfig, WebpEncoderConfig};
use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/images")
}

fn resource(rel: &str) -> PathBuf {
    fixture_root().join(rel)
}

fn generate_lossless_webp(w: u32, h: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            rgba.extend_from_slice(&[
                (x * 17 % 256) as u8,
                (y * 31 % 256) as u8,
                ((x ^ y) * 13 % 256) as u8,
                255,
            ]);
        }
    }
    let slice = zenpixels::PixelSlice::new(&rgba, w, h, (w * 4) as usize, PixelDescriptor::RGBA8_SRGB)
        .expect("slice");
    WebpEncoderConfig::lossless()
        .job()
        .encoder()
        .expect("encoder")
        .encode(slice)
        .expect("encode")
        .data()
        .to_vec()
}

fn decode_incremental_chunks(
    data: &[u8],
    chunk_size: usize,
    batch_rows: u32,
    preferred: &[PixelDescriptor],
) -> (u32, u32, PixelDescriptor, Vec<u8>) {
    let mut decoder = WebpIncrementalDecoder::new(
        DecodeConfig::default(),
        None,
        None,
        None,
        preferred,
        zencodec::OrientationHint::Preserve,
    )
    .with_batch_rows(batch_rows);

    let mut offset = 0;
    let mut collected_pixels = Vec::new();
    let mut total_rows = 0;
    let mut header_seen = false;

    while offset < data.len() {
        let end = (offset + chunk_size).min(data.len());
        let chunk = &data[offset..end];
        let is_eof = end == data.len();

        let outcome = decoder
            .push_chunk(chunk, is_eof, None)
            .expect("push_chunk should succeed");

        offset += outcome.bytes_consumed;

        if let Some(info) = decoder.info() {
            if !header_seen {
                assert!(info.width > 0 && info.height > 0);
                header_seen = true;
            }
        }

        while let Some((y, slice)) = decoder.pull_batch().expect("pull_batch should succeed") {
            assert_eq!(y, total_rows, "scanline row index must match accumulated rows");
            collected_pixels.extend_from_slice(slice.as_strided_bytes());
            total_rows += slice.rows();
        }

        if outcome.status == PushStatus::Complete {
            break;
        }
    }

    assert!(decoder.is_complete(), "decoder must report complete");
    assert_eq!(total_rows, decoder.height());

    let desc = decoder.descriptor().expect("descriptor must be populated");
    (decoder.width(), decoder.height(), desc, collected_pixels)
}

#[test]
fn test_incremental_lossy_and_lossless_chunks() {
    let synthetic_lossless = generate_lossless_webp(64, 48);

    let test_fixtures: Vec<(&str, Vec<u8>)> = vec![
        ("gallery1/1.webp", std::fs::read(resource("gallery1/1.webp")).expect("read 1.webp")),
        ("gallery1/2.webp", std::fs::read(resource("gallery1/2.webp")).expect("read 2.webp")),
        ("regression/tiny.webp", std::fs::read(resource("regression/tiny.webp")).expect("read tiny.webp")),
        ("synthetic_lossless.webp", synthetic_lossless),
    ];

    let chunk_sizes = [1, 17, 64, 4096, 65536, usize::MAX];
    let batch_row_configs = [1, 16, 32, 0]; // 0 means full image batch

    for (name, data) in test_fixtures {
        let (ref_pixels, rw, rh) = zenwebp::oneshot::decode_rgba(&data).expect("oneshot decode");

        for &chunk_size in &chunk_sizes {
            for &batch_rows in &batch_row_configs {
                let chunk_sz = if chunk_size == usize::MAX {
                    data.len()
                } else {
                    chunk_size
                };

                let preferred = [PixelDescriptor::RGBA8_SRGB];
                let (w, h, desc, pixels) = decode_incremental_chunks(&data, chunk_sz, batch_rows, &preferred);

                assert_eq!(w, rw, "{name}: width mismatch");
                assert_eq!(h, rh, "{name}: height mismatch");
                assert_eq!(desc, PixelDescriptor::RGBA8_SRGB);
                assert_eq!(pixels.len(), ref_pixels.len(), "{name}: byte len mismatch");
                assert_eq!(pixels, ref_pixels, "{name}: decoded pixel data mismatch at chunk_sz={chunk_sz}, batch_rows={batch_rows}");
            }
        }
    }
}

#[test]
fn test_bgra_negotiation() {
    let data = std::fs::read(resource("gallery1/1.webp")).expect("read 1.webp");
    let (ref_rgba, rw, rh) = zenwebp::oneshot::decode_rgba(&data).expect("oneshot decode");

    let preferred = [PixelDescriptor::BGRA8_SRGB];
    let (w, h, desc, bgra_pixels) = decode_incremental_chunks(&data, 256, 16, &preferred);

    assert_eq!(w, rw);
    assert_eq!(h, rh);
    assert_eq!(desc, PixelDescriptor::BGRA8_SRGB);
    assert_eq!(bgra_pixels.len(), ref_rgba.len());

    // Verify RGBA vs BGRA byte swap
    for (rgba, bgra) in ref_rgba.chunks_exact(4).zip(bgra_pixels.chunks_exact(4)) {
        assert_eq!(rgba[0], bgra[2]); // R
        assert_eq!(rgba[1], bgra[1]); // G
        assert_eq!(rgba[2], bgra[0]); // B
        assert_eq!(rgba[3], bgra[3]); // A
    }
}

#[test]
fn test_unexpected_eof() {
    let data = std::fs::read(resource("gallery1/1.webp")).expect("read 1.webp");
    let truncation_points = [4, 11, 20, 50, data.len() / 2, data.len() - 10];

    for &cut in &truncation_points {
        let truncated = &data[..cut];
        let mut decoder = WebpIncrementalDecoder::new(
            DecodeConfig::default(),
            None,
            None,
            None,
            &[PixelDescriptor::RGBA8_SRGB],
            zencodec::OrientationHint::Preserve,
        );

        let res = decoder.push_chunk(truncated, true, None);
        assert!(res.is_err(), "truncated input with is_eof=true must return Err at cut={cut}");
        let err = res.unwrap_err();
        let cat = zencodec::CategorizedError::category(&err);
        assert!(
            matches!(cat, zencodec::ErrorCategory::Image(_)),
            "cut={cut} error must categorize as incomplete/image error, got {cat:?}"
        );
    }
}

#[test]
fn test_dyn_incremental_decoder() {
    let data = std::fs::read(resource("gallery1/1.webp")).expect("read 1.webp");
    let (ref_pixels, _rw, rh) = zenwebp::oneshot::decode_rgba(&data).expect("oneshot decode");

    let config = WebpDecoderConfig::new();
    let preferred = [PixelDescriptor::RGBA8_SRGB];
    let mut dyn_dec: Box<dyn DynIncrementalDecoder> = zencodec::decode::DecoderConfig::job(config)
        .dyn_incremental_decoder(&preferred)
        .expect("dyn_incremental_decoder");

    let mut offset = 0;
    let chunk_size = 512;
    let mut total_rows = 0;
    let mut collected = Vec::new();

    while offset < data.len() {
        let end = (offset + chunk_size).min(data.len());
        let chunk = &data[offset..end];
        let is_eof = end == data.len();

        let outcome: PushOutcome = dyn_dec
            .push_chunk(chunk, is_eof, None)
            .expect("dyn push_chunk");

        offset += outcome.bytes_consumed;

        while let Some((y, slice)) = dyn_dec.pull_batch().expect("dyn pull_batch") {
            assert_eq!(y, total_rows);
            collected.extend_from_slice(slice.as_strided_bytes());
            total_rows += slice.rows();
        }

        if outcome.status == PushStatus::Complete {
            break;
        }
    }

    assert!(dyn_dec.is_complete());
    assert_eq!(total_rows, rh);
    assert_eq!(collected, ref_pixels);
}
