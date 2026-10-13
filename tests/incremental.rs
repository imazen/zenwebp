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

// ── Sans-I/O Consumer Parity Tests (Stdio + Tokio Async) ────────────────────

struct CollectSink {
    desc: Option<PixelDescriptor>,
    width: u32,
    height: u32,
    buf: Vec<u8>,
    finished: bool,
}

impl CollectSink {
    fn new() -> Self {
        Self {
            desc: None,
            width: 0,
            height: 0,
            buf: Vec::new(),
            finished: false,
        }
    }
}

impl zencodec::decode::DecodeRowSink for CollectSink {
    fn begin(&mut self, width: u32, height: u32, desc: PixelDescriptor) -> Result<(), zencodec::decode::SinkError> {
        self.width = width;
        self.height = height;
        self.desc = Some(desc);
        let bpp = desc.bytes_per_pixel();
        let stride = width as usize * bpp;
        self.buf = vec![0u8; stride * height as usize];
        Ok(())
    }

    fn provide_next_buffer(
        &mut self,
        y: u32,
        height: u32,
        width: u32,
        desc: PixelDescriptor,
    ) -> Result<zenpixels::PixelSliceMut<'_>, zencodec::decode::SinkError> {
        let bpp = desc.bytes_per_pixel();
        let stride = width as usize * bpp;
        let start = y as usize * stride;
        let len = height as usize * stride;
        let slice = &mut self.buf[start..start + len];
        zenpixels::PixelSliceMut::new(slice, width, height, stride, desc).map_err(|e| e.to_string().into())
    }

    fn finish(&mut self) -> Result<(), zencodec::decode::SinkError> {
        self.finished = true;
        Ok(())
    }
}

#[test]
fn test_zenwebp_stdio_consumer_pixel_parity() {
    let synthetic_lossless = generate_lossless_webp(64, 48);
    let (ref_pixels, rw, rh) = zenwebp::oneshot::decode_rgba(&synthetic_lossless).expect("oneshot decode");

    let mut inc_decoder = zencodec::decode::DecoderConfig::job(WebpDecoderConfig::new())
        .into_incremental_decoder(&[PixelDescriptor::RGBA8_SRGB])
        .expect("incremental decoder init");
    let mut sink = CollectSink::new();
    let reader = std::io::Cursor::new(&synthetic_lossless);

    let info = zencodec::helpers::sansio::pump_read_to_sink(reader, &mut inc_decoder, &mut sink, None, 256)
        .expect("stdio pump succeeds");

    assert_eq!(info.width, rw);
    assert_eq!(info.height, rh);
    assert!(sink.finished);
    assert_eq!(sink.buf.len(), ref_pixels.len());
    assert_eq!(
        sink.buf, ref_pixels,
        "Stdio consumer produced different pixels than oneshot decode!"
    );
}

async fn pump_async_read_to_sink<R, D>(
    mut reader: R,
    decoder: &mut D,
    sink: &mut (dyn zencodec::decode::DecodeRowSink + '_),
    buffer_size: usize,
) -> Result<zencodec::decode::OutputInfo, zencodec::helpers::sansio::PumpError<D::Error>>
where
    R: tokio::io::AsyncRead + Unpin,
    D: IncrementalDecode,
{
    use tokio::io::AsyncReadExt;
    use zencodec::helpers::sansio::PumpError;

    let buffer_size = buffer_size.max(512);
    let mut buf = vec![0u8; buffer_size];
    let mut sink_begun = false;
    let mut chosen_desc: Option<PixelDescriptor> = None;
    let mut total_w = 0u32;
    let mut total_h = 0u32;

    let mut drain_batches =
        |dec: &mut D, dst_sink: &mut (dyn zencodec::decode::DecodeRowSink + '_)| -> Result<(), PumpError<D::Error>> {
            if !sink_begun {
                let info_h = dec.info().map(|i| i.height);
                if let Some((meta, slice)) = dec.pull_batch_meta().map_err(PumpError::Codec)? {
                    let desc = slice.descriptor();
                    chosen_desc = Some(desc);
                    let w = slice.width();
                    let h = info_h.unwrap_or(slice.rows() as u32);
                    total_w = w;
                    total_h = h;
                    dst_sink
                        .begin(w, h, desc)
                        .map_err(|e| PumpError::Codec(D::wrap_sink_error(e)))?;
                    sink_begun = true;

                    let batch_rows = slice.rows() as u32;
                    let mut dst = dst_sink
                        .provide_next_buffer(meta.y, batch_rows, w, desc)
                        .map_err(|e| PumpError::Codec(D::wrap_sink_error(e)))?;
                    for row_idx in 0..batch_rows {
                        dst.row_mut(row_idx).copy_from_slice(slice.row(row_idx));
                    }
                }
            }

            if sink_begun {
                while let Some(_rows) = dec.pull_batch_into(dst_sink).map_err(PumpError::Codec)? {}
            }
            Ok(())
        };

    loop {
        drain_batches(decoder, sink)?;

        if decoder.is_complete() {
            if sink_begun {
                sink.finish()
                    .map_err(|e| PumpError::Codec(D::wrap_sink_error(e)))?;
            }
            let info = decoder.info().ok_or(PumpError::UnexpectedEof)?;
            let desc = chosen_desc.unwrap_or(PixelDescriptor::RGBA8_SRGB);
            return Ok(zencodec::decode::OutputInfo::full_decode(info.width, info.height, desc));
        }

        let n = reader.read(&mut buf).await.map_err(PumpError::Io)?;
        let is_eof = n == 0;

        let _outcome = decoder
            .push_chunk(&buf[..n], is_eof, None)
            .map_err(PumpError::Codec)?;

        drain_batches(decoder, sink)?;

        if is_eof {
            drain_batches(decoder, sink)?;
            if !decoder.is_complete() {
                return Err(PumpError::UnexpectedEof);
            }
            if sink_begun {
                sink.finish()
                    .map_err(|e| PumpError::Codec(D::wrap_sink_error(e)))?;
            }
            let desc = chosen_desc.unwrap_or(PixelDescriptor::RGBA8_SRGB);
            let w = if total_w > 0 {
                total_w
            } else {
                decoder.info().map(|i| i.width).unwrap_or(0)
            };
            let h = if total_h > 0 {
                total_h
            } else {
                decoder.info().map(|i| i.height).unwrap_or(0)
            };
            return Ok(zencodec::decode::OutputInfo::full_decode(w, h, desc));
        }
    }
}

#[tokio::test]
async fn test_zenwebp_tokio_async_consumer_pixel_parity() {
    let synthetic_lossless = generate_lossless_webp(64, 48);
    let (ref_pixels, rw, rh) = zenwebp::oneshot::decode_rgba(&synthetic_lossless).expect("oneshot decode");

    let mut inc_decoder = zencodec::decode::DecoderConfig::job(WebpDecoderConfig::new())
        .into_incremental_decoder(&[PixelDescriptor::RGBA8_SRGB])
        .expect("incremental decoder init");
    let mut sink = CollectSink::new();
    let reader = tokio::io::BufReader::new(&synthetic_lossless[..]);

    let info = pump_async_read_to_sink(reader, &mut inc_decoder, &mut sink, 128)
        .await
        .expect("tokio async pump succeeds");

    assert_eq!(info.width, rw);
    assert_eq!(info.height, rh);
    assert!(sink.finished);
    assert_eq!(sink.buf.len(), ref_pixels.len());
    assert_eq!(
        sink.buf, ref_pixels,
        "Tokio async consumer produced different pixels than oneshot decode!"
    );
}

