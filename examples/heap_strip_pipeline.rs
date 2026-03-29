//! Streaming decode demo: decode a WebP via `StreamingDecode::next_batch()`,
//! checksum each strip, and discard. Never holds full-frame RGB in memory.
//!
//! Usage:
//!   cargo run --release --features zencodec --example heap_strip_pipeline -- input.webp
//!
//! To measure peak memory:
//!   heaptrack target/release/examples/heap_strip_pipeline input.webp
//!
//! Expected peak: ~300 KB (input + strip + row cache) vs ~35 MB for full-frame decode of 3K.

use std::borrow::Cow;
use std::env;
use std::fs;

use zencodec::decode::{DecodeJob, DecoderConfig, StreamingDecode};
use zenwebp::zencodec::WebpDecoderConfig;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.webp>", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    let data = fs::read(path).expect("failed to read input file");
    let data_len = data.len();

    eprintln!("Input: {} ({} bytes)", path, data_len);

    // -- Streaming decode: strip-at-a-time, never holds full frame --
    let config = WebpDecoderConfig::new();
    let job = config.job();

    let mut decoder = match job.streaming_decoder(Cow::Borrowed(&data), &[]) {
        Ok(dec) => dec,
        Err(e) => {
            eprintln!("Streaming decode not supported for this file: {e}");
            eprintln!("(Only lossy VP8 is supported for streaming decode.)");
            std::process::exit(1);
        }
    };

    let info = decoder.info();
    let width = info.width;
    let height = info.height;
    eprintln!("Image: {width}x{height}");

    let mut total_rows = 0u32;
    let mut batch_count = 0u32;
    let mut checksum: u64 = 0;

    while let Some((y_start, strip)) = decoder.next_batch().expect("decode error") {
        let rows = strip.rows();
        batch_count += 1;

        // "Process" the strip: accumulate a checksum, then discard.
        // In a real pipeline, this would be a resize or filter operation.
        for row_idx in 0..rows {
            let row = strip.row(row_idx);
            for &byte in row {
                // Simple additive checksum (not cryptographic, just demonstrates processing)
                checksum = checksum.wrapping_add(u64::from(byte));
            }
        }

        total_rows += rows;

        eprintln!("  batch {batch_count}: y={y_start}, rows={rows} (total: {total_rows}/{height})");
    }

    eprintln!();
    eprintln!("Done: {total_rows} rows in {batch_count} batches");
    eprintln!("Checksum: {checksum:#018x}");
    eprintln!();
    eprintln!("Memory note: peak RSS should be ~300 KB for row cache + strip buffer,");
    eprintln!(
        "not ~{} MB for full-frame RGB.",
        width as u64 * height as u64 * 3 / (1024 * 1024)
    );
}
