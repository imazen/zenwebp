#![no_main]

//! Structural inventory: for any bytes, `DecodeJob::inventory` must return an
//! inventory that covers the input exactly (`validate`) without panicking.

use libfuzzer_sys::fuzz_target;
use zencodec::decode::{DecodeJob, DecoderConfig};
use zenwebp::zencodec::WebpDecoderConfig;

fuzz_target!(|data: &[u8]| {
    let inv = WebpDecoderConfig::new()
        .job()
        .inventory(data)
        .expect("inventory only errors on cancellation or the part cap")
        .expect("inventory is declared");
    assert_eq!(inv.input_len(), data.len() as u64);
    inv.validate().expect("inventory must tile the input");
});
