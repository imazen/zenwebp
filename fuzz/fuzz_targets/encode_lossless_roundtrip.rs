#![no_main]

//! Lossless encode → decode → exact-pixel-oracle fuzzing. All logic lives in
//! `encode_roundtrip_core.rs`, shared verbatim with `tests/fuzz_regression.rs`
//! so regression seeds replay identically on the stable toolchain.

use libfuzzer_sys::fuzz_target;

include!("encode_roundtrip_core.rs");

fuzz_target!(|input: &[u8]| {
    run_encode_lossless_roundtrip(input);
});
