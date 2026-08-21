#![no_main]

//! Lossy encode fuzzing across the settings grid (alpha plane carries an
//! exact oracle at `alpha_quality = 100`). All logic lives in
//! `encode_roundtrip_core.rs`, shared verbatim with `tests/fuzz_regression.rs`
//! so regression seeds replay identically on the stable toolchain.

use libfuzzer_sys::fuzz_target;

include!("encode_roundtrip_core.rs");

fuzz_target!(|input: &[u8]| {
    run_encode_lossy_roundtrip(input);
});
