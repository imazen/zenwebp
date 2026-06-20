#![no_main]

use libfuzzer_sys::fuzz_target;
use zenwebp::{DecodeConfig, Limits};

// Restrictive limits for fuzzing — mirrors zengif's `fuzz_decode` convention.
//
// The library's PRODUCTION default is `Limits::default()` (120 MP, 1 GB), which
// is the right ceiling for real decoding: a 16384x16384 WebP is spec-legal and a
// 52 MP image is a perfectly normal photo. But a malformed VP8L/VP8 header can
// declare a huge canvas from a handful of input bytes (a decompression bomb), and
// decoding tens of megapixels under libFuzzer's sanitizer-coverage instrumentation
// (~40x slower than a release build) blows past the 25s libFuzzer timeout even
// though the work is fully bounded and linear — ~480ms for 52 MP on bare metal.
//
// So we tighten the limits HERE, in the harness, not in the library. The fuzzer's
// job is to explore decode *logic* (header parse, Huffman/transform/backward-ref
// paths, color cache, animation compositing), not to spend its whole budget filling
// a giant trivially-encoded canvas. Inputs whose declared canvas exceeds these caps
// return a typed `Err` immediately instead of allocating and looping over millions
// of pixels. This was issue #68: a 104-byte input declaring 12801x4097 (52 MP).
fn fuzz_config() -> DecodeConfig {
    let limits = Limits::default()
        .max_dimensions(4096, 4096) // generous enough to reach real decode paths
        .max_total_pixels(4_000_000) // ~4 MP — well past every interesting code path
        .max_memory(64 * 1024 * 1024); // 64 MB working-set ceiling
    DecodeConfig::default().limits(limits)
}

fuzz_target!(|input: &[u8]| {
    let config = fuzz_config();

    // Try oneshot RGBA decode — should never panic on any input, and with the
    // fuzzing limits should never run unbounded on an oversized declared canvas.
    let _ = zenwebp::DecodeRequest::new(&config, input).decode_rgba();

    // Also try RGB path
    let _ = zenwebp::DecodeRequest::new(&config, input).decode_rgb();

    // Try animation decode (limits applied via the same config)
    if let Ok(mut decoder) = zenwebp::mux::AnimationDecoder::new_with_config(input, &config) {
        while let Ok(Some(_frame)) = decoder.next_frame() {}
    }
});
