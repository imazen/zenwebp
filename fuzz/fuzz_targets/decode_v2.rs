#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: &[u8]| {
    // Try oneshot RGBA decode — should never panic on any input
    let _ = zenwebp::oneshot::decode_rgba(input);

    // Also try RGB path
    let _ = zenwebp::oneshot::decode_rgb(input);

    // Try animation decode
    if let Ok(mut decoder) = zenwebp::mux::AnimationDecoder::new(input) {
        while let Ok(Some(_frame)) = decoder.next_frame() {}
    }
});
