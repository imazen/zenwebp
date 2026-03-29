#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: &[u8]| {
    // Try v2 lossy decode — should never panic on any input
    let mut ctx = zenwebp::DecoderContext::new();
    ctx.set_dithering_strength(0);
    let mut output = Vec::new();
    let _ = ctx.decode_to_rgb(input, &mut output, 3);

    // Also try RGBA path
    let _ = ctx.decode_to_rgb(input, &mut output, 4);

    // Try animation decode
    let _ = ctx.decode_animation(input, |_frame| true);
});
