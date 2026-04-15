//! Diagnostic: does libwebp's default `WebPEncodeLosslessBGRA` (exact=0) also
//! round-trip the rose byte-exactly on visible pixels? Or does it alter RGB
//! in ways zenwebp does not?
#![cfg(all(feature = "std", not(target_arch = "wasm32")))]

#[test]
fn libwebp_rose_lossless_bitexact_on_visible_pixels() {
    let data = std::fs::read("tests/images/gallery2/1_webp_ll.webp").expect("read rose");
    let (rgba, w, h) = zenwebp::oneshot::decode_rgba(&data).expect("decode rose");

    // Encode through libwebp (via webpx), default exact=false.
    let webp_bytes = webpx::EncoderConfig::new_lossless()
        // .exact(false) is default
        .encode_rgba(&rgba, w, h, webpx::Unstoppable)
        .expect("libwebp encode");

    // Decode with libwebp too (the "c-only" baseline path).
    let (lib_pixels, lw, lh) = webpx::decode_rgba(&webp_bytes).expect("libwebp decode");
    eprintln!(
        "libwebp decode produced {} bytes ({lw}x{lh})",
        lib_pixels.len()
    );
    assert_eq!(lw, w);
    assert_eq!(lh, h);

    // Compare on visible pixels only.
    let mut mismatch = 0usize;
    let mut max_d = [0u16; 4];
    for i in 0..(w * h) as usize {
        let src = &rgba[i * 4..i * 4 + 4];
        let dst = &lib_pixels[i * 4..i * 4 + 4];
        if src[3] == 0 {
            continue;
        }
        for c in 0..4 {
            let d = (src[c] as i16 - dst[c] as i16).unsigned_abs();
            max_d[c] = max_d[c].max(d);
        }
        if src != dst {
            mismatch += 1;
        }
    }
    eprintln!(
        "libwebp lossless RGBA roundtrip: {mismatch} visible pixels differ, maxΔ [R={} G={} B={} A={}]",
        max_d[0], max_d[1], max_d[2], max_d[3]
    );
    // Expect 0 — libwebp lossless preserves visible pixels exactly too.
    assert_eq!(
        mismatch, 0,
        "libwebp corrupts visible pixels?! maxΔ={max_d:?}"
    );
}
