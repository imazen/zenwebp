//! Prints libwebp's `decode_yuv` digests for the `tests/lossless_yuv.rs`
//! cases, next to whether zenwebp's `decode_yuv420` agrees.
//!
//! The expected values in that test come from libwebp main, not the 1.6.0
//! that the webpx dev-dependency links: build this example with
//! `[patch.crates-io] libwebp-sys = { path = ... }` pointing at a
//! libwebp-sys whose `vendor/` holds a libwebp main checkout (verified with
//! 1844042d1c, 2026-10-19). Against 1.6.0 every lossless case differs.
//!
//! cargo run --release --example lossless_yuv_oracle

#[path = "../tests/lossless_yuv/cases.rs"]
mod cases;

fn lib_digest(data: &[u8]) -> u64 {
    let p = webpx::Decoder::new(data)
        .unwrap()
        .config(webpx::DecoderConfig::new().limits(webpx::Limits::none()))
        .decode_yuv()
        .unwrap();
    cases::yuv_digest(p.width, p.height, &p.y, &p.u, &p.v)
}

fn zen_digest(data: &[u8]) -> u64 {
    let p = zenwebp::decoder::decode_yuv420(data).unwrap();
    cases::yuv_digest(p.y_width, p.y_height, &p.y, &p.u, &p.v)
}

fn verdict(lib: u64, zen: u64) -> &'static str {
    if lib == zen { "same" } else { "DIFFERS" }
}

fn main() {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus cache unavailable");
    for (dataset, name) in cases::CORPUS {
        let data = std::fs::read(corpus.get(dataset).unwrap().join(name)).unwrap();
        let (l, z) = (lib_digest(&data), zen_digest(&data));
        println!("(\"{name}\", 0x{l:016x}), // zenwebp {}", verdict(l, z));
    }
    for &(name, w, h) in cases::SYNTHETIC {
        let rgba = cases::synthetic_rgba(name, w, h);
        let data = cases::encode_exact(&rgba, w, h);
        let (l, z) = (lib_digest(&data), zen_digest(&data));
        println!("0x{l:016x}, // {name} {w}x{h}, zenwebp {}", verdict(l, z));
    }
}
