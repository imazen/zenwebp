//! `decode_yuv420` on lossless (VP8L) input, pinned to libwebp main.
//!
//! libwebp converts VP8L to YUV with its encoder's RGB->YUV import since
//! "Have lossless use ImportYUVAFromRGB" (bug 432241412, after 1.6.0), so a
//! lossless file decoded to YUV matches encoding the same RGB. The webpx
//! dev-dependency links 1.6.0, which used an older conversion, so the
//! expected digests come from libwebp main (1844042d1c), produced by
//! `dev/lossless_yuv_oracle.rs` linked against it. Lossless decoding
//! returns the source pixels exactly, so the digests do not depend on the
//! encoder.
#![forbid(unsafe_code)]
#![cfg(not(target_arch = "wasm32"))]

#[path = "lossless_yuv/cases.rs"]
mod cases;

fn digest(data: &[u8]) -> u64 {
    let p = zenwebp::decoder::decode_yuv420(data).unwrap();
    cases::yuv_digest(p.y_width, p.y_height, &p.y, &p.u, &p.v)
}

const CORPUS_MAIN: &[(&str, u64)] = &[
    ("1_webp_ll.webp", 0x69840463c1e718ba),
    ("5_webp_ll.webp", 0x3afab3b7610eeea5),
    ("multi-color.webp", 0x4d169adda264f787),
    ("simple_xmp.webp", 0xd69cc1bff5582c4f),
    ("2-color.webp", 0xd4fe9e41ead5e50f),
    ("simple.webp", 0xd69cc1bff5582c4f),
];

const SYNTHETIC_MAIN: &[u64] = &[
    0x2f390e6f3bcba717, // noise_alpha 1x1
    0xc22999733bc68571, // noise_alpha 2x2
    0x0de577c9ada0a184, // noise_alpha 3x5
    0xcca2611b83d45bdc, // noise_alpha 17x9
    0xadd5191558508f2c, // noise_alpha 64x64
    0xc455e6465606c2e9, // gradient_opaque 33x17
    0x6bcdf0e635e10938, // gradient_opaque 128x96
    0xd8e0237f6ec339f3, // alpha_ramp 64x31
    0xb0bf25c5bc8b3db1, // hidden_rgb 40x40
];

#[test]
fn lossless_yuv_matches_libwebp_main_on_synthetic_alpha() {
    assert_eq!(cases::SYNTHETIC.len(), SYNTHETIC_MAIN.len());
    let mut wrong = Vec::new();
    for (&(name, w, h), &want) in cases::SYNTHETIC.iter().zip(SYNTHETIC_MAIN) {
        let rgba = cases::synthetic_rgba(name, w, h);
        let webp = cases::encode_exact(&rgba, w, h);
        let (back, _, _) = zenwebp::decoder::decode_rgba(&webp).unwrap();
        assert!(back == rgba, "{name} {w}x{h}: lossless roundtrip");
        if digest(&webp) != want {
            wrong.push(format!("{name} {w}x{h}"));
        }
    }
    assert!(wrong.is_empty(), "YUV differs from libwebp main: {wrong:?}");
}

#[test]
fn lossless_yuv_matches_libwebp_main_on_corpus() {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus cache unavailable");
    let mut wrong = Vec::new();
    for (&(dataset, name), &(pinned, want)) in cases::CORPUS.iter().zip(CORPUS_MAIN) {
        assert_eq!(name, pinned);
        let dir = corpus
            .get(dataset)
            .unwrap_or_else(|e| panic!("codec-corpus dataset {dataset} unavailable: {e}"));
        let data = std::fs::read(dir.join(name)).unwrap();
        if digest(&data) != want {
            wrong.push(name);
        }
    }
    assert!(wrong.is_empty(), "YUV differs from libwebp main: {wrong:?}");
}
