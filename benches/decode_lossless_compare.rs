#![cfg(not(target_arch = "wasm32"))]
//! Lossless decode benchmark: zenwebp vs libwebp (C).
//! Decodes the SAME lossless WebP bytes through all three decoders.

use std::path::PathBuf;
use zenbench::{Throughput, black_box};

fn corpus_path(subdir: &str, filename: &str) -> Option<PathBuf> {
    let corpus = codec_corpus::Corpus::new().ok()?;
    let dir = corpus.get(subdir).ok()?;
    let path = dir.join(filename);
    if path.exists() { Some(path) } else { None }
}

fn load_png_rgba(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgb => buf[..info.buffer_size()]
            .chunks(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        _ => return None,
    };
    Some((rgba, info.width, info.height))
}

struct TestImage {
    name: &'static str,
    subdir: &'static str,
    filename: &'static str,
}

const IMAGES: &[TestImage] = &[
    TestImage {
        name: "photo_512",
        subdir: "CID22/CID22-512/validation",
        filename: "792079.png",
    },
    TestImage {
        name: "codec_wiki",
        subdir: "gb82-sc",
        filename: "codec_wiki.png",
    },
    TestImage {
        name: "terminal",
        subdir: "gb82-sc",
        filename: "terminal.png",
    },
];

zenbench::main!(decode_lossless);

fn decode_lossless(suite: &mut zenbench::Suite) {
    for img in IMAGES {
        let path = match corpus_path(img.subdir, img.filename) {
            Some(p) => p,
            None => {
                eprintln!("Skipping {}: not found", img.name);
                continue;
            }
        };
        let (rgba, w, h) = match load_png_rgba(&path) {
            Some(d) => d,
            None => continue,
        };
        let pixels = (w as u64) * (h as u64);

        // Encode lossless with zenwebp
        let config = zenwebp::EncoderConfig::new_lossless().with_method(4);
        let webp_data =
            zenwebp::EncodeRequest::new(&config, &rgba, zenwebp::PixelLayout::Rgba8, w, h)
                .encode()
                .unwrap();

        // Also encode with libwebp for a fair comparison (same format, different encoder)
        let libwebp_data = webpx::EncoderConfig::new_lossless()
            .method(4)
            .encode_rgba(&rgba, w, h, webpx::Unstoppable)
            .unwrap();

        eprintln!(
            "{}: {}x{}, zen={} lib={} bytes",
            img.name,
            w,
            h,
            webp_data.len(),
            libwebp_data.len()
        );

        // Decode zenwebp-encoded data
        let label = format!("lossless_decode_{}", img.name);
        let zen_webp = webp_data.clone();
        let lib_webp = libwebp_data.clone();

        suite.compare(&label, |group| {
            group.throughput(Throughput::Elements(pixels));
            group.throughput_unit("pixels");

            // zenwebp decoding zenwebp-encoded data
            let data = zen_webp.clone();
            group.bench("zenwebp", move |b| {
                let d = data.clone();
                b.with_input(move || d.clone()).run(|bytes| {
                    let config = zenwebp::DecodeConfig::default();
                    black_box(
                        zenwebp::DecodeRequest::new(&config, black_box(&bytes))
                            .decode_rgba()
                            .unwrap(),
                    )
                })
            });

            // libwebp decoding libwebp-encoded data (apples to apples — same encoder)
            let data = lib_webp.clone();
            group.bench("libwebp", move |b| {
                let d = data.clone();
                b.with_input(move || d.clone()).run(|bytes| {
                    let decoder = webp::Decoder::new(black_box(&bytes));
                    black_box(decoder.decode().unwrap())
                })
            });
        });
    }
}
