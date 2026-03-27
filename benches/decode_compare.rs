#![cfg(not(target_arch = "wasm32"))]
//! Lossy decode benchmark: zenwebp vs libwebp (C) vs image-webp (pure Rust).
//!
//! Encodes test images as lossy WebP Q75 m4, then benchmarks decoding the
//! same bytes with all three decoders. Reports throughput in Mpixels/s.
//!
//! Run with: RUSTFLAGS="-C target-cpu=native" cargo bench --bench decode_compare

use std::path::PathBuf;
use zenbench::{Throughput, black_box};

fn corpus_path(subdir: &str, filename: &str) -> Option<PathBuf> {
    let corpus = codec_corpus::Corpus::new().ok()?;
    let dir = corpus.get(subdir).ok()?;
    let path = dir.join(filename);
    if path.exists() { Some(path) } else { None }
}

fn load_png_rgb(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| &c[..3])
            .copied()
            .collect(),
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

struct TestImage {
    name: &'static str,
    subdir: &'static str,
    filename: &'static str,
}

const IMAGES: &[TestImage] = &[
    // ~4K screenshot (3508x2480)
    TestImage {
        name: "4k_wiki",
        subdir: "png-conformance",
        filename: "wm_upload_wikimedia_org_a23d1e831e128dff.png",
    },
    // ~3K screenshot (2940x1912)
    TestImage {
        name: "3k_imac",
        subdir: "gb82-sc",
        filename: "imac_g3.png",
    },
    // ~2.5K screenshot (2560x1664)
    TestImage {
        name: "2k_wiki",
        subdir: "gb82-sc",
        filename: "codec_wiki.png",
    },
    // ~2K photo (2048x2048)
    TestImage {
        name: "2k_photo",
        subdir: "clic2025/final-test",
        filename: "ebfd571f1c6824316047a29cb5f376eec15f56dd51821119c1842be068a8b950.png",
    },
    // ~1.6K screenshot (1646x1062)
    TestImage {
        name: "1k_term",
        subdir: "gb82-sc",
        filename: "terminal.png",
    },
    // 512px photo
    TestImage {
        name: "512_photo",
        subdir: "CID22/CID22-512/validation",
        filename: "792079.png",
    },
];

zenbench::main!(decode_compare);

fn decode_compare(suite: &mut zenbench::Suite) {
    for img in IMAGES {
        let path = match corpus_path(img.subdir, img.filename) {
            Some(p) => p,
            None => {
                eprintln!("Skipping {}: not found", img.name);
                continue;
            }
        };
        let (rgb, w, h) = match load_png_rgb(&path) {
            Some(d) => d,
            None => {
                eprintln!("Skipping {}: failed to load PNG", img.name);
                continue;
            }
        };

        // Encode as lossy WebP Q75 m4 using zenwebp
        let config = zenwebp::EncoderConfig::new_lossy()
            .with_quality(75.0)
            .with_method(4);
        let webp_data =
            zenwebp::EncodeRequest::new(&config, &rgb, zenwebp::PixelLayout::Rgb8, w, h)
                .encode()
                .unwrap();

        let pixels = (w as u64) * (h as u64);
        eprintln!(
            "{}: {}x{} ({:.2} MPix), WebP {} bytes",
            img.name,
            w,
            h,
            pixels as f64 / 1_000_000.0,
            webp_data.len()
        );

        suite.compare(&format!("decode_{}", img.name), |group| {
            group.throughput(Throughput::Elements(pixels));
            group.throughput_unit("pixels");

            let data = webp_data.clone();
            group.bench("zenwebp", move |b| {
                let d = data.clone();
                let config = zenwebp::DecodeConfig::default();
                b.with_input(move || d.clone()).run(|bytes| {
                    black_box(
                        zenwebp::DecodeRequest::new(&config, black_box(&bytes))
                            .decode_rgba()
                            .unwrap(),
                    )
                })
            });

            let data = webp_data.clone();
            group.bench("libwebp", move |b| {
                let d = data.clone();
                b.with_input(move || d.clone()).run(|bytes| {
                    let decoder = webp::Decoder::new(black_box(&bytes));
                    black_box(decoder.decode().unwrap())
                })
            });

            let data = webp_data.clone();
            group.bench("image-webp", move |b| {
                let d = data.clone();
                b.with_input(move || d.clone()).run(|bytes| {
                    let mut decoder =
                        image_webp::WebPDecoder::new(std::io::Cursor::new(black_box(&bytes)))
                            .unwrap();
                    let size = decoder.output_buffer_size().unwrap();
                    let mut out = vec![0u8; size];
                    decoder.read_image(&mut out).unwrap();
                    black_box(out)
                })
            });
        });
    }
}
