#![cfg(not(target_arch = "wasm32"))]
//! Three-way interleaved benchmark: zenwebp vs libwebp (C) vs image-webp (pure Rust).
//!
//! Run with: RUSTFLAGS="-C target-cpu=native" cargo bench --bench compare_zenbench

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

fn encode_zenwebp(rgb: &[u8], w: u32, h: u32, quality: f32, method: u8) -> Vec<u8> {
    let config = zenwebp::EncoderConfig::new_lossy()
        .with_quality(quality)
        .with_method(method);
    zenwebp::EncodeRequest::new(&config, rgb, zenwebp::PixelLayout::Rgb8, w, h)
        .encode()
        .unwrap()
}

fn encode_libwebp(rgb: &[u8], w: u32, h: u32, quality: f32, method: u8) -> Vec<u8> {
    webpx::EncoderConfig::with_preset(webpx::Preset::Default, quality)
        .method(method)
        .encode_rgb(rgb, w, h, webpx::Unstoppable)
        .unwrap()
}

struct TestImage {
    name: &'static str,
    subdir: &'static str,
    filename: &'static str,
}

const IMAGES: &[TestImage] = &[
    TestImage { name: "photo_512", subdir: "CID22/CID22-512/validation", filename: "792079.png" },
    TestImage { name: "codec_wiki", subdir: "gb82-sc", filename: "codec_wiki.png" },
    TestImage { name: "terminal", subdir: "gb82-sc", filename: "terminal.png" },
];

zenbench::main!(decode_compare, encode_compare, lossless_compare);

fn decode_compare(suite: &mut zenbench::Suite) {
    for img in IMAGES {
        let path = match corpus_path(img.subdir, img.filename) {
            Some(p) => p,
            None => { eprintln!("Skipping {}: not found", img.name); continue; }
        };
        let (rgb, w, h) = match load_png_rgb(&path) {
            Some(d) => d,
            None => continue,
        };
        let webp_data = encode_zenwebp(&rgb, w, h, 75.0, 4);
        let pixels = (w as u64) * (h as u64);

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
                    let mut decoder = image_webp::WebPDecoder::new(std::io::Cursor::new(black_box(&bytes))).unwrap();
                    let size = decoder.output_buffer_size().unwrap();
                    let mut out = vec![0u8; size];
                    decoder.read_image(&mut out).unwrap();
                    black_box(out)
                })
            });
        });
    }
}

fn encode_compare(suite: &mut zenbench::Suite) {
    let path = match corpus_path("CID22/CID22-512/validation", "792079.png") {
        Some(p) => p,
        None => { eprintln!("Skipping encode: corpus not found"); return; }
    };
    let (rgb, w, h) = match load_png_rgb(&path) {
        Some(d) => d,
        None => return,
    };
    let pixels = (w as u64) * (h as u64);

    for method in [0u8, 4, 6] {
        let label = format!("encode_m{method}_q75");
        let rgb = rgb.clone();

        suite.compare(&label, |group| {
            group.throughput(Throughput::Elements(pixels));
            group.throughput_unit("pixels");

            let data = rgb.clone();
            group.bench("zenwebp", move |b| {
                let d = data.clone();
                b.iter(move || {
                    let config = zenwebp::EncoderConfig::new_lossy()
                        .with_quality(75.0)
                        .with_method(method);
                    black_box(
                        zenwebp::EncodeRequest::new(
                            &config,
                            black_box(&d),
                            zenwebp::PixelLayout::Rgb8,
                            w,
                            h,
                        )
                        .encode()
                        .unwrap(),
                    )
                })
            });

            let data = rgb.clone();
            group.bench("libwebp", move |b| {
                let d = data.clone();
                b.iter(move || {
                    black_box(
                        webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
                            .method(method)
                            .encode_rgb(black_box(&d), w, h, webpx::Unstoppable)
                            .unwrap(),
                    )
                })
            });

            if method <= 4 {
                let data = rgb.clone();
                group.bench("image-webp", move |b| {
                    let d = data.clone();
                    b.iter(move || {
                        let mut out = Vec::new();
                        let mut encoder = image_webp::WebPEncoder::new(&mut out);
                        encoder.set_params(image_webp::EncoderParams::default());
                        encoder
                            .encode(black_box(&d), w, h, image_webp::ColorType::Rgb8)
                            .unwrap();
                        black_box(out)
                    })
                });
            }
        });
    }
}

fn lossless_compare(suite: &mut zenbench::Suite) {
    let path = match corpus_path("CID22/CID22-512/validation", "792079.png") {
        Some(p) => p,
        None => { eprintln!("Skipping lossless: corpus not found"); return; }
    };
    let (rgb, w, h) = match load_png_rgb(&path) {
        Some(d) => d,
        None => return,
    };
    let pixels = (w as u64) * (h as u64);
    let rgba: Vec<u8> = rgb.chunks(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect();

    // Default lossless (method 4, quality 75 for all three)
    suite.compare("lossless_m4", |group| {
        group.throughput(Throughput::Elements(pixels));
        group.throughput_unit("pixels");

        let data = rgba.clone();
        group.bench("zenwebp", move |b| {
            let d = data.clone();
            b.iter(move || {
                let config = zenwebp::EncoderConfig::new_lossless();
                black_box(
                    zenwebp::EncodeRequest::new(
                        &config,
                        black_box(&d),
                        zenwebp::PixelLayout::Rgba8,
                        w,
                        h,
                    )
                    .encode()
                    .unwrap(),
                )
            })
        });

        let data = rgba.clone();
        group.bench("libwebp", move |b| {
            let d = data.clone();
            b.iter(move || {
                black_box(
                    webpx::EncoderConfig::new_lossless()
                        .encode_rgba(black_box(&d), w, h, webpx::Unstoppable)
                        .unwrap(),
                )
            })
        });

        let data = rgba.clone();
        group.bench("image-webp", move |b| {
            let d = data.clone();
            b.iter(move || {
                let mut out = Vec::new();
                let mut encoder = image_webp::WebPEncoder::new(&mut out);
                // image-webp 0.2 is lossless-only (VP8L), no params needed
                encoder
                    .encode(black_box(&d), w, h, image_webp::ColorType::Rgba8)
                    .unwrap();
                black_box(out)
            })
        });
    });

    // Fast lossless (method 0) — speed-focused comparison
    suite.compare("lossless_m0", |group| {
        group.throughput(Throughput::Elements(pixels));
        group.throughput_unit("pixels");

        let data = rgba.clone();
        group.bench("zenwebp", move |b| {
            let d = data.clone();
            b.iter(move || {
                let config = zenwebp::EncoderConfig::new_lossless().with_method(0);
                black_box(
                    zenwebp::EncodeRequest::new(
                        &config,
                        black_box(&d),
                        zenwebp::PixelLayout::Rgba8,
                        w,
                        h,
                    )
                    .encode()
                    .unwrap(),
                )
            })
        });

        let data = rgba.clone();
        group.bench("libwebp", move |b| {
            let d = data.clone();
            b.iter(move || {
                black_box(
                    webpx::EncoderConfig::new_lossless()
                        .method(0)
                        .encode_rgba(black_box(&d), w, h, webpx::Unstoppable)
                        .unwrap(),
                )
            })
        });
    });
}
