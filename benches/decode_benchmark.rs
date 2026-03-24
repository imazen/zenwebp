#![cfg(not(target_arch = "wasm32"))]
//! Criterion benchmarks for zenwebp decoding performance.
//!
//! Compares zenwebp (single-threaded and threaded) against libwebp.
//! Uses codec-corpus for reproducible test images.
//!
//! Run with: cargo bench --bench decode_benchmark
//! Run with native: RUSTFLAGS="-C target-cpu=native" cargo bench --bench decode_benchmark

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::{Path, PathBuf};
use zenwebp::{DecodeConfig, DecodeRequest, EncodeRequest, EncoderConfig, PixelLayout};

/// Load a PNG image, encode to WebP at Q75 m4, return (webp_data, width, height).
fn make_webp(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;

    let rgb_data = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| &c[..3])
            .copied()
            .collect(),
        _ => return None,
    };

    let config = EncoderConfig::new_lossy().with_quality(75.0).with_method(4);
    let webp =
        EncodeRequest::new(&config, &rgb_data, PixelLayout::Rgb8, info.width, info.height)
            .encode()
            .ok()?;

    Some((webp, info.width, info.height))
}

fn corpus_path(subdir: &str, filename: &str) -> Option<PathBuf> {
    let corpus = codec_corpus::Corpus::new().ok()?;
    let dir = corpus.get(subdir).ok()?;
    let path = dir.join(filename);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

struct BenchImage {
    name: &'static str,
    subdir: &'static str,
    filename: &'static str,
}

// Large images (width >= 512, threading active)
const LARGE: &[BenchImage] = &[
    BenchImage {
        name: "codec_wiki_2560x1664",
        subdir: "gb82-sc",
        filename: "codec_wiki.png",
    },
    BenchImage {
        name: "imac_2940x1912",
        subdir: "gb82-sc",
        filename: "imac_g3.png",
    },
    BenchImage {
        name: "terminal_1646x1062",
        subdir: "gb82-sc",
        filename: "terminal.png",
    },
    BenchImage {
        name: "windows_2560x1392",
        subdir: "gb82-sc",
        filename: "windows.png",
    },
];

// Small images (width < 512, threading inactive — verify no regression)
const SMALL: &[BenchImage] = &[BenchImage {
    name: "shirt_400x400",
    subdir: "imageflow/test_inputs",
    filename: "shirt_transparent.png",
}];

fn bench_decode_threading(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_threaded");
    group.sample_size(20);

    for img in LARGE.iter().chain(SMALL.iter()) {
        let path = match corpus_path(img.subdir, img.filename) {
            Some(p) => p,
            None => {
                eprintln!("Skipping {}: not found in corpus", img.name);
                continue;
            }
        };

        let (webp_data, width, height) = match make_webp(&path) {
            Some(d) => d,
            None => {
                eprintln!("Skipping {}: encode failed", img.name);
                continue;
            }
        };

        let pixels = (width * height) as u64;
        group.throughput(Throughput::Elements(pixels));

        // zenwebp (single-threaded, current implementation)
        let config_zen = DecodeConfig::default();
        group.bench_with_input(
            BenchmarkId::new("zenwebp", img.name),
            &webp_data,
            |b, data| {
                b.iter(|| {
                    DecodeRequest::new(&config_zen, black_box(data))
                        .decode_rgba()
                        .unwrap()
                })
            },
        );

        // libwebp single-threaded (via webpx advanced API)
        let libwebp_cfg_1t = webpx::DecoderConfig::new().use_threads(false);
        group.bench_with_input(
            BenchmarkId::new("libwebp_1t", img.name),
            &webp_data,
            |b, data| {
                b.iter(|| {
                    webpx::Decoder::new(black_box(data))
                        .unwrap()
                        .config(libwebp_cfg_1t.clone())
                        .decode_rgba_raw()
                        .unwrap()
                })
            },
        );

        // libwebp threaded (via webpx advanced API, use_threads=true)
        let libwebp_cfg_2t = webpx::DecoderConfig::new().use_threads(true);
        group.bench_with_input(
            BenchmarkId::new("libwebp_2t", img.name),
            &webp_data,
            |b, data| {
                b.iter(|| {
                    webpx::Decoder::new(black_box(data))
                        .unwrap()
                        .config(libwebp_cfg_2t.clone())
                        .decode_rgba_raw()
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_decode_threading);
criterion_main!(benches);
