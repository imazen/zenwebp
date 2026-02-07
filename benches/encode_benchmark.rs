#![cfg(not(target_arch = "wasm32"))]
//! Criterion benchmarks for zenwebp encoding performance.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::path::Path;
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout, Preset};

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let rgb_data = match info.color_type {
        png::PixelLayout::Rgb => buf[..info.buffer_size()].to_vec(),
        png::PixelLayout::Rgba => {
            let mut rgb = Vec::with_capacity((info.width * info.height * 3) as usize);
            for chunk in buf[..info.buffer_size()].chunks(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        png::PixelLayout::Grayscale => {
            let mut rgb = Vec::with_capacity((info.width * info.height * 3) as usize);
            for &g in &buf[..info.buffer_size()] {
                rgb.extend_from_slice(&[g, g, g]);
            }
            rgb
        }
        png::PixelLayout::GrayscaleAlpha => {
            let mut rgb = Vec::with_capacity((info.width * info.height * 3) as usize);
            for chunk in buf[..info.buffer_size()].chunks(2) {
                let g = chunk[0];
                rgb.extend_from_slice(&[g, g, g]);
            }
            rgb
        }
        _ => return None,
    };

    Some((rgb_data, info.width, info.height))
}

struct TestImage {
    name: &'static str,
    path: &'static str,
    category: &'static str,
}

const TEST_IMAGES: &[TestImage] = &[
    TestImage {
        name: "photo_512",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/792079.png",
        category: "photo",
    },
    TestImage {
        name: "screenshot",
        path: "/home/lilith/work/codec-corpus/gb82-sc/gui.png",
        category: "screenshot",
    },
    TestImage {
        name: "flowers",
        path: "/home/lilith/work/codec-corpus/gb82/flowers-lossless.png",
        category: "photo",
    },
];

fn bench_encode_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_method");

    let (rgb_data, width, height) = TEST_IMAGES
        .iter()
        .find_map(|img| load_png(Path::new(img.path)))
        .expect("No test images found");

    let pixels = (width * height) as u64;
    group.throughput(Throughput::Elements(pixels));

    for method in [0, 2, 4, 6] {
        group.bench_with_input(
            BenchmarkId::new("zenwebp", method),
            &method,
            |b, &method| {
                b.iter(|| {
                    let config = EncoderConfig::new_lossy()
                        .with_quality(75.0)
                        .with_method(method);
                    EncodeRequest::new(
                        &config,
                        black_box(&rgb_data),
                        PixelLayout::Rgb8,
                        width,
                        height,
                    )
                    .encode()
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_encode_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_quality");

    let (rgb_data, width, height) = TEST_IMAGES
        .iter()
        .find_map(|img| load_png(Path::new(img.path)))
        .expect("No test images found");

    let pixels = (width * height) as u64;
    group.throughput(Throughput::Elements(pixels));

    for quality in [50.0, 75.0, 90.0] {
        group.bench_with_input(
            BenchmarkId::new("zenwebp", quality as u32),
            &quality,
            |b, &quality| {
                b.iter(|| {
                    let config = EncoderConfig::new_lossy()
                        .with_quality(quality)
                        .with_method(4);
                    EncodeRequest::new(
                        &config,
                        black_box(&rgb_data),
                        PixelLayout::Rgb8,
                        width,
                        height,
                    )
                    .encode()
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_encode_presets(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_preset");

    let (rgb_data, width, height) = TEST_IMAGES
        .iter()
        .find_map(|img| load_png(Path::new(img.path)))
        .expect("No test images found");

    let pixels = (width * height) as u64;
    group.throughput(Throughput::Elements(pixels));

    let presets = [
        ("default", Preset::Default),
        ("photo", Preset::Photo),
        ("drawing", Preset::Drawing),
        ("icon", Preset::Icon),
    ];

    for (name, preset) in presets {
        group.bench_with_input(BenchmarkId::new("zenwebp", name), &preset, |b, &preset| {
            b.iter(|| {
                let config = EncoderConfig::with_preset(preset, 75.0).with_method(4);
                EncodeRequest::new(
                    &config,
                    black_box(&rgb_data),
                    PixelLayout::Rgb8,
                    width,
                    height,
                )
                .encode()
                .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_encode_by_image_type(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_image_type");

    for test_img in TEST_IMAGES {
        if let Some((rgb_data, width, height)) = load_png(Path::new(test_img.path)) {
            let pixels = (width * height) as u64;
            group.throughput(Throughput::Elements(pixels));

            group.bench_with_input(
                BenchmarkId::new(test_img.category, test_img.name),
                &(&rgb_data, width, height),
                |b, (data, w, h)| {
                    b.iter(|| {
                        let config = EncoderConfig::new_lossy().with_quality(75.0).with_method(4);
                        EncodeRequest::new(&config, black_box(data), PixelLayout::Rgb8, *w, *h)
                            .encode()
                            .unwrap()
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_encode_methods,
    bench_encode_quality,
    bench_encode_presets,
    bench_encode_by_image_type,
);
criterion_main!(benches);
