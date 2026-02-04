//! Criterion benchmarks for zenwebp encoding performance.
//!
//! Tracks performance across:
//! - Methods 0-6 (speed/quality tradeoff)
//! - Quality levels (50, 75, 90)
//! - Image types (photo, screenshot, icon)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::Path;
use zenwebp::{EncoderConfig, Preset};

/// Load a PNG image and return RGB data with dimensions.
fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let rgb_data = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity((info.width * info.height * 3) as usize);
            for chunk in buf[..info.buffer_size()].chunks(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        png::ColorType::Grayscale => {
            let mut rgb = Vec::with_capacity((info.width * info.height * 3) as usize);
            for &g in &buf[..info.buffer_size()] {
                rgb.extend_from_slice(&[g, g, g]);
            }
            rgb
        }
        png::ColorType::GrayscaleAlpha => {
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

/// Test image with metadata
struct TestImage {
    name: &'static str,
    path: &'static str,
    category: &'static str,
}

const TEST_IMAGES: &[TestImage] = &[
    // Photos from CID22
    TestImage {
        name: "photo_512",
        path: "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/792079.png",
        category: "photo",
    },
    // Screenshots
    TestImage {
        name: "screenshot",
        path: "/home/lilith/work/codec-corpus/gb82-sc/gui.png",
        category: "screenshot",
    },
    // Photos from gb82
    TestImage {
        name: "flowers",
        path: "/home/lilith/work/codec-corpus/gb82/flowers-lossless.png",
        category: "photo",
    },
];

fn bench_encode_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_method");

    // Load first available test image
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
                    EncoderConfig::new()
                        .quality(75.0)
                        .method(method)
                        .encode_rgb(black_box(&rgb_data), width, height)
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
                    EncoderConfig::new()
                        .quality(quality)
                        .method(4)
                        .encode_rgb(black_box(&rgb_data), width, height)
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
                EncoderConfig::with_preset(preset, 75.0)
                    .method(4)
                    .encode_rgb(black_box(&rgb_data), width, height)
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
                        EncoderConfig::new()
                            .quality(75.0)
                            .method(4)
                            .encode_rgb(black_box(data), *w, *h)
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
