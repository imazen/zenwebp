//! Criterion benchmarks comparing zenwebp vs libwebp encoding performance.
//!
//! Tracks performance parity across methods 0-6 and quality levels.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::Path;
use zenwebp::EncoderConfig;

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

/// Default test image path
const DEFAULT_IMAGE: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/792079.png";

fn bench_methods_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("method_comparison");

    let (rgb_data, width, height) =
        load_png(Path::new(DEFAULT_IMAGE)).expect("Test image not found");

    let pixels = (width * height) as u64;
    group.throughput(Throughput::Elements(pixels));

    // Test methods 0, 2, 4, 6 (representative of speed range)
    for method in [0, 2, 4, 6] {
        // zenwebp
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

        // libwebp
        group.bench_with_input(
            BenchmarkId::new("libwebp", method),
            &method,
            |b, &method| {
                let mut config = webp::WebPConfig::new().unwrap();
                config.quality = 75.0;
                config.method = method as i32;

                b.iter(|| {
                    let encoder = webp::Encoder::from_rgb(black_box(&rgb_data), width, height);
                    encoder.encode_advanced(&config).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_quality_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_comparison");

    let (rgb_data, width, height) =
        load_png(Path::new(DEFAULT_IMAGE)).expect("Test image not found");

    let pixels = (width * height) as u64;
    group.throughput(Throughput::Elements(pixels));

    // Test quality 50, 75, 90 at method 4
    for quality in [50.0, 75.0, 90.0] {
        // zenwebp
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

        // libwebp
        group.bench_with_input(
            BenchmarkId::new("libwebp", quality as u32),
            &quality,
            |b, &quality| {
                let mut config = webp::WebPConfig::new().unwrap();
                config.quality = quality as f32;
                config.method = 4;

                b.iter(|| {
                    let encoder = webp::Encoder::from_rgb(black_box(&rgb_data), width, height);
                    encoder.encode_advanced(&config).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_overall_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("overall_throughput");

    let (rgb_data, width, height) =
        load_png(Path::new(DEFAULT_IMAGE)).expect("Test image not found");

    let pixels = (width * height) as u64;
    group.throughput(Throughput::Elements(pixels));

    // Default settings (method 4, quality 75)
    group.bench_function("zenwebp_default", |b| {
        b.iter(|| {
            EncoderConfig::new()
                .quality(75.0)
                .method(4)
                .encode_rgb(black_box(&rgb_data), width, height)
                .unwrap()
        });
    });

    group.bench_function("libwebp_default", |b| {
        let mut config = webp::WebPConfig::new().unwrap();
        config.quality = 75.0;
        config.method = 4;

        b.iter(|| {
            let encoder = webp::Encoder::from_rgb(black_box(&rgb_data), width, height);
            encoder.encode_advanced(&config).unwrap()
        });
    });

    // Fast settings (method 0)
    group.bench_function("zenwebp_fast", |b| {
        b.iter(|| {
            EncoderConfig::new()
                .quality(75.0)
                .method(0)
                .encode_rgb(black_box(&rgb_data), width, height)
                .unwrap()
        });
    });

    group.bench_function("libwebp_fast", |b| {
        let mut config = webp::WebPConfig::new().unwrap();
        config.quality = 75.0;
        config.method = 0;

        b.iter(|| {
            let encoder = webp::Encoder::from_rgb(black_box(&rgb_data), width, height);
            encoder.encode_advanced(&config).unwrap()
        });
    });

    // Best quality settings (method 6)
    group.bench_function("zenwebp_best", |b| {
        b.iter(|| {
            EncoderConfig::new()
                .quality(75.0)
                .method(6)
                .encode_rgb(black_box(&rgb_data), width, height)
                .unwrap()
        });
    });

    group.bench_function("libwebp_best", |b| {
        let mut config = webp::WebPConfig::new().unwrap();
        config.quality = 75.0;
        config.method = 6;

        b.iter(|| {
            let encoder = webp::Encoder::from_rgb(black_box(&rgb_data), width, height);
            encoder.encode_advanced(&config).unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_methods_comparison,
    bench_quality_comparison,
    bench_overall_throughput,
);
criterion_main!(benches);
