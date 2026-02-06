//! Criterion benchmarks comparing zenwebp vs libwebp encoding performance.
//!
//! Uses webpx crate (safe Rust wrapper around libwebp) for fair library-to-library
//! comparison — both encoders receive identical in-memory RGB data with matched settings.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
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

fn bench_methods_diagnostic(c: &mut Criterion) {
    let mut group = c.benchmark_group("method_diagnostic");

    let (rgb_data, width, height) =
        load_png(Path::new(DEFAULT_IMAGE)).expect("Test image not found");

    let pixels = (width * height) as u64;
    group.throughput(Throughput::Elements(pixels));

    // Diagnostic settings: SNS=0, filter=0, segments=1
    // Isolates encoder core from preprocessing
    for method in [0, 2, 4, 6] {
        group.bench_with_input(
            BenchmarkId::new("zenwebp", method),
            &method,
            |b, &method| {
                b.iter(|| {
                    EncoderConfig::new()
                        .quality(75.0)
                        .method(method)
                        .sns_strength(0)
                        .filter_strength(0)
                        .segments(1)
                        .encode_rgb(black_box(&rgb_data), width, height)
                        .unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("libwebp", method),
            &method,
            |b, &method| {
                b.iter(|| {
                    webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
                        .method(method)
                        .sns_strength(0)
                        .filter_strength(0)
                        .filter_sharpness(0)
                        .segments(1)
                        .encode_rgb(black_box(&rgb_data), width, height, webpx::Unstoppable)
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_methods_default(c: &mut Criterion) {
    let mut group = c.benchmark_group("method_default");

    let (rgb_data, width, height) =
        load_png(Path::new(DEFAULT_IMAGE)).expect("Test image not found");

    let pixels = (width * height) as u64;
    group.throughput(Throughput::Elements(pixels));

    // Default settings (SNS=50, filter=60, segments=4)
    // Production-realistic comparison
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

        group.bench_with_input(
            BenchmarkId::new("libwebp", method),
            &method,
            |b, &method| {
                b.iter(|| {
                    webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
                        .method(method)
                        .encode_rgb(black_box(&rgb_data), width, height, webpx::Unstoppable)
                        .unwrap()
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

    for quality in [50.0f32, 75.0, 90.0] {
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

        group.bench_with_input(
            BenchmarkId::new("libwebp", quality as u32),
            &quality,
            |b, &quality| {
                b.iter(|| {
                    webpx::EncoderConfig::with_preset(webpx::Preset::Default, quality)
                        .method(4)
                        .encode_rgb(black_box(&rgb_data), width, height, webpx::Unstoppable)
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_methods_diagnostic,
    bench_methods_default,
    bench_quality_comparison,
);
criterion_main!(benches);
