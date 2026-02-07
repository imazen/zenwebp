//! Memory profiling harness for building accurate estimation formulas.
//!
//! This program generates test images of various sizes and formats,
//! encodes/decodes them while measuring actual memory usage,
//! and outputs CSV data for analysis.
//!
//! Usage:
//!   cargo build --release --example profile_memory
//!   target/release/examples/profile_memory encode > encode_profile.csv
//!   target/release/examples/profile_memory decode > decode_profile.csv
//!
//! Then analyze with:
//!   cargo run --release --example analyze_profiles

use std::env;
use std::fs;
use std::time::Instant;
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout, Preset};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <encode|decode>", args[0]);
        std::process::exit(1);
    }

    match args[1].as_str() {
        "encode" => profile_encode(),
        "decode" => profile_decode(),
        _ => {
            eprintln!("Unknown mode: {}", args[1]);
            std::process::exit(1);
        }
    }
}

fn profile_encode() {
    // CSV header
    println!("width,height,pixels,bpp,format,method,quality,lossless,preset,output_size,peak_rss_kb,peak_vm_kb,time_us");

    // Test dimensions (representative sample from 64x64 to 4K)
    let sizes = [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (768, 768),
        (1024, 1024),
        (1280, 720),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
    ];

    // Test formats
    let formats = [
        (PixelLayout::Rgb8, 3, "rgb"),
        (PixelLayout::Rgba8, 4, "rgba"),
    ];

    // Test methods for lossy
    let methods = [0, 2, 4, 6];

    // Test qualities
    let qualities = [50.0, 75.0, 90.0];

    // Test presets
    let presets = [
        Preset::Default,
        Preset::Photo,
        Preset::Drawing,
        Preset::Icon,
        Preset::Text,
    ];

    for (width, height) in sizes {
        let pixels = width * height;

        for (layout, bpp, format_name) in &formats {
            // Lossy encoding
            for &method in &methods {
                for &quality in &qualities {
                    force_gc();
                    let img = generate_test_image(width, height, *bpp);

                    let mem_before = get_memory_usage();
                    let start = Instant::now();
                    let config = EncoderConfig::new().quality(quality).method(method);

                    if let Ok(output) = EncodeRequest::new(&config, &img, *layout, width, height)
                        .encode()
                    {
                        let elapsed_us = start.elapsed().as_micros();
                        let mem_after = get_memory_usage();
                        let peak_rss = mem_after.rss_kb.saturating_sub(mem_before.rss_kb);
                        let peak_vm = mem_after.vm_kb.saturating_sub(mem_before.vm_kb);

                        println!(
                            "{},{},{},{},{},{},{},false,none,{},{},{},{}",
                            width,
                            height,
                            pixels,
                            bpp,
                            format_name,
                            method,
                            quality,
                            output.len(),
                            peak_rss,
                            peak_vm,
                            elapsed_us
                        );
                    }
                    drop(img);
                }
            }

            // Lossless encoding
            for &method in &[0, 3, 6] {
                for &quality in &[50.0, 75.0, 90.0, 100.0] {
                    force_gc();
                    let img = generate_test_image(width, height, *bpp);

                    let mem_before = get_memory_usage();
                    let start = Instant::now();
                    let config = EncoderConfig::new_lossless()
                        .quality(quality)
                        .method(method);

                    if let Ok(output) =
                        EncodeRequest::new(&config, &img, *layout, width, height).encode()
                    {
                        let elapsed_us = start.elapsed().as_micros();
                        let mem_after = get_memory_usage();
                        let peak_rss = mem_after.rss_kb.saturating_sub(mem_before.rss_kb);
                        let peak_vm = mem_after.vm_kb.saturating_sub(mem_before.vm_kb);

                        println!(
                            "{},{},{},{},{},{},{},true,none,{},{},{},{}",
                            width,
                            height,
                            pixels,
                            bpp,
                            format_name,
                            method,
                            quality,
                            output.len(),
                            peak_rss,
                            peak_vm,
                            elapsed_us
                        );
                    }
                    drop(img);
                }
            }

            // Preset-based encoding
            for preset in &presets {
                force_gc();
                let img = generate_test_image(width, height, *bpp);

                let mem_before = get_memory_usage();
                let start = Instant::now();
                let config = EncoderConfig::with_preset(*preset, 75.0);

                if let Ok(output) =
                    EncodeRequest::new(&config, &img, *layout, width, height).encode()
                {
                    let elapsed_us = start.elapsed().as_micros();
                    let mem_after = get_memory_usage();
                    let peak_rss = mem_after.rss_kb.saturating_sub(mem_before.rss_kb);
                    let peak_vm = mem_after.vm_kb.saturating_sub(mem_before.vm_kb);

                    println!(
                        "{},{},{},{},{},75,false,{:?},{},{},{},{}",
                        width,
                        height,
                        pixels,
                        bpp,
                        format_name,
                        preset,
                        output.len(),
                        peak_rss,
                        peak_vm,
                        elapsed_us
                    );
                }
                drop(img);
            }
        }
    }
}

fn profile_decode() {
    // CSV header
    println!("width,height,pixels,input_size,is_lossless,peak_rss_kb,peak_vm_kb,time_us");

    let sizes = [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (768, 768),
        (1024, 1024),
        (1280, 720),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
    ];

    for (width, height) in sizes {
        let pixels = width * height;

        // Lossy test image
        let img = generate_test_image(width, height, 4);
        let config = EncoderConfig::new().quality(75.0).method(4);

        if let Ok(webp) =
            EncodeRequest::new(&config, &img, PixelLayout::Rgba8, width, height).encode()
        {
            force_gc();
            let mem_before = get_memory_usage();
            let start = Instant::now();

            if let Ok((_, w, h)) = zenwebp::decode_rgba(&webp) {
                let elapsed_us = start.elapsed().as_micros();
                let mem_after = get_memory_usage();
                let peak_rss = mem_after.rss_kb.saturating_sub(mem_before.rss_kb);
                let peak_vm = mem_after.vm_kb.saturating_sub(mem_before.vm_kb);

                println!(
                    "{},{},{},{},false,{},{},{}",
                    w, h, pixels, webp.len(), peak_rss, peak_vm, elapsed_us
                );
            }
        }

        // Lossless test image
        let config = EncoderConfig::new_lossless();
        if let Ok(webp) =
            EncodeRequest::new(&config, &img, PixelLayout::Rgba8, width, height).encode()
        {
            force_gc();
            let mem_before = get_memory_usage();
            let start = Instant::now();

            if let Ok((_, w, h)) = zenwebp::decode_rgba(&webp) {
                let elapsed_us = start.elapsed().as_micros();
                let mem_after = get_memory_usage();
                let peak_rss = mem_after.rss_kb.saturating_sub(mem_before.rss_kb);
                let peak_vm = mem_after.vm_kb.saturating_sub(mem_before.vm_kb);

                println!(
                    "{},{},{},{},true,{},{},{}",
                    w, h, pixels, webp.len(), peak_rss, peak_vm, elapsed_us
                );
            }
        }
    }
}

/// Generate a test image with representative content.
/// Uses a gradient pattern to ensure moderate compression (not solid color, not noise).
fn generate_test_image(width: u32, height: u32, bpp: usize) -> Vec<u8> {
    let mut img = vec![0u8; (width * height) as usize * bpp];

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) as usize) * bpp;

            // Gradient + some structured pattern
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = ((x + y) * 127 / (width + height)) as u8;

            img[idx] = r;
            if bpp > 1 {
                img[idx + 1] = g;
            }
            if bpp > 2 {
                img[idx + 2] = b;
            }
            if bpp > 3 {
                img[idx + 3] = 255; // alpha
            }
        }
    }

    img
}

#[derive(Debug, Clone, Copy)]
struct MemoryUsage {
    rss_kb: u64,
    vm_kb: u64,
}

/// Get current memory usage from /proc/self/status
fn get_memory_usage() -> MemoryUsage {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            let mut rss_kb = 0;
            let mut vm_kb = 0;

            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    rss_kb = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                } else if line.starts_with("VmSize:") {
                    vm_kb = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                }
            }

            return MemoryUsage { rss_kb, vm_kb };
        }
    }

    MemoryUsage { rss_kb: 0, vm_kb: 0 }
}

/// Force a garbage collection pass (best effort)
fn force_gc() {
    // Rust doesn't have explicit GC, but we can encourage deallocation
    // by doing some allocations and drops
    let _dummy: Vec<u8> = Vec::with_capacity(1024 * 1024);
    drop(_dummy);

    // Sleep briefly to let the allocator settle
    std::thread::sleep(std::time::Duration::from_millis(10));
}
