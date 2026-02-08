//! Analyze memory profile data and generate estimation formulas.
//!
//! Usage:
//!   cargo run --release --example analyze_profiles encode_profile.csv decode_profile.csv

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <encode_profile.csv> <decode_profile.csv>",
            args[0]
        );
        std::process::exit(1);
    }

    println!("=== Memory Profile Analysis ===\n");

    analyze_encode(&args[1]);
    println!();
    analyze_decode(&args[2]);
}

#[derive(Debug)]
#[allow(dead_code)]
struct EncodeRow {
    pixels: u64,
    method: u8,
    lossless: bool,
    peak_rss_kb: u64,
    time_us: u64,
    output_size: u64,
}

fn analyze_encode(path: &str) {
    println!("## Encoding Analysis");
    println!("Data file: {}\n", path);

    let file = File::open(path).expect("Failed to open encode profile");
    let reader = BufReader::new(file);

    let mut rows = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }

        let line = line.expect("Failed to read line");
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 13 {
            continue;
        }

        let pixels = parts[2].parse().unwrap_or(0);
        let method = parts[5].parse().unwrap_or(0);
        let lossless = parts[7] == "true";
        let output_size = parts[9].parse().unwrap_or(0);
        let peak_rss_kb = parts[10].parse().unwrap_or(0);
        let time_us = parts[12].parse().unwrap_or(0);

        rows.push(EncodeRow {
            pixels,
            method,
            lossless,
            peak_rss_kb,
            time_us,
            output_size,
        });
    }

    println!("Total samples: {}", rows.len());

    // Analyze lossy encoding
    let lossy: Vec<_> = rows.iter().filter(|r| !r.lossless).collect();
    analyze_memory_formula("Lossy", &lossy);

    // Analyze lossless encoding
    let lossless: Vec<_> = rows.iter().filter(|r| r.lossless).collect();
    analyze_memory_formula("Lossless", &lossless);

    // Analyze by method
    for method in 0..=6 {
        let method_rows: Vec<_> = lossy
            .iter()
            .filter(|r| r.method == method)
            .copied()
            .collect();
        if !method_rows.is_empty() {
            analyze_throughput(&format!("Lossy method {}", method), &method_rows);
        }
    }

    for method in [0, 3, 6] {
        let method_rows: Vec<_> = lossless
            .iter()
            .filter(|r| r.method == method)
            .copied()
            .collect();
        if !method_rows.is_empty() {
            analyze_throughput(&format!("Lossless method {}", method), &method_rows);
        }
    }
}

fn analyze_memory_formula(name: &str, rows: &[&EncodeRow]) {
    if rows.is_empty() {
        return;
    }

    println!("\n### {} Encoding", name);

    // Linear regression: peak_memory = fixed + bytes_per_pixel * pixels
    // Using least squares: find (fixed, bpp) that minimizes sum of squared errors

    let n = rows.len() as f64;
    let sum_pixels: f64 = rows.iter().map(|r| r.pixels as f64).sum();
    let sum_mem: f64 = rows.iter().map(|r| (r.peak_rss_kb * 1024) as f64).sum();
    let sum_pixels_sq: f64 = rows.iter().map(|r| (r.pixels as f64).powi(2)).sum();
    let sum_pixels_mem: f64 = rows
        .iter()
        .map(|r| r.pixels as f64 * (r.peak_rss_kb * 1024) as f64)
        .sum();

    let denom = n * sum_pixels_sq - sum_pixels.powi(2);
    if denom.abs() < 1e-10 {
        println!("  Insufficient data for regression");
        return;
    }

    let bpp = (n * sum_pixels_mem - sum_pixels * sum_mem) / denom;
    let fixed = (sum_mem - bpp * sum_pixels) / n;

    println!("  Samples: {}", rows.len());
    println!(
        "  Formula: peak_memory = {:.0} + {:.2} * pixels",
        fixed, bpp
    );
    println!("  Fixed overhead: {:.1} KB", fixed / 1024.0);
    println!("  Bytes per pixel: {:.2}", bpp);

    // Calculate R² (coefficient of determination)
    let mean_mem = sum_mem / n;
    let ss_tot: f64 = rows
        .iter()
        .map(|r| ((r.peak_rss_kb * 1024) as f64 - mean_mem).powi(2))
        .sum();
    let ss_res: f64 = rows
        .iter()
        .map(|r| {
            let pred = fixed + bpp * r.pixels as f64;
            let actual = (r.peak_rss_kb * 1024) as f64;
            (actual - pred).powi(2)
        })
        .sum();

    let r_squared = 1.0 - (ss_res / ss_tot);
    println!("  R²: {:.4}", r_squared);

    // Show method-based variation
    let mut method_stats: Vec<(u8, Vec<f64>)> = vec![];
    for method in 0..=6 {
        let bpps: Vec<f64> = rows
            .iter()
            .filter(|r| r.method == method)
            .map(|r| ((r.peak_rss_kb * 1024) as f64 - fixed) / r.pixels as f64)
            .collect();

        if !bpps.is_empty() {
            method_stats.push((method, bpps));
        }
    }

    if method_stats.len() > 1 {
        println!("\n  Bytes/pixel by method:");
        for (method, bpps) in method_stats {
            let avg: f64 = bpps.iter().sum::<f64>() / bpps.len() as f64;
            let min = bpps.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = bpps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "    Method {}: {:.2} (range: {:.2}-{:.2})",
                method, avg, min, max
            );
        }
    }
}

fn analyze_throughput(name: &str, rows: &[&EncodeRow]) {
    if rows.is_empty() {
        return;
    }

    let throughputs: Vec<f64> = rows
        .iter()
        .map(|r| {
            let mpixels = r.pixels as f64 / 1_000_000.0;
            let seconds = r.time_us as f64 / 1_000_000.0;
            mpixels / seconds
        })
        .collect();

    let avg = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
    let min = throughputs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = throughputs
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("\n### {} Throughput", name);
    println!("  Average: {:.1} Mpix/s", avg);
    println!("  Range: {:.1} - {:.1} Mpix/s", min, max);
}

#[derive(Debug)]
struct DecodeRow {
    pixels: u64,
    is_lossless: bool,
    peak_rss_kb: u64,
    time_us: u64,
}

fn analyze_decode(path: &str) {
    println!("## Decoding Analysis");
    println!("Data file: {}\n", path);

    let file = File::open(path).expect("Failed to open decode profile");
    let reader = BufReader::new(file);

    let mut rows = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }

        let line = line.expect("Failed to read line");
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 8 {
            continue;
        }

        let pixels = parts[2].parse().unwrap_or(0);
        let is_lossless = parts[4] == "true";
        let peak_rss_kb = parts[5].parse().unwrap_or(0);
        let time_us = parts[7].parse().unwrap_or(0);

        rows.push(DecodeRow {
            pixels,
            is_lossless,
            peak_rss_kb,
            time_us,
        });
    }

    println!("Total samples: {}", rows.len());

    // Memory formula
    let n = rows.len() as f64;
    let sum_pixels: f64 = rows.iter().map(|r| r.pixels as f64).sum();
    let sum_mem: f64 = rows.iter().map(|r| (r.peak_rss_kb * 1024) as f64).sum();
    let sum_pixels_sq: f64 = rows.iter().map(|r| (r.pixels as f64).powi(2)).sum();
    let sum_pixels_mem: f64 = rows
        .iter()
        .map(|r| r.pixels as f64 * (r.peak_rss_kb * 1024) as f64)
        .sum();

    let denom = n * sum_pixels_sq - sum_pixels.powi(2);
    if denom.abs() > 1e-10 {
        let bpp = (n * sum_pixels_mem - sum_pixels * sum_mem) / denom;
        let fixed = (sum_mem - bpp * sum_pixels) / n;

        println!("\n### Decode Memory");
        println!("  Samples: {}", rows.len());
        println!(
            "  Formula: peak_memory = {:.0} + {:.2} * pixels",
            fixed, bpp
        );
        println!("  Fixed overhead: {:.1} KB", fixed / 1024.0);
        println!("  Bytes per pixel: {:.2}", bpp);
    }

    // Throughput
    let throughputs: Vec<f64> = rows
        .iter()
        .map(|r| {
            let mpixels = r.pixels as f64 / 1_000_000.0;
            let seconds = r.time_us as f64 / 1_000_000.0;
            mpixels / seconds
        })
        .collect();

    let avg = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
    let min = throughputs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = throughputs
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("\n### Decode Throughput");
    println!("  Average: {:.1} Mpix/s", avg);
    println!("  Range: {:.1} - {:.1} Mpix/s", min, max);

    // By type
    let lossy: Vec<_> = rows.iter().filter(|r| !r.is_lossless).collect();
    let lossless: Vec<_> = rows.iter().filter(|r| r.is_lossless).collect();

    if !lossy.is_empty() {
        let avg: f64 = lossy
            .iter()
            .map(|r| (r.pixels as f64 / 1_000_000.0) / (r.time_us as f64 / 1_000_000.0))
            .sum::<f64>()
            / lossy.len() as f64;
        println!("  Lossy: {:.1} Mpix/s", avg);
    }

    if !lossless.is_empty() {
        let avg: f64 = lossless
            .iter()
            .map(|r| (r.pixels as f64 / 1_000_000.0) / (r.time_us as f64 / 1_000_000.0))
            .sum::<f64>()
            / lossless.len() as f64;
        println!("  Lossless: {:.1} Mpix/s", avg);
    }
}
