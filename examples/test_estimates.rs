//! Test current heuristics against measured data

use zenwebp::heuristics::*;
use zenwebp::EncoderConfig;

fn main() {
    println!("=== Comparing Current Estimates vs Measured Data ===\n");

    // Test cases from our profile data
    let cases = [
        (1024, 1024, 4, false, 4), // 1MP RGBA lossy
        (1920, 1080, 4, false, 4), // 2MP RGBA lossy
        (3840, 2160, 4, false, 4), // 8MP RGBA lossy (4K)
        (1024, 1024, 4, true, 6),  // 1MP RGBA lossless
    ];

    for (width, height, bpp, lossless, method) in cases {
        let mut config = EncoderConfig::new_lossy().method(method);
        if lossless {
            config = config.lossless(true);
        }

        let est = estimate_encode(width, height, bpp, &config);
        let mpixels = (width as f64 * height as f64) / 1_000_000.0;

        println!(
            "{}x{} {} method {}:",
            width,
            height,
            if lossless { "lossless" } else { "lossy" },
            method
        );
        println!(
            "  Peak memory: {:.1} MB (range: {:.1}-{:.1} MB)",
            est.peak_memory_bytes as f64 / 1_000_000.0,
            est.peak_memory_bytes_min as f64 / 1_000_000.0,
            est.peak_memory_bytes_max as f64 / 1_000_000.0,
        );
        println!(
            "  Throughput: {:.1} Mpix/s ({:.1} ms)",
            mpixels / (est.time_ms as f64 / 1000.0),
            est.time_ms
        );
        println!();
    }

    // Measured throughput from our profile (avg):
    println!("Measured throughput (from profile data):");
    println!("  Lossy method 0: 25.7 Mpix/s");
    println!("  Lossy method 4: 14.5 Mpix/s");
    println!("  Lossy method 6: 11.1 Mpix/s");
    println!("  Lossless method 6: 221.0 Mpix/s");
    println!("  Decode (lossy): 85.5 Mpix/s");
    println!("  Decode (lossless): 351.6 Mpix/s");
}
