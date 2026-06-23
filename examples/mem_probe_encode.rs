//! Encode peak-memory probe — one WebP encode, report measured peak RSS (VmHWM).
//!
//! The zenwebp ENCODE counterpart to `zenjpeg/examples/mem_probe_encode.rs`.
//! Used by the heaptrack / VmHWM sweep to calibrate the encode peak-memory
//! model (`heuristics::estimate_encode`, surfaced as the zencodec
//! `estimate_encode_resources` adapter in `src/codec.rs:446`) against measured
//! reality, *per method level* (0..6, lossy and lossless), instead of the
//! current structural guess (fixed overhead + method-interpolated bytes/pixel).
//!
//!   cargo build -p zenwebp --release --example mem_probe_encode
//!   GLIBC_TUNABLES=glibc.malloc.mmap_threshold=131072 \
//!     ./target/release/examples/mem_probe_encode <rgb8.bin> <w> <h> <lossy|lossless> <method 0..6> <quality>
//!   heaptrack ./target/release/examples/mem_probe_encode ...   # allocator peak heap
//!
//! One encode per process — peak RSS is a per-process high-water mark, so the
//! input must come from a cheap file read (raw RGB8 bin), never an in-process
//! decode (whose own peak would pollute VmHWM above the encode peak).
//!
//! TSV row:
//!   w  h  pixels  mode  method  quality
//!   out_bytes  pre_rss_kb  vmhwm_kb  marginal_kb

use zenwebp::{EncoderConfig, LosslessConfig, LossyConfig, PixelLayout};

/// A `/proc/self/status` field in KiB (e.g. `VmRSS:`, `VmHWM:`).
fn status_kb(field: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(field))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 7 {
        eprintln!(
            "usage: mem_probe_encode <rgb8.bin> <w> <h> <lossy|lossless> <method 0..6> <quality> [est]"
        );
        std::process::exit(2);
    }
    let path = &a[1];
    let w: u32 = a[2].parse().expect("w");
    let h: u32 = a[3].parse().expect("h");
    let mode = match a[4].as_str() {
        "lossy" | "lossless" => a[4].clone(),
        other => panic!("mode must be lossy|lossless, got {other}"),
    };
    let method: u8 = a[5].parse().expect("method");
    assert!(method <= 6, "method must be 0..=6, got {method}");
    let quality: f32 = a[6].parse().expect("quality");

    // Input is packed RGB8 (no alpha). The probe sweeps the RGB8 path; alpha is
    // a separate axis (RGBA8 input → VP8L alpha plane) the parent can add later
    // by feeding a w*h*4 bin and switching PixelLayout to Rgba8.
    let data = std::fs::read(path).expect("read rgb8.bin");
    assert_eq!(
        data.len(),
        (w as usize) * (h as usize) * 3,
        "bin size {} != w*h*3 {}",
        data.len(),
        (w as usize) * (h as usize) * 3
    );

    // Estimate-only mode (`est` as a 7th arg): print what the CURRENT model
    // predicts for this cell (typical peak + max), no encode — so we can
    // compare model vs measured without an encode polluting anything.
    //
    // `heuristics::estimate_encode(w, h, bpp, &EncoderConfig)` is exactly what
    // the zencodec `estimate_encode_resources` adapter calls (codec.rs:454):
    //   peak_memory_bytes      → ResourceEstimate.peak  (the recalibration target)
    //   peak_memory_bytes_max  → .with_peak_max(..)
    // bpp = 3 for RGB8 input.
    if a.get(7).map(String::as_str) == Some("est") {
        // VERIFY: `EncoderConfig` (the encoder-level enum), `new_lossy`,
        // `new_lossless`, `with_method`, `with_quality`, and the
        // `heuristics::estimate_encode` fn are all public — confirmed in
        // src/encoder/config.rs (855, 867, 875, 901, 889) + src/lib.rs:152.
        let cfg: EncoderConfig = if mode == "lossless" {
            EncoderConfig::new_lossless()
        } else {
            EncoderConfig::new_lossy()
        }
        .with_method(method)
        .with_quality(quality);

        let bpp: u8 = 3; // RGB8 input
        let est = zenwebp::heuristics::estimate_encode(w, h, bpp, &cfg);
        let pixels = (w as u64) * (h as u64);
        println!(
            "{w}\t{h}\t{pixels}\t{mode}\t{method}\t{quality}\tEST\tpeak_kb={}\tmax_kb={}\tpeak_bpp={:.2}\tmax_bpp={:.2}\tout_est_kb={}",
            est.peak_memory_bytes / 1024,
            est.peak_memory_bytes_max / 1024,
            est.peak_memory_bytes as f64 / pixels as f64,
            est.peak_memory_bytes_max as f64 / pixels as f64,
            est.output_bytes / 1024,
        );
        return;
    }

    // Baseline RSS: process + libs + the input `data` we hold. Marginal =
    // VmHWM − pre isolates the encode's own working set (what the model predicts).
    let pre = status_kb("VmRSS:");

    // Build the request for (mode, method, quality) on the RGB8 input and encode
    // once. Mirrors the api_guide.rs flow: typed config → EncodeRequest::lossy /
    // ::lossless → .encode(). RGB8 lossy converts to YUV420 during the push;
    // lossless keeps RGB (no alpha plane for RGB8 input).
    let out: Vec<u8> = if mode == "lossless" {
        let cfg = LosslessConfig::new()
            .with_method(method)
            .with_quality(quality);
        zenwebp::EncodeRequest::lossless(&cfg, &data, PixelLayout::Rgb8, w, h)
            .encode()
            .expect("lossless encode")
    } else {
        let cfg = LossyConfig::new().with_method(method).with_quality(quality);
        zenwebp::EncodeRequest::lossy(&cfg, &data, PixelLayout::Rgb8, w, h)
            .encode()
            .expect("lossy encode")
    };

    // High-water mark immediately after encode — VmHWM is monotonic, so it
    // reflects the peak *during* the encode.
    let peak = status_kb("VmHWM:");

    let pixels = (w as u64) * (h as u64);
    println!(
        "{w}\t{h}\t{pixels}\t{mode}\t{method}\t{quality}\t{}\t{pre}\t{peak}\t{}",
        out.len(),
        peak.saturating_sub(pre)
    );
    std::hint::black_box(&out);
}
