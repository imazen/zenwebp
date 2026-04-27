//! Issue #50 repro / triage: high-q encoder divergence on a single 1024x1024
//! photo. zenwebp lags libwebp by 2-4 SSIMULACRA2 points at q>=90 only on
//! `3a695d7842d66b4d_1024sq.png`; other CID22-style natural photos show
//! parity at the same q.
//!
//! Usage:
//!   cargo run --release --features target-zensim --example issue50_repro -- <png> [mode]
//!
//! mode:
//!   sweep      (default) baseline + per-mechanism toggles at q=95
//!   reproduce  baseline at q in {90,91,92,95,100}
//!   pre37      single q=95 baseline (intended to be re-run after `jj new 92d8e86^`)
//!
//! Output: TSV to stdout with columns
//!   tag, encoder, q, bytes, ssim2

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;

use fast_ssim2::{
    ColorPrimaries, Rgb as SsimRgb, TransferCharacteristic, compute_frame_ssimulacra2,
};

use zenwebp::decoder::decode_rgb;
use zenwebp::{CostModel, EncodeRequest, LossyConfig, PixelLayout, Preset};

fn load_png(path: &PathBuf) -> (Vec<u8>, u32, u32) {
    let f = fs::File::open(path).expect("open png");
    let dec = png::Decoder::new(std::io::BufReader::new(f));
    let mut r = dec.read_info().expect("read info");
    let info = r.info();
    let (w, h) = (info.width, info.height);
    let mut buf = vec![0u8; r.output_buffer_size().expect("size")];
    let frame = r.next_frame(&mut buf).expect("frame");
    let info = r.info();
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..frame.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..frame.buffer_size()]
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..frame.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        _ => panic!("unsupported color type {:?}", info.color_type),
    };
    (rgb, w, h)
}

fn srgb_to_linear(v: u8) -> f32 {
    let v = v as f32 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn ssim2(orig: &[u8], dec: &[u8], w: u32, h: u32) -> f64 {
    let to_lin = |buf: &[u8]| -> Vec<[f32; 3]> {
        buf.chunks_exact(3)
            .map(|p| [srgb_to_linear(p[0]), srgb_to_linear(p[1]), srgb_to_linear(p[2])])
            .collect()
    };
    let oi = SsimRgb::new(
        to_lin(orig),
        w as usize,
        h as usize,
        TransferCharacteristic::Linear,
        ColorPrimaries::BT709,
    )
    .unwrap();
    let di = SsimRgb::new(
        to_lin(dec),
        w as usize,
        h as usize,
        TransferCharacteristic::Linear,
        ColorPrimaries::BT709,
    )
    .unwrap();
    compute_frame_ssimulacra2(oi, di).unwrap_or(f64::NAN)
}

fn encode_zen(cfg: &LossyConfig, rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    EncodeRequest::lossy(cfg, rgb, PixelLayout::Rgb8, w, h)
        .encode()
        .expect("zen encode")
}

fn encode_lib(rgb: &[u8], w: u32, h: u32, q: f32, m: u8) -> Vec<u8> {
    webpx::EncoderConfig::new()
        .quality(q)
        .method(m)
        .encode_rgb(rgb, w, h, webpx::Unstoppable)
        .expect("lib encode")
}

fn measure_zen(tag: &str, q: f32, cfg: &LossyConfig, rgb: &[u8], w: u32, h: u32) {
    let bytes = encode_zen(cfg, rgb, w, h);
    let (dec, _, _) = decode_rgb(&bytes).expect("zen decode");
    let s = ssim2(rgb, &dec, w, h);
    println!(
        "{tag}\tzenwebp\t{q}\t{}\t{:.3}",
        bytes.len(),
        s
    );
}

fn measure_lib(tag: &str, q: f32, rgb: &[u8], w: u32, h: u32) {
    let bytes = encode_lib(rgb, w, h, q, 4);
    let (dec, _, _) = decode_rgb(&bytes).expect("zen decode");
    let s = ssim2(rgb, &dec, w, h);
    println!(
        "{tag}\tlibwebp\t{q}\t{}\t{:.3}",
        bytes.len(),
        s
    );
}

fn run_reproduce(rgb: &[u8], w: u32, h: u32) {
    println!("# tag\tencoder\tq\tbytes\tssim2");
    for q in [90.0_f32, 91.0, 92.0, 95.0, 100.0] {
        let cfg = LossyConfig::new().with_quality(q).with_method(4);
        measure_zen("baseline", q, &cfg, rgb, w, h);
        measure_lib("baseline", q, rgb, w, h);
    }
}

/// Wide q sweep used to map where the bump starts firing.
fn run_qrange(rgb: &[u8], w: u32, h: u32) {
    println!("# tag\tencoder\tq\tbytes\tssim2");
    for q in (50..=100).step_by(5) {
        let q = q as f32;
        let cfg = LossyConfig::new().with_quality(q).with_method(4);
        measure_zen("baseline", q, &cfg, rgb, w, h);
        measure_lib("baseline", q, rgb, w, h);
    }
}

fn run_sweep(rgb: &[u8], w: u32, h: u32) {
    println!("# tag\tencoder\tq\tbytes\tssim2");
    let q = 95.0_f32;

    // Baseline
    let cfg = LossyConfig::new().with_quality(q).with_method(4);
    measure_zen("baseline_m4", q, &cfg, rgb, w, h);
    measure_lib("baseline_m4", q, rgb, w, h);

    // SNS off
    let cfg = LossyConfig::new()
        .with_quality(q)
        .with_method(4)
        .with_sns_strength(0);
    measure_zen("sns0", q, &cfg, rgb, w, h);

    // StrictLibwebpParity
    let cfg = LossyConfig::new()
        .with_quality(q)
        .with_method(4)
        .with_cost_model(CostModel::StrictLibwebpParity);
    measure_zen("cost_strict", q, &cfg, rgb, w, h);

    // Method 6 (with trellis)
    let cfg = LossyConfig::new().with_quality(q).with_method(6);
    measure_zen("m6", q, &cfg, rgb, w, h);

    // Filter level 0
    let cfg = LossyConfig::new()
        .with_quality(q)
        .with_method(4)
        .with_filter_strength(0);
    measure_zen("filt0", q, &cfg, rgb, w, h);

    // Single segment
    let cfg = LossyConfig::new()
        .with_quality(q)
        .with_method(4)
        .with_segments(1);
    measure_zen("seg1", q, &cfg, rgb, w, h);

    // Photo preset
    let cfg = LossyConfig::with_preset(Preset::Photo, q).with_method(4);
    measure_zen("preset_photo", q, &cfg, rgb, w, h);

    // multi_pass_stats
    let cfg = LossyConfig::new()
        .with_quality(q)
        .with_method(4)
        .with_multi_pass_stats(true);
    measure_zen("multipass", q, &cfg, rgb, w, h);

    // Combinations
    let cfg = LossyConfig::new()
        .with_quality(q)
        .with_method(4)
        .with_sns_strength(0)
        .with_cost_model(CostModel::StrictLibwebpParity);
    measure_zen("sns0+cost_strict", q, &cfg, rgb, w, h);

    let cfg = LossyConfig::new()
        .with_quality(q)
        .with_method(4)
        .with_sns_strength(0)
        .with_segments(1);
    measure_zen("sns0+seg1", q, &cfg, rgb, w, h);

    let cfg = LossyConfig::new()
        .with_quality(q)
        .with_method(4)
        .with_segments(1)
        .with_cost_model(CostModel::StrictLibwebpParity);
    measure_zen("seg1+cost_strict", q, &cfg, rgb, w, h);

    let cfg = LossyConfig::new()
        .with_quality(q)
        .with_method(4)
        .with_sns_strength(0)
        .with_segments(1)
        .with_cost_model(CostModel::StrictLibwebpParity);
    measure_zen("sns0+seg1+cost_strict", q, &cfg, rgb, w, h);
}

fn run_filtsweep(rgb: &[u8], w: u32, h: u32) {
    println!("# tag\tencoder\tq\tbytes\tssim2");
    let q = 95.0_f32;
    for fs in [0u8, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60] {
        let cfg = LossyConfig::new()
            .with_quality(q)
            .with_method(4)
            .with_filter_strength(fs);
        let tag = format!("filt{fs}");
        measure_zen(&tag, q, &cfg, rgb, w, h);
    }
    measure_lib("libwebp_default", q, rgb, w, h);
}

fn run_pre37(rgb: &[u8], w: u32, h: u32) {
    println!("# tag\tencoder\tq\tbytes\tssim2");
    let q = 95.0_f32;
    let cfg = LossyConfig::new().with_quality(q).with_method(4);
    measure_zen("pre37_baseline_m4", q, &cfg, rgb, w, h);
    measure_lib("pre37_baseline_m4", q, rgb, w, h);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: issue50_repro <png> [reproduce|sweep|pre37]");
        std::process::exit(2);
    }
    let path = PathBuf::from(&args[1]);
    let mode = args.get(2).map(String::as_str).unwrap_or("sweep");
    let (rgb, w, h) = load_png(&path);
    match mode {
        "reproduce" => run_reproduce(&rgb, w, h),
        "sweep" => run_sweep(&rgb, w, h),
        "filtsweep" => run_filtsweep(&rgb, w, h),
        "qrange" => run_qrange(&rgb, w, h),
        "pre37" => run_pre37(&rgb, w, h),
        other => panic!("unknown mode {other}"),
    }
}
