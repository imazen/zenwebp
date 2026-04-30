//! Generate size-dense training images: 20 representative images each
//! resized to up to 20 log-spaced sizes via Lanczos. Used to expand
//! training-corpus coverage along the size axis without re-encoding
//! the full 250-image set.
//!
//! Reads image paths from --list <file>, writes resized PNGs to
//! --out-dir, one per (source, size) pair. Skips upscaling — only
//! produces variants where target_max <= source_max.
//!
//! Usage:
//!   cargo run --release --features analyzer --example build_size_dense_corpus -- \
//!     --list /tmp/zenwebp_repr_images.txt \
//!     --out-dir /mnt/v/output/zenwebp/picker-corpus-size-dense

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use enough::Unstoppable;
use rayon::prelude::*;
use zenpixels_convert::PixelBufferConvertTypedExt;
use zenpng::PngDecodeConfig;

// 20 log-spaced sizes covering tiny → large.
const SIZES: &[u32] = &[
    32, 40, 48, 64, 80, 96, 128, 160, 192, 256,
    320, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 4096,
];

fn parse_args() -> (PathBuf, PathBuf) {
    let mut list = PathBuf::new();
    let mut out_dir = PathBuf::new();
    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--list" => list = PathBuf::from(it.next().unwrap()),
            "--out-dir" => out_dir = PathBuf::from(it.next().unwrap()),
            other => panic!("unknown arg: {other}"),
        }
    }
    if list.as_os_str().is_empty() || out_dir.as_os_str().is_empty() {
        eprintln!("usage: build_size_dense_corpus --list <file> --out-dir <dir>");
        std::process::exit(2);
    }
    (list, out_dir)
}

fn load_png_rgb8(path: &PathBuf) -> Option<(Vec<u8>, u32, u32)> {
    let bytes = fs::read(path).ok()?;
    let output = zenpng::decode(&bytes, &PngDecodeConfig::default(), &Unstoppable).ok()?;
    let w = output.info.width;
    let h = output.info.height;
    let rgb_buf = output.pixels.to_rgb8();
    let slice = rgb_buf.as_slice();
    if let Some(contiguous) = slice.as_contiguous_bytes() {
        return Some((contiguous.to_vec(), w, h));
    }
    let mut out = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for y in 0..h {
        let row = slice.row(y);
        out.extend_from_slice(&row[..(w as usize) * 3]);
    }
    Some((out, w, h))
}

fn resize_to(rgb: &[u8], w: u32, h: u32, target_max: u32) -> Option<(Vec<u8>, u32, u32)> {
    let src_max = w.max(h);
    if target_max >= src_max {
        // No upscaling — return None to skip this size variant.
        return None;
    }
    let scale = target_max as f32 / src_max as f32;
    let new_w = ((w as f32 * scale).round() as u32).max(1);
    let new_h = ((h as f32 * scale).round() as u32).max(1);
    let cfg = zenresize::ResizeConfig::builder(w, h, new_w, new_h)
        .format(zenresize::PixelDescriptor::RGB8_SRGB)
        .filter(zenresize::Filter::Mitchell)
        .srgb()
        .build();
    let resized = zenresize::Resizer::new(&cfg).resize(rgb);
    Some((resized, new_w, new_h))
}

fn save_png_rgb8(path: &PathBuf, rgb: &[u8], w: u32, h: u32) -> Result<(), String> {
    let f = fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let buf = std::io::BufWriter::new(f);
    let mut encoder = png::Encoder::new(buf, w, h);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| format!("png header: {e}"))?;
    writer.write_image_data(rgb).map_err(|e| format!("png write: {e}"))?;
    Ok(())
}

fn main() {
    let (list, out_dir) = parse_args();
    fs::create_dir_all(&out_dir).expect("create out_dir");

    let paths: Vec<PathBuf> = fs::read_to_string(&list)
        .expect("read --list")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(PathBuf::from)
        .collect();

    eprintln!(
        "[size_dense] {} source images × up to {} sizes (log-spaced 32..4096)",
        paths.len(),
        SIZES.len()
    );
    eprintln!("[size_dense] out: {}", out_dir.display());

    let started = Instant::now();
    let total_written: usize = paths.par_iter().map(|src| {
        let (rgb, w, h) = match load_png_rgb8(src) {
            Some(t) => t,
            None => {
                eprintln!("  load failed: {}", src.display());
                return 0;
            }
        };
        let stem = src.file_stem().unwrap().to_string_lossy().to_string();
        let mut written = 0;
        for &target in SIZES {
            match resize_to(&rgb, w, h, target) {
                Some((rgb_r, w_r, h_r)) => {
                    let dst = out_dir.join(format!("{stem}__sz{target}.png"));
                    if let Err(e) = save_png_rgb8(&dst, &rgb_r, w_r, h_r) {
                        eprintln!("  save failed {}: {e}", dst.display());
                    } else {
                        written += 1;
                    }
                }
                None => {} // upscale skipped
            }
        }
        eprintln!(
            "  {} ({}x{}) → {} variants",
            src.file_name().unwrap().to_string_lossy(),
            w, h, written
        );
        written
    }).sum();

    eprintln!(
        "[size_dense] wrote {} variants in {:.1}s",
        total_written,
        started.elapsed().as_secs_f64()
    );
}
