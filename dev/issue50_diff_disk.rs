//! Diff zenwebp's fresh-decoded q80 vs the on-disk q80.png from the issue's
//! ledger to determine if the decoded pixels differ.
//!
//! Usage:
//!   cargo run --release --features target-zensim --example issue50_diff_disk \
//!     -- <source.png> <ledger_decoded.png> <q>

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;
use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};

fn main() {
    let src = env::args().nth(1).expect("need src");
    let disk = env::args().nth(2).expect("need disk decoded");
    let q: u8 = env::args().nth(3).and_then(|s| s.parse().ok()).expect("need q");

    let (rgb_src, w, h) = decode_png_rgb(&PathBuf::from(&src)).expect("src decode");
    eprintln!("source: {}x{}", w, h);

    // Encode with zenwebp m4 default and decode.
    let cfg = LossyConfig::new().with_quality(q as f32).with_method(4);
    let req = EncodeRequest::lossy(&cfg, &rgb_src, PixelLayout::Rgb8, w, h);
    let webp = req.encode().expect("zenwebp encode");
    eprintln!("fresh webp bytes: {}", webp.len());
    let (rgb_fresh, fw, fh) = zenwebp::oneshot::decode_rgb(&webp).expect("fresh decode");
    assert_eq!((fw, fh), (w, h));

    // Save fresh webp to /tmp for comparison, and decode the on-disk PNG ref.
    fs::write("/tmp/issue50_fresh.webp", &webp).expect("write webp");

    let (rgb_disk, dw, dh) = decode_png_rgb(&PathBuf::from(&disk)).expect("disk decode");
    eprintln!("disk PNG: {}x{}", dw, dh);
    if (dw, dh) != (w, h) {
        eprintln!("DIMENSION MISMATCH disk={}x{} src={}x{}", dw, dh, w, h);
        return;
    }

    let n = (w as usize * h as usize) * 3;
    let mut max_diff: i32 = 0;
    let mut sum_sq_diff: u64 = 0;
    let mut n_diff = 0u64;
    for i in 0..n {
        let d = (rgb_fresh[i] as i32 - rgb_disk[i] as i32).abs();
        if d != 0 {
            n_diff += 1;
        }
        if d > max_diff {
            max_diff = d;
        }
        sum_sq_diff += (d * d) as u64;
    }
    let mse = sum_sq_diff as f64 / n as f64;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    };
    println!(
        "fresh-vs-disk: max_diff={} n_diff={}/{} ({:.2}%) mse={:.2} psnr={:.2}",
        max_diff,
        n_diff,
        n,
        100.0 * n_diff as f64 / n as f64,
        mse,
        psnr,
    );

    // ssim2: source vs fresh-decoded, vs source vs disk-decoded.
    let src_chunks: Vec<[u8; 3]> = rgb_src[..n].chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
    let fresh_chunks: Vec<[u8; 3]> = rgb_fresh[..n].chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
    let disk_chunks: Vec<[u8; 3]> = rgb_disk[..n].chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
    let src_img = imgref::ImgRef::new(&src_chunks, w as usize, h as usize);
    let fresh_img = imgref::ImgRef::new(&fresh_chunks, w as usize, h as usize);
    let disk_img = imgref::ImgRef::new(&disk_chunks, w as usize, h as usize);
    let s_fresh = fast_ssim2::compute_ssimulacra2(src_img, fresh_img).unwrap();
    let s_disk = fast_ssim2::compute_ssimulacra2(src_img, disk_img).unwrap();
    println!("ssim2(src,fresh)={:.2} ssim2(src,disk)={:.2}", s_fresh, s_disk);
}

fn decode_png_rgb(path: &PathBuf) -> Option<(Vec<u8>, u32, u32)> {
    let bytes = fs::read(path).ok()?;
    let cur = std::io::Cursor::new(bytes);
    let mut decoder = png::Decoder::new(cur);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().ok()?;
    let info = reader.info();
    let w = info.width;
    let h = info.height;
    let color = info.color_type;
    let bytes_per_pixel = match color {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Indexed => 3,
    };
    let mut buf = vec![0u8; (w as usize * h as usize) * bytes_per_pixel];
    reader.next_frame(&mut buf).ok()?;
    let rgb = match color {
        png::ColorType::Rgb | png::ColorType::Indexed => buf,
        png::ColorType::Rgba => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 3);
            for px in buf.chunks_exact(4) {
                out.extend_from_slice(&[px[0], px[1], px[2]]);
            }
            out
        }
        png::ColorType::Grayscale => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 3);
            for &g in &buf {
                out.extend_from_slice(&[g, g, g]);
            }
            out
        }
        png::ColorType::GrayscaleAlpha => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 3);
            for px in buf.chunks_exact(2) {
                out.extend_from_slice(&[px[0], px[0], px[0]]);
            }
            out
        }
    };
    Some((rgb, w, h))
}
