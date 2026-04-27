//! Cross-decode test: encode with zenwebp, decode with libwebp (and vice versa)
//! to localize the bug to encoder vs decoder side.

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let path = env::args().nth(1).expect("need src");
    let q: u8 = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(80);
    let (rgb, w, h) = decode_png_rgb(&PathBuf::from(&path)).expect("decode");
    let n = (w as usize * h as usize) * 3;
    eprintln!("source: {}x{}", w, h);

    // 1) zenwebp encode + zenwebp decode
    let zen_cfg = zenwebp::LossyConfig::new().with_quality(q as f32).with_method(4);
    let zen_webp = zenwebp::EncodeRequest::lossy(&zen_cfg, &rgb, zenwebp::PixelLayout::Rgb8, w, h)
        .encode()
        .expect("zenwebp encode");
    let (zz_rgb, _, _) = zenwebp::oneshot::decode_rgb(&zen_webp).expect("zen decode zen");

    // 2) libwebp encode + libwebp decode
    let lib_webp = webpx::EncoderConfig::with_preset(webpx::Preset::Default, q as f32)
        .method(4)
        .encode_rgb(&rgb, w, h, webpx::Unstoppable)
        .expect("libwebp encode");
    let (ll_rgb, _, _) = webpx::decode_rgb(&lib_webp).expect("libwebp decode");

    // 3) zenwebp encode + libwebp decode (cross 1)
    let (zl_rgb, _, _) = webpx::decode_rgb(&zen_webp).expect("libwebp decode of zenwebp output");

    // 4) libwebp encode + zenwebp decode (cross 2)
    let (lz_rgb, _, _) = zenwebp::oneshot::decode_rgb(&lib_webp).expect("zen decode lib");

    // Score all four.
    let src_chunks: Vec<[u8; 3]> = rgb[..n].chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
    let s_zz = score(&rgb, &zz_rgb, w, h);
    let s_ll = score(&rgb, &ll_rgb, w, h);
    let s_zl = score(&rgb, &zl_rgb, w, h);
    let s_lz = score(&rgb, &lz_rgb, w, h);

    println!("q={}", q);
    println!("zenwebp encode + zenwebp decode: ssim2={:.2}  bytes={}", s_zz, zen_webp.len());
    println!("libwebp encode + libwebp decode: ssim2={:.2}  bytes={}", s_ll, lib_webp.len());
    println!("zenwebp encode + libwebp decode: ssim2={:.2}", s_zl);
    println!("libwebp encode + zenwebp decode: ssim2={:.2}", s_lz);
    let _ = src_chunks;
}

fn score(src: &[u8], dec: &[u8], w: u32, h: u32) -> f32 {
    let n = (w as usize * h as usize) * 3;
    let s: Vec<[u8; 3]> = src[..n].chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
    let d: Vec<[u8; 3]> = dec[..n].chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
    fast_ssim2::compute_ssimulacra2(
        imgref::ImgRef::new(&s, w as usize, h as usize),
        imgref::ImgRef::new(&d, w as usize, h as usize),
    )
    .map(|s| s as f32)
    .unwrap_or(f32::NAN)
}

fn decode_png_rgb(path: &PathBuf) -> Option<(Vec<u8>, u32, u32)> {
    let bytes = fs::read(path).ok()?;
    let cur = std::io::Cursor::new(bytes);
    let mut decoder = png::Decoder::new(cur);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().ok()?;
    let info = reader.info();
    let (w, h) = (info.width, info.height);
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
            for px in buf.chunks_exact(4) { out.extend_from_slice(&[px[0], px[1], px[2]]); }
            out
        }
        png::ColorType::Grayscale => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 3);
            for &g in &buf { out.extend_from_slice(&[g, g, g]); }
            out
        }
        png::ColorType::GrayscaleAlpha => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 3);
            for px in buf.chunks_exact(2) { out.extend_from_slice(&[px[0], px[0], px[0]]); }
            out
        }
    };
    Some((rgb, w, h))
}
