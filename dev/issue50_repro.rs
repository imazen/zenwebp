//! Issue #50 reproducer: q-monotonicity bug.
//!
//! Encode a PNG with both zenwebp and libwebp at multiple q values,
//! decode each result, measure ssim2 against the source, and print
//! a table. Used to verify the bug and isolate (zenwebp vs libwebp).
//!
//! Usage:
//!   cargo run --release --features target-zensim --example issue50_repro -- <png> [q_lo] [q_hi]
//!
//! Defaults: q_lo=70, q_hi=80.

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;
use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};

fn main() {
    let path = env::args().nth(1).expect("need path");
    let q_list: Vec<u8> = env::args()
        .skip(2)
        .filter_map(|s| s.parse().ok())
        .collect();
    let q_list = if q_list.is_empty() {
        vec![70, 75, 80, 87, 90, 95]
    } else {
        q_list
    };

    let path_buf = PathBuf::from(&path);
    let (rgb, w, h) = decode_png_rgb(&path_buf).expect("decode png");
    eprintln!("source: {} {}x{}", path, w, h);

    let z = zensim::Zensim::new(zensim::ZensimProfile::latest());
    let pre = build_pre(&z, &rgb, w, h).expect("precompute reference");

    println!("encoder\tq\tbytes\tzensim\tfast_ssim2");

    for &q in &q_list {
        // zenwebp m4
        let cfg = LossyConfig::new().with_quality(q as f32).with_method(4);
        let req = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, w, h);
        let zen_bytes = req.encode().expect("zenwebp encode");
        let (zen_zensim, zen_fastssim2) = decode_and_score(&z, &pre, &rgb, &zen_bytes, w, h);
        println!(
            "zenwebp\t{}\t{}\t{:.2}\t{:.2}",
            q,
            zen_bytes.len(),
            zen_zensim,
            zen_fastssim2
        );

        // libwebp m4 (same settings as zenwebp default: SNS=50, filter=60, segs=4 are the
        // libwebp Default Preset defaults, NOT the zero-config used in benches)
        let lib_bytes = webpx::EncoderConfig::with_preset(webpx::Preset::Default, q as f32)
            .method(4)
            .encode_rgb(&rgb, w, h, webpx::Unstoppable)
            .expect("libwebp encode");
        let (lib_zensim, lib_fastssim2) = decode_and_score(&z, &pre, &rgb, &lib_bytes, w, h);
        println!(
            "libwebp\t{}\t{}\t{:.2}\t{:.2}",
            q,
            lib_bytes.len(),
            lib_zensim,
            lib_fastssim2
        );
    }
}

fn decode_and_score(
    z: &zensim::Zensim,
    pre: &zensim::PrecomputedReference,
    src_rgb: &[u8],
    webp: &[u8],
    w: u32,
    h: u32,
) -> (f32, f32) {
    let (rgb_dec, w2, h2) = zenwebp::oneshot::decode_rgb(webp).expect("decode");
    assert_eq!(w2, w);
    assert_eq!(h2, h);
    let n = (w as usize * h as usize) * 3;
    let dec_chunks: Vec<[u8; 3]> = rgb_dec[..n]
        .chunks_exact(3)
        .map(|p| [p[0], p[1], p[2]])
        .collect();
    let dec_slice = zensim::RgbSlice::new(&dec_chunks, w as usize, h as usize);
    let zensim_score = z
        .compute_with_ref(pre, &dec_slice)
        .map(|r| r.score() as f32)
        .unwrap_or(f32::NAN);
    // fast-ssim2 (canonical SSIMULACRA2)
    let src_chunks: Vec<[u8; 3]> = src_rgb[..n]
        .chunks_exact(3)
        .map(|p| [p[0], p[1], p[2]])
        .collect();
    let src_img = imgref::ImgRef::new(&src_chunks, w as usize, h as usize);
    let dec_img = imgref::ImgRef::new(&dec_chunks, w as usize, h as usize);
    let fastssim2 = fast_ssim2::compute_ssimulacra2(src_img, dec_img)
        .map(|s| s as f32)
        .unwrap_or(f32::NAN);
    (zensim_score, fastssim2)
}

fn build_pre(z: &zensim::Zensim, rgb: &[u8], w: u32, h: u32) -> Option<zensim::PrecomputedReference> {
    let chunks: Vec<[u8; 3]> = rgb[..(w as usize * h as usize * 3)]
        .chunks_exact(3)
        .map(|p| [p[0], p[1], p[2]])
        .collect();
    let slice = zensim::RgbSlice::new(&chunks, w as usize, h as usize);
    z.precompute_reference(&slice).ok()
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
