//! Empirical test: do tiny zenwebp-encoded images have an intrinsic ceiling
//! on achievable zensim score?
//!
//! Loads 5 representative images at each of 5 sizes (32, 48, 64, 96, 128),
//! encodes each at q ∈ {90, 92, 94, 96, 98, 100} × method ∈ {4, 6} with
//! default sns/filter/sharpness/segments, decodes back to RGB, computes
//! zensim against the source. Emits a TSV.
//!
//! Usage:
//!   cargo run --release --features analyzer --example zensim_ceiling_probe \
//!     > /tmp/zensim_ceiling_probe.tsv

#![forbid(unsafe_code)]

use std::path::PathBuf;

use zensim::{RgbSlice, Zensim, ZensimProfile};
use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};

const CORPUS: &str = "/mnt/v/output/zenwebp/picker-corpus-size-dense";

// 5 representative source IDs — mix of numeric-named photos, screenshot,
// drawing, and special-purpose chart images present in the corpus.
const IMAGES: &[&str] = &[
    "1129482",                                  // photo (numeric ID)
    "pexels-photo-3568544",                     // pexels photo
    "Beam-Space-Processing",                    // technical/diagram
    "Temperament-pie-chart-according-to-Eysenck", // chart / line-art
    "26103251787_d8635e260d_o",                 // flickr photo
];

const SIZES: &[u32] = &[32, 48, 64, 96, 128];
const Q_GRID: &[u8] = &[90, 92, 94, 96, 98, 100];
const METHODS: &[u8] = &[4, 6];

fn load_png_rgb8(path: &PathBuf) -> Option<(Vec<u8>, u32, u32)> {
    use enough::Unstoppable;
    use zenpixels_convert::PixelBufferConvertTypedExt;
    use zenpng::PngDecodeConfig;

    let bytes = std::fs::read(path).ok()?;
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

fn encode_decode_score(
    z: &Zensim,
    pre: &zensim::PrecomputedReference,
    rgb: &[u8],
    w: u32,
    h: u32,
    q: u8,
    method: u8,
) -> Option<(usize, f32)> {
    let cfg = LossyConfig::new()
        .with_quality(q as f32)
        .with_method(method);
    let req = EncodeRequest::lossy(&cfg, rgb, PixelLayout::Rgb8, w, h);
    let webp = req.encode().ok()?;
    let bytes = webp.len();

    let (rgb_dec, w2, h2) = zenwebp::oneshot::decode_rgb(&webp).ok()?;
    if w2 != w || h2 != h {
        return None;
    }
    let n = (w as usize) * (h as usize) * 3;
    let dec_chunks: &[[u8; 3]] = bytemuck::cast_slice(&rgb_dec[..n]);
    let dec_slice = RgbSlice::new(dec_chunks, w as usize, h as usize);
    let res = z.compute_with_ref(pre, &dec_slice).ok()?;
    Some((bytes, res.score() as f32))
}

fn main() {
    let z = Zensim::new(ZensimProfile::latest());

    println!("image\tw\th\tsize\tq\tmethod\tbytes\tzensim");

    for &img_id in IMAGES {
        for &size in SIZES {
            let path: PathBuf = format!("{CORPUS}/{img_id}__sz{size}.png").into();
            let (rgb, w, h) = match load_png_rgb8(&path) {
                Some(v) => v,
                None => {
                    eprintln!("[skip] missing: {}", path.display());
                    continue;
                }
            };
            let n = (w as usize) * (h as usize) * 3;
            let src_chunks: &[[u8; 3]] = bytemuck::cast_slice(&rgb[..n]);
            let src_slice = RgbSlice::new(src_chunks, w as usize, h as usize);
            let pre = match z.precompute_reference(&src_slice) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[skip] zensim precompute failed: {}: {e:?}", path.display());
                    continue;
                }
            };

            for &q in Q_GRID {
                for &m in METHODS {
                    match encode_decode_score(&z, &pre, &rgb, w, h, q, m) {
                        Some((bytes, score)) => {
                            println!(
                                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}",
                                img_id, w, h, size, q, m, bytes, score
                            );
                        }
                        None => {
                            eprintln!(
                                "[fail] {} sz{} q{} m{}",
                                img_id, size, q, m
                            );
                        }
                    }
                }
            }
        }
    }
}
