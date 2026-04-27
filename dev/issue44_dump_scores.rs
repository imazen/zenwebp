//! Dump zensim regression matrix scores so we can recalibrate floors after
//! issue #44 fix (D > min_disto gate). Mirrors `tests/zensim_regression_matrix.rs`
//! but writes a TSV instead of asserting against floors.
//!
//! Usage:
//!   cargo run --release --example issue44_dump_scores > /tmp/issue44_scores.tsv

#![forbid(unsafe_code)]

use zensim::{RgbaSlice, Zensim, ZensimProfile};
use zensim_regress::generators;
use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};

const W: u32 = 96;
const H: u32 = 96;

type ImgGen = fn(u32, u32) -> Vec<u8>;

fn gen_gradient(w: u32, h: u32) -> Vec<u8> {
    generators::gradient(w, h)
}
fn gen_noise(w: u32, h: u32) -> Vec<u8> {
    generators::value_noise(w, h, 17)
}
fn gen_color_blocks(w: u32, h: u32) -> Vec<u8> {
    generators::color_blocks(w, h)
}
fn gen_mandelbrot(w: u32, h: u32) -> Vec<u8> {
    generators::mandelbrot(w, h)
}
fn gen_checker(w: u32, h: u32) -> Vec<u8> {
    generators::checkerboard(w, h, 8)
}

const ALL_IMAGES: &[(&str, ImgGen)] = &[
    ("gradient", gen_gradient),
    ("noise", gen_noise),
    ("color_blocks", gen_color_blocks),
    ("mandelbrot", gen_mandelbrot),
    ("checker", gen_checker),
];

const METHODS: &[u8] = &[0, 4, 6];
const QUALITIES: &[f32] = &[10.0, 25.0, 50.0, 75.0, 90.0];

fn rgba_to_l8(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .map(|px| (0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32) as u8)
        .collect()
}
fn l8_to_rgba(l8: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(l8.len() * 4);
    for &g in l8 {
        out.extend_from_slice(&[g, g, g, 255]);
    }
    out
}
fn rgba_to_la8(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|px| {
            let y = (0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32) as u8;
            [y, px[3]]
        })
        .collect()
}
fn la8_to_rgba(la8: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(la8.len() * 2);
    for px in la8.chunks_exact(2) {
        out.extend_from_slice(&[px[0], px[0], px[0], px[1]]);
    }
    out
}
fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|px| px[..3].to_vec())
        .collect()
}
fn rgb_to_opaque_rgba(rgb: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len() / 3 * 4);
    for px in rgb.chunks_exact(3) {
        out.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }
    out
}
fn rgba_to_bgra(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|px| [px[2], px[1], px[0], px[3]])
        .collect()
}
fn bgra_to_rgba(bgra: &[u8]) -> Vec<u8> {
    rgba_to_bgra(bgra)
}
fn rgba_to_argb(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|px| [px[3], px[0], px[1], px[2]])
        .collect()
}
fn argb_to_rgba(argb: &[u8]) -> Vec<u8> {
    argb.chunks_exact(4)
        .flat_map(|px| [px[1], px[2], px[3], px[0]])
        .collect()
}
fn rgba_to_bgr(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|px| [px[2], px[1], px[0]])
        .collect()
}
fn bgr_to_opaque_rgba(bgr: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bgr.len() / 3 * 4);
    for px in bgr.chunks_exact(3) {
        out.extend_from_slice(&[px[2], px[1], px[0], 255]);
    }
    out
}

fn id(p: &[u8]) -> Vec<u8> {
    p.to_vec()
}

struct Layout {
    name: &'static str,
    layout: PixelLayout,
    rgba_to_layout: fn(&[u8]) -> Vec<u8>,
    layout_to_ref_rgba: fn(&[u8]) -> Vec<u8>,
}

const LAYOUTS: &[Layout] = &[
    Layout {
        name: "rgba8",
        layout: PixelLayout::Rgba8,
        rgba_to_layout: id,
        layout_to_ref_rgba: id,
    },
    Layout {
        name: "rgb8",
        layout: PixelLayout::Rgb8,
        rgba_to_layout: rgba_to_rgb,
        layout_to_ref_rgba: rgb_to_opaque_rgba,
    },
    Layout {
        name: "bgra8",
        layout: PixelLayout::Bgra8,
        rgba_to_layout: rgba_to_bgra,
        layout_to_ref_rgba: bgra_to_rgba,
    },
    Layout {
        name: "bgr8",
        layout: PixelLayout::Bgr8,
        rgba_to_layout: rgba_to_bgr,
        layout_to_ref_rgba: bgr_to_opaque_rgba,
    },
    Layout {
        name: "argb8",
        layout: PixelLayout::Argb8,
        rgba_to_layout: rgba_to_argb,
        layout_to_ref_rgba: argb_to_rgba,
    },
    Layout {
        name: "l8",
        layout: PixelLayout::L8,
        rgba_to_layout: rgba_to_l8,
        layout_to_ref_rgba: l8_to_rgba,
    },
    Layout {
        name: "la8",
        layout: PixelLayout::La8,
        rgba_to_layout: rgba_to_la8,
        layout_to_ref_rgba: la8_to_rgba,
    },
];

fn enc_lossy(pixels: &[u8], layout: PixelLayout, w: u32, h: u32, m: u8, q: f32) -> Vec<u8> {
    let cfg = LossyConfig::new().with_quality(q).with_method(m);
    EncodeRequest::lossy(&cfg, pixels, layout, w, h)
        .encode()
        .unwrap_or_else(|e| panic!("lossy m{m} q{q} {layout:?}: {e}"))
}

fn dec_rgba(webp: &[u8]) -> Vec<u8> {
    zenwebp::oneshot::decode_rgba(webp).expect("decode_rgba").0
}

fn as_rgba_pixels(data: &[u8]) -> &[[u8; 4]] {
    assert!(data.len().is_multiple_of(4));
    let pixels: &[[u8; 4]] = bytemuck::cast_slice(data);
    pixels
}

fn score(a: &[u8], b: &[u8], w: u32, h: u32) -> f64 {
    let z = Zensim::new(ZensimProfile::latest());
    let a = RgbaSlice::new(as_rgba_pixels(a), w as usize, h as usize);
    let b = RgbaSlice::new(as_rgba_pixels(b), w as usize, h as usize);
    z.compute(&a, &b).expect("zensim compute").score()
}

fn main() {
    println!("image\tlayout\tmethod\tquality\tbytes\tscore");
    for &(name, genfn) in ALL_IMAGES {
        let rgba_orig = genfn(W, H);
        for layout in LAYOUTS {
            let layout_pixels = (layout.rgba_to_layout)(&rgba_orig);
            let reference = (layout.layout_to_ref_rgba)(&layout_pixels);
            for &m in METHODS {
                for &q in QUALITIES {
                    let webp = enc_lossy(&layout_pixels, layout.layout, W, H, m, q);
                    let decoded = dec_rgba(&webp);
                    let s = score(&reference, &decoded, W, H);
                    println!("{name}\t{}\t{m}\t{q}\t{}\t{s:.3}", layout.name, webp.len());
                }
            }
        }
    }
}
