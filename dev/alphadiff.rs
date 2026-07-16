//! ALPH-chunk layering diff vs libwebp (#38 alpha parity).
//!
//! Encodes the same RGBA input through zenwebp (StrictLibwebpParity) and
//! libwebp (webpx), extracts each file's ALPH chunk, and reports the
//! divergence LAYER:
//!
//!   1. header byte — compression method | filter | preprocessing bits
//!   2. pipeline    — which filter each side chose
//!   3. payload     — first differing byte of the compressed plane
//!
//! It then isolates the VP8L-stream distance from the pipeline distance:
//! it reproduces libwebp's exact filtered plane (quantize_levels + the
//! filter named in LIB's header) and re-encodes that plane through zen's
//! VP8L at libwebp's alpha operating point. Any difference in THAT
//! comparison is pure VP8L-stream divergence — the pipeline is out of the
//! picture by construction.
//!
//! Usage:
//!   cargo run --release --features __expert --example alphadiff -- \
//!       [WxH:SEED:KIND] [q] [m] [alpha_q]
//! KIND: 0=opaque 1=gradient 2=checker (same generator as byteparity_sweep)

use zenwebp::{CostModel, EncodeRequest, LossyConfig, PixelLayout};

fn synth_rgba(w: u32, h: u32, seed: u32, alpha_kind: u8) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut px = Vec::with_capacity((w as usize) * (h as usize) * 4);
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let n = (s >> 24) as u8 / 8;
            let r = ((x * 255 / w.max(1)) as u8).wrapping_add(n);
            let g = ((y * 255 / h.max(1)) as u8).wrapping_add(n);
            let b = (((x + y) * 255 / (w + h).max(1)) as u8).wrapping_add(n);
            let a = match alpha_kind {
                0 => 255u8,
                1 => (x * 255 / w.max(1)) as u8,
                _ => {
                    if ((x / 8) + (y / 8)) % 2 == 0 {
                        0
                    } else {
                        255
                    }
                }
            };
            px.extend_from_slice(&[r, g, b, a]);
        }
    }
    px
}

/// Extract a chunk's payload from a RIFF/WEBP container.
fn find_chunk<'a>(webp: &'a [u8], fourcc: &[u8; 4]) -> Option<&'a [u8]> {
    let mut pos = 12usize; // RIFF....WEBP
    while pos + 8 <= webp.len() {
        let id = &webp[pos..pos + 4];
        let sz = u32::from_le_bytes(webp[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if id == fourcc {
            return Some(&webp[pos + 8..pos + 8 + sz]);
        }
        pos += 8 + sz + (sz & 1);
    }
    None
}

fn first_diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let spec = args.get(1).map(String::as_str).unwrap_or("64x64:31:1");
    let q: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(75);
    let m: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let aq: u8 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(100);

    let parts: Vec<&str> = spec.split(':').collect();
    let (dims, seed, kind) = (parts[0], parts[1], parts[2]);
    let (w, h) = dims.split_once('x').expect("WxH");
    let (w, h): (u32, u32) = (w.parse().unwrap(), h.parse().unwrap());
    let seed: u32 = seed.parse().unwrap();
    let kind: u8 = kind.parse().unwrap();
    let rgba = synth_rgba(w, h, seed, kind);
    println!("input: {w}x{h} seed{seed} kind{kind} q{q} m{m} alpha_q{aq}");

    let mut cfg = LossyConfig::new()
        .with_quality(f32::from(q))
        .with_method(m)
        .with_segments(4)
        .with_sns_strength(50)
        .with_filter_strength(60)
        .with_filter_sharpness(0)
        .with_cost_model(CostModel::StrictLibwebpParity);
    cfg.alpha_quality = aq;
    let zen = EncodeRequest::lossy(&cfg, &rgba, PixelLayout::Rgba8, w, h)
        .encode()
        .unwrap();
    let lib = webpx::EncoderConfig::new()
        .quality(f32::from(q))
        .method(m)
        .segments(4)
        .sns_strength(50)
        .filter_strength(60)
        .filter_sharpness(0)
        .alpha_quality(aq)
        .encode_rgba(&rgba, w, h, webpx::Unstoppable)
        .unwrap();

    println!(
        "file bytes: zen={} lib={} {}",
        zen.len(),
        lib.len(),
        if zen == lib { "IDENTICAL" } else { "DIFFER" }
    );

    let zalph = find_chunk(&zen, b"ALPH");
    let lalph = find_chunk(&lib, b"ALPH");
    match (zalph, lalph) {
        (None, None) => println!("no ALPH chunk on either side (opaque)"),
        (Some(za), Some(la)) => {
            let dec = |h: u8| {
                (
                    h & 3,        // compression method
                    (h >> 2) & 3, // filter
                    (h >> 4) & 3, // preprocessing
                )
            };
            let (zc, zf, zp) = dec(za[0]);
            let (lc, lf, lp) = dec(la[0]);
            println!(
                "ALPH header: zen method={zc} filter={zf} pre={zp} | lib method={lc} filter={lf} pre={lp}"
            );
            println!(
                "ALPH payload: zen={} lib={} 1st-diff@{}",
                za.len() - 1,
                la.len() - 1,
                first_diff(&za[1..], &la[1..])
            );

            // Layer isolation: reproduce LIB's filtered plane and run zen's
            // VP8L on it at libwebp's alpha operating point.
            let mut plane: Vec<u8> = rgba.chunks_exact(4).map(|p| p[3]).collect();
            if lp == 1 {
                zenwebp::__expert::alpha_quantize_levels(
                    &mut plane,
                    zenwebp::__expert::alpha_levels_for_quality(aq),
                );
            }
            let filtered =
                zenwebp::__expert::alpha_apply_filter(lf, &plane, w as usize, h as usize);
            let zpayload =
                zenwebp::__expert::alpha_vp8l_payload(&filtered, w, h, m.min(6), aq == 100)
                    .unwrap();
            println!(
                "VP8L-on-LIB-plane: zen={} lib={} 1st-diff@{} {}",
                zpayload.len(),
                la.len() - 1,
                first_diff(&zpayload, &la[1..]),
                if zpayload.as_slice() == &la[1..] {
                    "IDENTICAL"
                } else {
                    "DIFFER"
                }
            );
        }
        (za, la) => println!(
            "ALPH presence differs: zen={} lib={}",
            za.is_some(),
            la.is_some()
        ),
    }
}
