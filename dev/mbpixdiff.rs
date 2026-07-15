//! Find the FIRST EMITTED macroblock divergence vs libwebp (#38).
//!
//! Decodes zenwebp's (`StrictLibwebpParity`) and libwebp's encodes of the same
//! source and diffs reconstructed pixels per macroblock. The first differing MB
//! is the clean root to trace: everything before it is byte-identical, so its
//! prediction inputs and neighbour context are provably shared.
//!
//! Usage: move to examples/ or add an [[example]] entry, then:
//!   cargo run --release --example mbpixdiff -- [image.png] [quality] [method]
//!
//! ## Why this exists (read before tracing #38 by hand)
//!
//! zenwebp calls `pick_best_intra16`/`pick_best_intra4` ~4x per macroblock (the
//! I16 candidate, the I4-bail's `i16_score` probe, `intra16_d`, ...), each with
//! DIFFERENT evolving state. So `MB_DEBUG` output MIXES emission and probe
//! calls, and a probe's numbers look exactly like a real divergence. On
//! 2026-07-15 that cost a long trace of mb(3,0) at q40/m6 which turned out to
//! be a probe — zen emits the same mode as libwebp there. The first real
//! emitted divergence was mb(11,8).
//!
//! So: run THIS first to get the MB, then point `MB_DEBUG` at that MB only.
//! Do not trust a per-MB debug print until this tool says the MB diverges.
//!
//! Caveat: a byte-parity failure with NO per-MB pixel difference means the
//! divergence is header/probability-only, not mode/coefficient — trace with
//! `dev/bitexact_diff.rs` instead.

use zenwebp::{CostModel, EncodeRequest, LossyConfig, PixelLayout};

/// Same loader as `dev/bitexact_diff.rs` (the `png` crate is the dev-dep; the
/// `image` crate is not available here).
fn load(path: &str) -> (Vec<u8>, u32, u32) {
    let file = std::fs::File::open(path).unwrap();
    let mut d = png::Decoder::new(std::io::BufReader::new(file));
    d.set_transformations(png::Transformations::normalize_to_color8());
    let mut r = d.read_info().unwrap();
    let mut buf = vec![0u8; r.output_buffer_size().unwrap()];
    let info = r.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => buf
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        _ => unreachable!(),
    };
    (rgb, info.width, info.height)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let img_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/home/lilith/.cache/codec-corpus/v1/CID22/CID22-512/validation/382297.png");
    let q: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40);
    let m: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);
    // Held at the config the #38 trace uses; edit if chasing another cell.
    let (sns, flt, segs) = (0u8, 0u8, 1u8);

    let (rgb, w, h) = load(img_path);
    let (mb_w, mb_h) = (w.div_ceil(16) as usize, h.div_ceil(16) as usize);
    println!(
        "image: {img_path} {w}x{h} ({mb_w}x{mb_h} MBs) q{q} m{m} sns{sns} flt{flt} segs{segs}"
    );

    let cfg = LossyConfig::new()
        .with_quality(f32::from(q))
        .with_method(m)
        .with_segments(segs)
        .with_sns_strength(sns)
        .with_filter_strength(flt)
        .with_filter_sharpness(0)
        .with_cost_model(CostModel::StrictLibwebpParity);
    let zen = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, w, h)
        .encode()
        .unwrap();
    let lib = webpx::EncoderConfig::new()
        .quality(f32::from(q))
        .method(m)
        .segments(segs)
        .sns_strength(sns)
        .filter_strength(flt)
        .filter_sharpness(0)
        .encode_rgb(&rgb, w, h, webpx::Unstoppable)
        .unwrap();
    println!(
        "bytes: zen={} lib={} {}",
        zen.len(),
        lib.len(),
        if zen == lib { "IDENTICAL" } else { "DIFFER" }
    );

    // zenwebp's decoder is bit-exact vs libwebp's for a GIVEN bitstream, so any
    // pixel difference here is the two ENCODERS disagreeing, not the decoders.
    let (zpix, zw, zh) = zenwebp::oneshot::decode_rgba(&zen).unwrap();
    let (lpix, lw, lh) = zenwebp::oneshot::decode_rgba(&lib).unwrap();
    assert_eq!((zw, zh), (lw, lh));
    let stride = (zw as usize) * 4;

    let mut first: Option<(usize, usize)> = None;
    let mut diffs: Vec<(usize, usize, i32)> = Vec::new();
    for mby in 0..mb_h {
        for mbx in 0..mb_w {
            let mut mbmax = 0i32;
            for dy in 0..16 {
                let py = mby * 16 + dy;
                if py >= zh as usize {
                    break;
                }
                for dx in 0..16 {
                    let px = mbx * 16 + dx;
                    if px >= zw as usize {
                        break;
                    }
                    let o = py * stride + px * 4;
                    for c in 0..3 {
                        let d = (i32::from(zpix[o + c]) - i32::from(lpix[o + c])).abs();
                        if d > mbmax {
                            mbmax = d;
                        }
                    }
                }
            }
            if mbmax > 0 {
                if first.is_none() {
                    first = Some((mbx, mby));
                }
                diffs.push((mbx, mby, mbmax));
            }
        }
    }

    match first {
        None => println!(
            "NO per-MB pixel difference — any byte divergence is header/probability-only; \
             use dev/bitexact_diff.rs"
        ),
        Some((fx, fy)) => {
            println!(
                "FIRST differing MB: mb({fx},{fy})   (total differing MBs: {})",
                diffs.len()
            );
            println!("trace THIS MB only:  MB_DEBUG={fx},{fy} cargo run --release --example ...");
            println!("first differing MBs (mbx,mby,maxdiff):");
            for (mbx, mby, d) in diffs.iter().take(12) {
                println!("   mb({mbx},{mby}) maxdiff={d}");
            }
        }
    }
}
