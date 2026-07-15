//! Byte-parity grid score for `CostModel::StrictLibwebpParity` (#38).
//!
//! THE score for #38: how much of the (image x quality x config x method) grid
//! is byte-identical to libwebp. Reports `<ident>/<total>` plus every
//! non-identical cell with its sizes and first-differing byte offset.
//!
//! Usage: move to examples/ or add an [[example]] entry, then:
//!   cargo run --release --example byteparity_sweep
//!
//! Companion tools:
//!   * `dev/mbpixdiff.rs`     — find the first EMITTED divergence in one cell
//!   * `dev/bitexact_diff.rs` — header fields + per-MB mode stream for one cell
//!
//! Method note: this builds zenwebp into the example's own target, so a stale
//! binary silently reports phantom divergences after a lib change. Always
//! rebuild before trusting a delta.
//!
//! `1st-diff@4` means only the RIFF size field differs so far, i.e. the payloads
//! diverge in content, not in an early header field.
//!
//! ## Baseline: 3578/4004 = 89.4% (2026-07-15, this grid)
//!
//! **This number is NOT comparable to the 3488/4004 (87.1%) quoted in
//! `benchmarks/byteparity_scope_2026-07-14.md`.** That measurement came from an
//! ad-hoc harness that lived in `/tmp` and was wiped; 10 of the 13 images here
//! are synthetic, and their generator was reconstructed from scratch rather
//! than recovered, so the synthetic cells are different content. The encoder is
//! byte-identical between the two runs — the delta is the grid, not progress.
//! The 3 CID22 images are unchanged and account for 299 of the 426 failures.
//!
//! Treat 3578/4004 as the durable baseline: this file is committed, so the grid
//! is now reproducible. Don't compare across grids; re-run this tool for a
//! before/after on any encoder change.
//!
//! The 24% -> ~87-89% climb across five parity-gated fixes on 2026-07-15 is
//! recorded in `benchmarks/byteparity_scope_2026-07-14.md`.

use std::path::Path;

use zenwebp::{CostModel, EncodeRequest, LossyConfig, PixelLayout};

/// Same loader as `dev/bitexact_diff.rs` (the `png` crate is the dev-dep; the
/// `image` crate is not available here). Returns None when the corpus file is
/// absent so the synthetic cells still run.
fn load_png(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    if !Path::new(path).exists() {
        return None;
    }
    let file = std::fs::File::open(path).ok()?;
    let mut d = png::Decoder::new(std::io::BufReader::new(file));
    d.set_transformations(png::Transformations::normalize_to_color8());
    let mut r = d.read_info().ok()?;
    let mut buf = vec![0u8; r.output_buffer_size()?];
    let info = r.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => buf
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

/// Deterministic gradient+noise synthetic (no corpus dependency).
fn synth(w: u32, h: u32, seed: u32) -> (Vec<u8>, u32, u32) {
    let mut px = Vec::with_capacity((w as usize) * (h as usize) * 3);
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let n = (s >> 24) as u8 / 8;
            let r = ((x * 255 / w.max(1)) as u8).wrapping_add(n);
            let g = ((y * 255 / h.max(1)) as u8).wrapping_add(n);
            let b = (((x + y) * 255 / (w + h).max(1)) as u8).wrapping_add(n);
            px.extend_from_slice(&[r, g, b]);
        }
    }
    (px, w, h)
}

fn main() {
    // Real images (skipped if absent) + synthetic tiny/odd/edge cases. The
    // tiny and odd-dimension entries matter: they exercise partial-MB edges and
    // fixed header overhead, where parity bugs cluster.
    let reals = [
        "/home/lilith/.cache/codec-corpus/v1/CID22/CID22-512/validation/382297.png",
        "/home/lilith/.cache/codec-corpus/v1/CID22/CID22-512/validation/1025469.png",
        "/home/lilith/.cache/codec-corpus/v1/CID22/CID22-512/validation/1418519.png",
    ];
    let mut imgs: Vec<(String, Vec<u8>, u32, u32)> = Vec::new();
    for p in reals {
        if let Some((rgb, w, h)) = load_png(p) {
            imgs.push((p.rsplit('/').next().unwrap().to_string(), rgb, w, h));
        }
    }
    for (w, h, s) in [
        (1u32, 1u32, 7u32),
        (2, 2, 11),
        (3, 3, 13),
        (16, 16, 17),
        (17, 17, 19),
        (33, 17, 23),
        (31, 47, 29),
        (64, 64, 31),
        (129, 127, 37),
        (256, 255, 41),
    ] {
        let (rgb, w, h) = synth(w, h, s);
        imgs.push((format!("synth_{w}x{h}"), rgb, w, h));
    }

    let qs: &[u8] = &[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95];
    // (sns, filter, segments). The two sns=0 rows isolate segmentation; the
    // other two are the shipped-default-ish and a mid config.
    let configs: &[(u8, u8, u8)] = &[(0, 0, 1), (50, 60, 4), (0, 0, 4), (30, 20, 2)];

    let mut total = 0u32;
    let mut ident = 0u32;
    let mut fails: Vec<String> = Vec::new();

    for (name, rgb, w, h) in &imgs {
        for &(sns, flt, segs) in configs {
            for &q in qs {
                for m in 0u8..=6 {
                    let cfg = LossyConfig::new()
                        .with_quality(f32::from(q))
                        .with_method(m)
                        .with_segments(segs)
                        .with_sns_strength(sns)
                        .with_filter_strength(flt)
                        .with_filter_sharpness(0)
                        .with_cost_model(CostModel::StrictLibwebpParity);
                    let zen =
                        match EncodeRequest::lossy(&cfg, rgb, PixelLayout::Rgb8, *w, *h).encode() {
                            Ok(z) => z,
                            Err(e) => {
                                fails.push(format!(
                                    "{name} q{q} m{m} sns{sns} flt{flt} segs{segs}: zen ERR {e:?}"
                                ));
                                total += 1;
                                continue;
                            }
                        };
                    let lib = webpx::EncoderConfig::new()
                        .quality(f32::from(q))
                        .method(m)
                        .segments(segs)
                        .sns_strength(sns)
                        .filter_strength(flt)
                        .filter_sharpness(0)
                        .encode_rgb(rgb, *w, *h, webpx::Unstoppable)
                        .unwrap();
                    total += 1;
                    if zen == lib {
                        ident += 1;
                    } else {
                        let common = zen.iter().zip(&lib).take_while(|(a, b)| a == b).count();
                        fails.push(format!(
                            "{name} q{q} m{m} sns{sns} flt{flt} segs{segs}: zen={} lib={} 1st-diff@{common}",
                            zen.len(),
                            lib.len()
                        ));
                    }
                }
            }
        }
    }

    println!(
        "# byte-parity sweep: {ident}/{total} BYTE-IDENTICAL ({} images, {} q, {} configs, m0-6)",
        imgs.len(),
        qs.len(),
        configs.len()
    );
    if fails.is_empty() {
        println!("ALL BYTE-IDENTICAL");
    } else {
        println!("--- {} non-identical cells ---", fails.len());
        for f in &fails {
            println!("{f}");
        }
    }
}
