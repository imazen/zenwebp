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
//! ## Phase 1 — THE base grid (4004 cells)
//!
//! 13 images x q{5..95} x 4 (sns,flt,segs) configs x m0-6. Complete at
//! 4004/4004 since 2026-07-16; the climb from 24% is recorded in
//! `benchmarks/byteparity_scope_2026-07-14.md`. This phase's tally is THE
//! score — keep its grid frozen so before/after deltas stay comparable.
//!
//! ## Phase 2 — axis permutations (added 2026-07-16)
//!
//! Sweeps the previously-unswept setting axes both encoders expose:
//! filter_sharpness 1-7, segments 3 + sns/filter extremes, quality edges
//! (q0/q1/q99/q100), pinned partition_limit, sharp_yuv, and alpha (RGBA
//! input, alpha_quality 100/90). Each axis tallies separately so a
//! not-yet-parity axis (e.g. alpha's separately-coded plane) doesn't blur
//! the score of the others.
//!
//! Out of scope (no matched knob or architecturally different):
//! target_size/target_PSNR (zen drives an outer q-search loop, libwebp a
//! multi-pass StatLoop search), pass>1, autofilter, filter_type,
//! preprocessing bits, multi-partition output (zenwebp always emits 1
//! token partition; libwebp default `partitions=0` matches), low_memory,
//! emulate_jpeg_size, qmin/qmax.

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

/// RGBA variant of `synth` with a chosen alpha pattern.
///   0 = fully opaque (both sides must omit the ALPH chunk)
///   1 = horizontal alpha gradient
///   2 = 8x8 binary checker (0 / 255)
fn synth_rgba(w: u32, h: u32, seed: u32, alpha_kind: u8) -> Vec<u8> {
    let (rgb, _, _) = synth(w, h, seed);
    let mut px = Vec::with_capacity((w as usize) * (h as usize) * 4);
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) as usize) * 3;
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
            px.extend_from_slice(&[rgb[i], rgb[i + 1], rgb[i + 2], a]);
        }
    }
    px
}

/// One comparison cell across every swept axis. Defaults reproduce the
/// phase-1 base grid exactly.
#[derive(Clone, Copy)]
struct Cell {
    q: u8,
    m: u8,
    sns: u8,
    flt: u8,
    segs: u8,
    sharp: u8,
    /// `Some(v)` pins partition_limit on BOTH sides (zen `None` default is
    /// an auto-retry that starts at 0 = libwebp's default).
    plimit: Option<u8>,
    sharp_yuv: bool,
    /// `Some(aq)` encodes RGBA input with this alpha_quality.
    alpha_q: Option<u8>,
}

impl Cell {
    const fn base(q: u8, m: u8, sns: u8, flt: u8, segs: u8) -> Self {
        Self {
            q,
            m,
            sns,
            flt,
            segs,
            sharp: 0,
            plimit: None,
            sharp_yuv: false,
            alpha_q: None,
        }
    }

    fn label(&self, name: &str) -> String {
        let mut s = format!(
            "{name} q{} m{} sns{} flt{} segs{}",
            self.q, self.m, self.sns, self.flt, self.segs
        );
        if self.sharp != 0 {
            s.push_str(&format!(" sh{}", self.sharp));
        }
        if let Some(p) = self.plimit {
            s.push_str(&format!(" plim{p}"));
        }
        if self.sharp_yuv {
            s.push_str(" sharpyuv");
        }
        if let Some(a) = self.alpha_q {
            s.push_str(&format!(" alphaq{a}"));
        }
        s
    }
}

/// Encode one cell through both encoders; Ok(None) = byte-identical,
/// Ok(Some(msg)) = diverged, Err = zen encode error.
fn run_cell(name: &str, pixels: &[u8], w: u32, h: u32, c: Cell) -> Result<Option<String>, String> {
    let mut cfg = LossyConfig::new()
        .with_quality(f32::from(c.q))
        .with_method(c.m)
        .with_segments(c.segs)
        .with_sns_strength(c.sns)
        .with_filter_strength(c.flt)
        .with_filter_sharpness(c.sharp)
        .with_cost_model(CostModel::StrictLibwebpParity);
    if let Some(p) = c.plimit {
        cfg = cfg.with_partition_limit(p);
    }
    if c.sharp_yuv {
        cfg = cfg.with_sharp_yuv(true);
    }
    if let Some(aq) = c.alpha_q {
        cfg.alpha_quality = aq;
    }
    let layout = if c.alpha_q.is_some() {
        PixelLayout::Rgba8
    } else {
        PixelLayout::Rgb8
    };
    let zen = EncodeRequest::lossy(&cfg, pixels, layout, w, h)
        .encode()
        .map_err(|e| format!("{}: zen ERR {e:?}", c.label(name)))?;

    let mut lcfg = webpx::EncoderConfig::new()
        .quality(f32::from(c.q))
        .method(c.m)
        .segments(c.segs)
        .sns_strength(c.sns)
        .filter_strength(c.flt)
        .filter_sharpness(c.sharp);
    if let Some(p) = c.plimit {
        lcfg = lcfg.partition_limit(p);
    }
    if c.sharp_yuv {
        lcfg = lcfg.sharp_yuv(true);
    }
    let lib = if let Some(aq) = c.alpha_q {
        lcfg.alpha_quality(aq)
            .encode_rgba(pixels, w, h, webpx::Unstoppable)
            .unwrap()
    } else {
        lcfg.encode_rgb(pixels, w, h, webpx::Unstoppable).unwrap()
    };

    if zen == lib {
        Ok(None)
    } else {
        let common = zen.iter().zip(&lib).take_while(|(a, b)| a == b).count();
        Ok(Some(format!(
            "{}: zen={} lib={} 1st-diff@{common}",
            c.label(name),
            zen.len(),
            lib.len()
        )))
    }
}

/// Tally for one phase/axis: run the cells, collect failures.
struct Tally {
    total: u32,
    ident: u32,
    fails: Vec<String>,
}

impl Tally {
    fn new() -> Self {
        Self {
            total: 0,
            ident: 0,
            fails: Vec::new(),
        }
    }
    fn run(&mut self, name: &str, pixels: &[u8], w: u32, h: u32, c: Cell) {
        self.total += 1;
        match run_cell(name, pixels, w, h, c) {
            Ok(None) => self.ident += 1,
            Ok(Some(msg)) => self.fails.push(msg),
            Err(msg) => self.fails.push(msg),
        }
    }
    fn report(&self, label: &str, cap: usize) {
        println!("AXIS {label}: {}/{} byte-identical", self.ident, self.total);
        for f in self.fails.iter().take(cap) {
            println!("  {f}");
        }
        if self.fails.len() > cap {
            println!("  ... and {} more", self.fails.len() - cap);
        }
    }
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

    // ===== Phase 1: THE base grid (frozen — this tally is the score) =====
    let mut base = Tally::new();
    for (name, rgb, w, h) in &imgs {
        for &(sns, flt, segs) in configs {
            for &q in qs {
                for m in 0u8..=6 {
                    base.run(name, rgb, *w, *h, Cell::base(q, m, sns, flt, segs));
                }
            }
        }
    }
    println!(
        "# byte-parity sweep: {}/{} BYTE-IDENTICAL ({} images, {} q, {} configs, m0-6)",
        base.ident,
        base.total,
        imgs.len(),
        qs.len(),
        configs.len()
    );
    if base.fails.is_empty() {
        println!("ALL BYTE-IDENTICAL");
    } else {
        println!("--- {} non-identical cells ---", base.fails.len());
        for f in &base.fails {
            println!("{f}");
        }
    }

    // Skip the axis phase with --base-only (keeps the fast score loop).
    if std::env::args().any(|a| a == "--base-only") {
        return;
    }
    let by_name = |n: &str| imgs.iter().find(|(name, ..)| name == n);
    let pick = |names: &[&str]| -> Vec<&(String, Vec<u8>, u32, u32)> {
        names.iter().filter_map(|n| by_name(n)).collect()
    };

    println!("\n# ===== Phase 2: axis permutations =====");

    // --- filter_sharpness 1..7 ---
    let mut t = Tally::new();
    for (name, rgb, w, h) in pick(&["382297.png", "1025469.png", "synth_33x17", "synth_129x127"]) {
        for sharp in 1u8..=7 {
            for &(sns, flt, segs) in &[(50u8, 60u8, 4u8), (30, 20, 2)] {
                for &q in &[5u8, 20, 50, 75, 90] {
                    for m in 0u8..=6 {
                        let c = Cell {
                            sharp,
                            ..Cell::base(q, m, sns, flt, segs)
                        };
                        t.run(name, rgb, *w, *h, c);
                    }
                }
            }
        }
    }
    t.report("filter_sharpness 1-7", 20);

    // --- segments=3 + sns/filter extremes ---
    let mut t = Tally::new();
    for (name, rgb, w, h) in pick(&["382297.png", "1025469.png", "1418519.png", "synth_129x127"]) {
        for &(sns, flt, segs) in &[
            (50u8, 60u8, 3u8),
            (100, 100, 4),
            (80, 30, 4),
            (25, 10, 2),
            (100, 0, 4),
            (0, 100, 1),
            // sns>0 with ONE segment: exercises the uv_alpha-derived UV quant
            // deltas on the single-segment path (uv_alpha defaults to 0 when
            // the analysis pass doesn't run, m2+). (#38)
            (80, 60, 1),
            (100, 30, 1),
        ] {
            for &q in &[5u8, 20, 50, 75, 90] {
                for m in 0u8..=6 {
                    t.run(name, rgb, *w, *h, Cell::base(q, m, sns, flt, segs));
                }
            }
        }
    }
    t.report("segments-3 + sns/filter extremes", 20);

    // --- quality edges q0/q1/q99/q100 ---
    let mut t = Tally::new();
    for (name, rgb, w, h) in &imgs {
        for &(sns, flt, segs) in configs {
            for &q in &[0u8, 1, 99, 100] {
                for m in 0u8..=6 {
                    t.run(name, rgb, *w, *h, Cell::base(q, m, sns, flt, segs));
                }
            }
        }
    }
    t.report("quality edges 0/1/99/100", 20);

    // --- pinned partition_limit ---
    let mut t = Tally::new();
    for (name, rgb, w, h) in pick(&["382297.png", "1025469.png"]) {
        for &pl in &[30u8, 60, 100] {
            for &(sns, flt, segs) in &[(50u8, 60u8, 4u8), (0, 0, 1)] {
                for &q in &[5u8, 50, 90] {
                    for m in 0u8..=6 {
                        let c = Cell {
                            plimit: Some(pl),
                            ..Cell::base(q, m, sns, flt, segs)
                        };
                        t.run(name, rgb, *w, *h, c);
                    }
                }
            }
        }
    }
    t.report("partition_limit 30/60/100", 20);

    // --- sharp_yuv input conversion ---
    let mut t = Tally::new();
    for (name, rgb, w, h) in pick(&["382297.png", "1025469.png", "1418519.png"]) {
        for &(sns, flt, segs) in &[(50u8, 60u8, 4u8), (0, 0, 1)] {
            for &q in &[5u8, 50, 75, 90] {
                for &m in &[0u8, 2, 4, 6] {
                    let c = Cell {
                        sharp_yuv: true,
                        ..Cell::base(q, m, sns, flt, segs)
                    };
                    t.run(name, rgb, *w, *h, c);
                }
            }
        }
    }
    t.report("sharp_yuv", 20);

    // --- alpha (RGBA input) ---
    let mut t = Tally::new();
    for &(w, h, seed) in &[(64u32, 64u32, 31u32), (33, 17, 23)] {
        for alpha_kind in 0u8..=2 {
            let rgba = synth_rgba(w, h, seed, alpha_kind);
            let name = format!(
                "synthA{alpha_kind}_{w}x{h}",
                alpha_kind = alpha_kind,
                w = w,
                h = h
            );
            for &aq in &[100u8, 90] {
                for &q in &[5u8, 50, 75, 90] {
                    for &m in &[0u8, 2, 4, 6] {
                        let c = Cell {
                            alpha_q: Some(aq),
                            ..Cell::base(q, m, 50, 60, 4)
                        };
                        t.run(&name, &rgba, w, h, c);
                    }
                }
            }
        }
    }
    t.report("alpha (RGBA, alpha_quality 100/90)", 20);
}
