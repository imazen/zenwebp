//! Extended convergence evaluation: CLIC2025 corpus + per-frame
//! convergence on an animated WebP. Companion to
//! `dev/zensim_convergence_eval.rs` — that one stays focused on the
//! standard 76-image RGB sweep; this one adds the broader sanity
//! checks the user wants for PR #48 Task 3.
//!
//! Plus an RGBA "unsupported-layout" smoke test that asserts the
//! encoder cleanly refuses RGBA + target_zensim with the typed
//! [`zenwebp::EncodeError::TargetZensimUnsupportedLayout`] error
//! (rather than silently dropping alpha to make iteration fire). This
//! documents the current limitation in code; proper RGBA support is
//! tracked separately.
//!
//! Usage:
//!   zensim_extended_eval [--clic /path/to/clic2025-1024]
//!     [--anim /path/to/anim.webp]
//!     [--targets 75,80,85]
//!     [--max-passes 2]
//!     [--max-overshoot 1.5]
//!     [--clic-limit 30]
//!
//! Defaults (when flags omitted):
//!   --clic   $HOME/work/codec-corpus/clic2025-1024
//!   --anim   $HOME/work/codec-corpus/webp-conformance/valid/anim.webp
//!   --targets 75,80,85
//!
//! Reports aggregate stats per phase. Saves a CLIC TSV under
//! /mnt/v/output/zenwebp/zensim-clic-eval/ for traceability (not
//! committed).

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use zenwebp::{EncodeError, EncodeRequest, LossyConfig, PixelLayout, ZensimTarget};

#[derive(Clone)]
struct Cli {
    clic: PathBuf,
    anim: PathBuf,
    targets: Vec<f32>,
    max_passes: u8,
    max_overshoot: f32,
    method: u8,
    clic_limit: usize,
}

fn parse_args() -> Cli {
    let home = env::var("HOME").expect("HOME not set");
    let mut cli = Cli {
        clic: PathBuf::from(format!("{home}/work/codec-corpus/clic2025-1024")),
        anim: PathBuf::from(format!(
            "{home}/work/codec-corpus/webp-conformance/valid/anim.webp"
        )),
        targets: vec![75.0, 80.0, 85.0],
        max_passes: 2,
        max_overshoot: 1.5,
        method: 4,
        clic_limit: 30,
    };
    let argv: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        let next = || -> &str { argv.get(i + 1).map(|s| s.as_str()).unwrap_or("") };
        match a.as_str() {
            "--clic" => {
                cli.clic = PathBuf::from(next());
                i += 2;
            }
            "--anim" => {
                cli.anim = PathBuf::from(next());
                i += 2;
            }
            "--targets" => {
                cli.targets = next()
                    .split(',')
                    .map(|s| s.trim().parse::<f32>().expect("bad --targets"))
                    .collect();
                i += 2;
            }
            "--max-passes" => {
                cli.max_passes = next().parse().unwrap();
                i += 2;
            }
            "--max-overshoot" => {
                cli.max_overshoot = next().parse().unwrap();
                i += 2;
            }
            "--method" => {
                cli.method = next().parse().unwrap();
                i += 2;
            }
            "--clic-limit" => {
                cli.clic_limit = next().parse().unwrap();
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    cli
}

#[derive(Default, Clone)]
struct Agg {
    n: u32,
    passes_sum: u32,
    passes_one: u32,
    achieved_sum: f64,
    targets_met: u32,
    undershoot: u32,
    overshoot: u32,
    bytes: Vec<u64>,
}

fn fold(into: &mut Agg, src: &Agg) {
    into.n += src.n;
    into.passes_sum += src.passes_sum;
    into.passes_one += src.passes_one;
    into.achieved_sum += src.achieved_sum;
    into.targets_met += src.targets_met;
    into.undershoot += src.undershoot;
    into.overshoot += src.overshoot;
    into.bytes.extend(src.bytes.iter().copied());
}

fn agg_one(target: f32, max_overshoot: f32, m: zenwebp::ZensimEncodeMetrics, bytes: usize) -> Agg {
    Agg {
        n: 1,
        passes_sum: u32::from(m.passes_used),
        passes_one: u32::from(m.passes_used == 1),
        achieved_sum: f64::from(m.achieved_score),
        targets_met: u32::from(m.targets_met),
        undershoot: u32::from(m.achieved_score < target),
        overshoot: u32::from(m.achieved_score > target + max_overshoot),
        bytes: vec![bytes as u64],
    }
}

fn median_u64(v: &[u64]) -> u64 {
    if v.is_empty() {
        return 0;
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    if s.len() % 2 == 1 {
        s[s.len() / 2]
    } else {
        (s[s.len() / 2 - 1] + s[s.len() / 2]) / 2
    }
}

fn print_summary(label: &str, target: Option<f32>, a: &Agg) {
    if a.n == 0 {
        return;
    }
    let n = a.n as f32;
    let tgt = target.map(|t| format!("{t:.1}")).unwrap_or("ALL".into());
    println!(
        "{:24} target={:>5} n={:>4}  passes_avg={:.2}  pass1_share={:.0}%  achieved_avg={:.2}  met={}/{}  und={} ovs={}  med_bytes={}",
        label,
        tgt,
        a.n,
        a.passes_sum as f32 / n,
        100.0 * a.passes_one as f32 / n,
        a.achieved_sum / a.n as f64,
        a.targets_met,
        a.n,
        a.undershoot,
        a.overshoot,
        median_u64(&a.bytes),
    );
}

fn main() {
    let cli = parse_args();
    eprintln!(
        "extended-eval: targets={:?} max_passes={} max_overshoot={} method={}",
        cli.targets, cli.max_passes, cli.max_overshoot, cli.method
    );

    // ============================================================
    // 3a — CLIC2025 1024px corpus (RGB)
    // ============================================================
    println!("\n========================================");
    println!("=== Phase 3a: CLIC2025 1024px corpus ===");
    println!("========================================");
    run_clic_phase(&cli);

    // ============================================================
    // 3b — RGBA "unsupported-layout" smoke test
    // ============================================================
    println!("\n==============================================");
    println!("=== Phase 3b: RGBA unsupported-layout smoke ===");
    println!("==============================================");
    run_rgba_unsupported_smoke();

    // ============================================================
    // 3c — Animation per-frame convergence
    // ============================================================
    println!("\n========================================");
    println!("=== Phase 3c: animation per-frame    ===");
    println!("========================================");
    run_anim_phase(&cli);
}

// --------------------------------------------------------------------
// 3a: CLIC2025
// --------------------------------------------------------------------

fn run_clic_phase(cli: &Cli) {
    let mut paths: Vec<PathBuf> = match fs::read_dir(&cli.clic) {
        Ok(it) => it
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "png" || e == "PNG"))
            .collect(),
        Err(e) => {
            eprintln!(
                "WARN: cannot read CLIC corpus at {}: {e:?}",
                cli.clic.display()
            );
            return;
        }
    };
    paths.sort();
    let total_count = paths.len();
    paths.truncate(cli.clic_limit);
    eprintln!(
        "CLIC: {} candidates at {}, limited to {}",
        total_count,
        cli.clic.display(),
        paths.len()
    );

    // Persist per-image TSV.
    let out_dir = PathBuf::from("/mnt/v/output/zenwebp/zensim-clic-eval");
    let _ = fs::create_dir_all(&out_dir);
    let out_path = out_dir.join("clic1024_2026-04-27.tsv");
    let mut tsv = match fs::File::create(&out_path) {
        Ok(f) => Some(std::io::BufWriter::new(f)),
        Err(e) => {
            eprintln!("warn: cannot create {} ({e:?})", out_path.display());
            None
        }
    };
    if let Some(w) = tsv.as_mut() {
        let _ = writeln!(
            w,
            "image\twidth\theight\ttarget\tpasses\tachieved\tbytes\ttargets_met"
        );
    }

    use std::collections::BTreeMap;
    let mut by_target: BTreeMap<u32, Agg> = BTreeMap::new();
    let mut overall = Agg::default();

    for path in &paths {
        let (rgb, w, h) = match decode_png_rgb(path) {
            Some(t) => t,
            None => {
                eprintln!("skip: cannot decode {}", path.display());
                continue;
            }
        };
        if w < 32 || h < 32 {
            continue;
        }
        let img_name = path.file_name().unwrap().to_string_lossy().to_string();
        for &target in &cli.targets {
            let cfg = LossyConfig::new()
                .with_method(cli.method)
                .with_target_zensim(
                    ZensimTarget::new(target)
                        .with_max_overshoot(Some(cli.max_overshoot))
                        .with_max_passes(cli.max_passes),
                );
            let r = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, w, h).encode_with_metrics();
            match r {
                Ok((b, m)) => {
                    if let Some(out) = tsv.as_mut() {
                        let _ = writeln!(
                            out,
                            "{}\t{}\t{}\t{:.2}\t{}\t{:.4}\t{}\t{}",
                            img_name,
                            w,
                            h,
                            target,
                            m.passes_used,
                            m.achieved_score,
                            b.len(),
                            u8::from(m.targets_met),
                        );
                    }
                    let a = agg_one(target, cli.max_overshoot, m, b.len());
                    fold(&mut overall, &a);
                    fold(by_target.entry((target * 100.0) as u32).or_default(), &a);
                }
                Err(e) => eprintln!("CLIC ERROR {img_name} target={target}: {:?}", e),
            }
        }
        eprint!(".");
    }
    eprintln!();
    if let Some(mut w) = tsv {
        let _ = w.flush();
    }
    println!("--- CLIC by target ---");
    for (t100, a) in &by_target {
        print_summary("CLIC1024", Some(*t100 as f32 / 100.0), a);
    }
    print_summary("CLIC1024 ALL", None, &overall);
    eprintln!("CLIC TSV: {}", out_path.display());
}

// --------------------------------------------------------------------
// 3b: RGBA "unsupported-layout" smoke test
// --------------------------------------------------------------------

/// Confirm the encoder cleanly refuses RGBA + target_zensim with the
/// typed [`EncodeError::TargetZensimUnsupportedLayout`] error. This
/// documents the current limitation in code: an earlier iteration of
/// PR #48 silently dropped alpha into a transient RGB buffer for the
/// iteration probes — that hid the unhonored-target footgun. The gate
/// now refuses non-RGB8 inputs whenever target_zensim is set, leaving
/// proper RGBA support as a separate follow-up.
///
/// Synthetic 32x32 RGBA image, single target. We're checking the
/// gate, not convergence quality — a tiny image keeps the test fast
/// even though it never actually reaches the iteration loop.
fn run_rgba_unsupported_smoke() {
    let w: u32 = 32;
    let h: u32 = 32;
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w) as u8;
            let g = ((y * 255) / h) as u8;
            let b = (((x + y) * 255) / (w + h)) as u8;
            let a = if (x + y) & 1 == 0 { 255 } else { 128 };
            rgba.extend_from_slice(&[r, g, b, a]);
        }
    }
    let cfg = LossyConfig::new()
        .with_method(4)
        .with_target_zensim(ZensimTarget::new(80.0));
    let r = EncodeRequest::lossy(&cfg, &rgba, PixelLayout::Rgba8, w, h).encode_with_metrics();
    match r {
        Ok((bytes, m)) => {
            panic!(
                "expected TargetZensimUnsupportedLayout error for RGBA + target_zensim, \
                 got Ok({} bytes, achieved={:.2}, passes={})",
                bytes.len(),
                m.achieved_score,
                m.passes_used,
            );
        }
        Err(at) => {
            // Decompose to inspect the inner variant. Other variants
            // would mean the gate is misbehaving.
            let inner: EncodeError = at.into();
            match inner {
                EncodeError::TargetZensimUnsupportedLayout(layout) => {
                    eprintln!(
                        "  ok: RGBA + target_zensim -> TargetZensimUnsupportedLayout({:?})",
                        layout,
                    );
                    println!(
                        "RGBA-smoke: PASS — target_zensim cleanly rejects PixelLayout::{:?}",
                        layout,
                    );
                }
                other => {
                    panic!(
                        "expected TargetZensimUnsupportedLayout, got: {:?} ({})",
                        other, other,
                    );
                }
            }
        }
    }
}

// --------------------------------------------------------------------
// 3c: Animation per-frame
// --------------------------------------------------------------------

fn run_anim_phase(cli: &Cli) {
    let bytes = match fs::read(&cli.anim) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("WARN: cannot read anim {}: {e:?}", cli.anim.display());
            return;
        }
    };
    let demuxer = match zenwebp::mux::WebPDemuxer::new(&bytes) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("WARN: demux failed: {:?}", e);
            return;
        }
    };
    eprintln!(
        "anim: {}x{}, {} frames, animated={}",
        demuxer.canvas_width(),
        demuxer.canvas_height(),
        demuxer.num_frames(),
        demuxer.is_animated(),
    );

    let target: f32 = cli
        .targets
        .iter()
        .copied()
        .find(|&t| (t - 80.0).abs() < 0.01)
        .unwrap_or(80.0);
    let max_passes = cli.max_passes.max(2);

    use std::collections::BTreeMap;
    let mut by_frame: BTreeMap<u32, Agg> = BTreeMap::new();
    let mut overall = Agg::default();

    for n in 1..=demuxer.num_frames() {
        // Re-decode the whole anim at frame n (compose-over canvas).
        let rgba = match decode_anim_frame_rgba(&bytes, n) {
            Some(t) => t,
            None => {
                eprintln!("anim: frame {n} decode failed");
                continue;
            }
        };
        let (rgba_buf, w, h) = rgba;
        // target_zensim is RGB8-only; explicitly drop alpha here at
        // the test driver so the encoder gate doesn't refuse the
        // input. Alpha-preserving target_zensim is a tracked
        // follow-up; this test is purely about per-frame iteration
        // convergence on the underlying RGB content.
        let mut rgb_buf = Vec::with_capacity(rgba_buf.len() / 4 * 3);
        for px in rgba_buf.chunks_exact(4) {
            rgb_buf.extend_from_slice(&[px[0], px[1], px[2]]);
        }
        let cfg = LossyConfig::new()
            .with_method(cli.method)
            .with_target_zensim(
                ZensimTarget::new(target)
                    .with_max_overshoot(Some(cli.max_overshoot))
                    .with_max_passes(max_passes),
            );
        let r = EncodeRequest::lossy(&cfg, &rgb_buf, PixelLayout::Rgb8, w, h).encode_with_metrics();
        match r {
            Ok((b, m)) => {
                let a = agg_one(target, cli.max_overshoot, m, b.len());
                fold(&mut overall, &a);
                fold(by_frame.entry(n).or_default(), &a);
                eprintln!(
                    "  frame {n}: {}x{} achieved={:.2} passes={} bytes={} met={}",
                    w,
                    h,
                    m.achieved_score,
                    m.passes_used,
                    b.len(),
                    m.targets_met,
                );
            }
            Err(e) => eprintln!("anim frame {n} ERROR: {:?}", e),
        }
    }
    println!("--- anim per-frame (target={target}, max_passes={max_passes}) ---");
    for (n, a) in &by_frame {
        print_summary(&format!("frame_{n}"), Some(target), a);
    }
    print_summary("anim ALL frames", Some(target), &overall);
}

/// Decode a single animation frame (1-based) of a WebP file to RGBA.
/// Uses the zenwebp full-frame oneshot decoder against the per-frame
/// bitstream extracted by the demuxer. This bypasses canvas
/// composition (dispose/blend) — adequate for confirming the encoder
/// loop converges; not a faithful playback.
fn decode_anim_frame_rgba(data: &[u8], n: u32) -> Option<(Vec<u8>, u32, u32)> {
    let demuxer = zenwebp::mux::WebPDemuxer::new(data).ok()?;
    let frame = demuxer.frame(n)?;
    // Extract per-frame WebP bitstream and decode standalone. We need
    // a complete WebP (RIFF wrapper) — rebuild one around the VP8/VP8L
    // bitstream so the standard decoder accepts it.
    let mut webp = Vec::with_capacity(frame.bitstream.len() + 32);
    webp.extend_from_slice(b"RIFF");
    let body_size: u32 = 4 + 8 + frame.bitstream.len() as u32;
    webp.extend_from_slice(&body_size.to_le_bytes());
    webp.extend_from_slice(b"WEBP");
    let chunk = if frame.is_lossy { *b"VP8 " } else { *b"VP8L" };
    webp.extend_from_slice(&chunk);
    webp.extend_from_slice(&(frame.bitstream.len() as u32).to_le_bytes());
    webp.extend_from_slice(frame.bitstream);
    if frame.bitstream.len() % 2 == 1 {
        webp.push(0);
    }
    let (rgba, w, h) = zenwebp::oneshot::decode_rgba(&webp).ok()?;
    Some((rgba, w, h))
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
