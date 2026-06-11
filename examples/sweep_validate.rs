//! Empirical validation of the curated sweep axes (`zenwebp::sweep`).
//!
//! Playbook patterns 6 + 14 + 15 (`zenjpeg/docs/VARIANT_GENERATION.md`):
//! encodes the default stratum plus every single-deviation stratum of
//! `modes_full` on a mixed corpus and hard-fails on inert steps,
//! undecodable cells (pattern 14: every cell must decode), lossless
//! roundtrip inexactness (zero-tolerance as a gate), queue-ordering
//! breakage, and ssim2 sanity-floor violations. The corpus crosses
//! WebP's partition topology (16-px macroblocks) via a 509×381 crop —
//! partial macroblocks on both axes (pattern 15).
//!
//! Run:
//! ```bash
//! GIT_COMMIT=$(git rev-parse --short HEAD) cargo run --release \
//!   --example sweep_validate --features __expert -- \
//!   --out benchmarks/sweep_validate_webp_$(date +%F).tsv
//! ```

use std::collections::HashMap;
use std::io::Write as _;

use zenwebp::sweep::{QualityGrid, SweepAxes, SweepBuilder, SweepVariant};
use zenwebp::{EncodeRequest, PixelLayout};

const DEFAULT_BASE: &str = "vp8-m4_def";
const Q_GRID: [f32; 3] = [10.0, 50.0, 85.0];

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

struct Image {
    name: String,
    w: u32,
    h: u32,
    rgb: Vec<u8>,
}

fn srgb_to_linear(v: u8) -> f32 {
    let f = f32::from(v) / 255.0;
    if f <= 0.04045 {
        f / 12.92
    } else {
        ((f + 0.055) / 1.055).powf(2.4)
    }
}

/// Decode the cell and score it. `Err` = the stream did not decode —
/// a HARD failure at any quality (pattern 14). `Ok(None)` = decoded
/// but ssim2 skipped (tiny image below metric minimum).
fn decode_and_score(img: &Image, webp: &[u8]) -> Result<(Vec<u8>, Option<f64>), String> {
    use fast_ssim2::{ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2};
    let (decoded, dw, dh) =
        zenwebp::decoder::decode_rgb(webp).map_err(|e| format!("decode failed: {e:?}"))?;
    if (dw, dh) != (img.w, img.h) {
        return Err(format!("decoded {dw}x{dh}, expected {}x{}", img.w, img.h));
    }
    if img.w < 64 || img.h < 64 {
        return Ok((decoded, None));
    }
    let lin = |bytes: &[u8]| -> Vec<[f32; 3]> {
        bytes
            .chunks_exact(3)
            .map(|p| {
                [
                    srgb_to_linear(p[0]),
                    srgb_to_linear(p[1]),
                    srgb_to_linear(p[2]),
                ]
            })
            .collect()
    };
    let mk = |px: Vec<[f32; 3]>| {
        Rgb::new(
            px,
            img.w as usize,
            img.h as usize,
            TransferCharacteristic::Linear,
            ColorPrimaries::BT709,
        )
        .map_err(|e| format!("ssim2 frame: {e:?}"))
    };
    let score = compute_frame_ssimulacra2(mk(lin(&img.rgb))?, mk(lin(&decoded))?)
        .map_err(|e| format!("ssim2: {e:?}"))?;
    Ok((decoded, Some(score)))
}

fn encode_cell(v: &SweepVariant, img: &Image) -> Result<Vec<u8>, String> {
    let cfg = v.build();
    EncodeRequest::new(&cfg, &img.rgb, PixelLayout::Rgb8, img.w, img.h)
        .encode()
        .map_err(|e| format!("encode failed: {e:?}"))
}

fn xorshift_noise(w: u32, h: u32, mut state: u32) -> Vec<u8> {
    (0..(w * h * 3) as usize)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

fn checkerboard(w: u32, h: u32, block: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let on = ((x / block) + (y / block)).is_multiple_of(2);
            let c = if on { 255 } else { 0 };
            v.extend_from_slice(&[c, c, c]);
        }
    }
    v
}

fn load_corpus() -> Vec<Image> {
    let mut images = Vec::new();
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
    let dir = corpus
        .get("CID22/CID22-512/validation")
        .expect("CID22-512 validation set");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("read corpus dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    files.sort();
    for p in files.iter().take(3) {
        let data = std::fs::read(p).expect("read png");
        let mut reader = png::Decoder::new(std::io::Cursor::new(data))
            .read_info()
            .expect("png info");
        let mut buf = vec![0u8; reader.output_buffer_size().expect("buffer size")];
        let info = reader.next_frame(&mut buf).expect("png frame");
        assert_eq!(info.color_type, png::ColorType::Rgb, "expect RGB corpus");
        images.push(Image {
            name: format!("cid_{}", p.file_stem().unwrap().to_string_lossy()),
            w: info.width,
            h: info.height,
            rgb: buf[..info.buffer_size()].to_vec(),
        });
    }
    // Pattern 15: cross the 16-px macroblock topology — partial MBs on
    // both axes (509 = 31×16+13, 381 = 23×16+13).
    {
        let src = &images[0];
        let (w, h) = (509u32, 381u32);
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            let off = (y * src.w * 3) as usize;
            rgb.extend_from_slice(&src.rgb[off..off + (w * 3) as usize]);
        }
        images.push(Image {
            name: "cid_odd509x381".into(),
            w,
            h,
            rgb,
        });
    }
    images.push(Image {
        name: "noise512".into(),
        w: 512,
        h: 512,
        rgb: xorshift_noise(512, 512, 0x9e37_79b9),
    });
    images.push(Image {
        name: "checker512".into(),
        w: 512,
        h: 512,
        rgb: checkerboard(512, 512, 8),
    });
    images.push(Image {
        name: "tiny48".into(),
        w: 48,
        h: 48,
        rgb: xorshift_noise(48, 48, 0x1234_5678),
    });
    images
}

fn parse_label(base: &str) -> (usize, String) {
    // Lossless ids are their own single-token grammar.
    if let Some(rest) = base.strip_prefix("vp8l-") {
        let devs = if rest == "m4" { 0 } else { 1 };
        return (devs, format!("vp8l-{rest}"));
    }
    let def: Vec<&str> = DEFAULT_BASE.splitn(2, '_').collect();
    let got: Vec<&str> = base.splitn(2, '_').collect();
    assert_eq!(got.len(), 2, "unparseable id {base}");
    let mut devs = Vec::new();
    if got[0] != def[0] {
        devs.push(got[0].to_string());
    }
    let mut labelflags = got[1].split('-');
    let label = labelflags.next().unwrap_or_default();
    if label != "def" {
        devs.push(label.to_string());
    }
    for flag in labelflags {
        devs.push(flag.to_string());
    }
    (devs.len(), devs.join("+"))
}

struct Measure {
    bytes: usize,
    hash: u64,
    ssim2: Option<f64>,
}

fn main() {
    let out_path = {
        let args: Vec<String> = std::env::args().collect();
        args.iter()
            .position(|a| a == "--out")
            .and_then(|i| args.get(i + 1).cloned())
            .unwrap_or_else(|| "benchmarks/sweep_validate_webp.tsv".to_string())
    };
    let mut hard_failures: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    let images = load_corpus();
    let plan = SweepBuilder::new(
        SweepAxes::modes_full(),
        QualityGrid::Explicit(Q_GRID.to_vec()),
    )
    .plan();
    println!(
        "plan: {} cells, {} merged",
        plan.cells.len(),
        plan.duplicates_merged
    );
    if plan.cells[0].deviations != 0 || !plan.cells[0].id.starts_with(DEFAULT_BASE) {
        hard_failures.push(format!("ordering: first cell {}", plan.cells[0].id));
    }
    if plan
        .cells
        .windows(2)
        .any(|w| w[1].deviations < w[0].deviations)
    {
        hard_failures.push("ordering: deviations not non-decreasing".into());
    }

    let subset: Vec<usize> = plan
        .cells
        .iter()
        .enumerate()
        .take_while(|(_, c)| c.deviations <= 1)
        .map(|(i, _)| i)
        .collect();
    println!("subset: {} cells x {} images", subset.len(), images.len());

    let t0 = std::time::Instant::now();
    let mut measures: HashMap<(usize, usize), Measure> = HashMap::new();
    for (ii, img) in images.iter().enumerate() {
        for &ci in &subset {
            let cell = &plan.cells[ci];
            let webp = match encode_cell(&cell.variant, img) {
                Ok(b) => b,
                Err(e) => {
                    hard_failures.push(format!("ENCODE FAILED: {} on {}: {e}", cell.id, img.name));
                    continue;
                }
            };
            let (decoded, score) = match decode_and_score(img, &webp) {
                Ok(x) => x,
                Err(e) => {
                    hard_failures.push(format!(
                        "UNDECODABLE CELL: {} on {}: {e}",
                        cell.id, img.name
                    ));
                    continue;
                }
            };
            // Zero-tolerance gate: lossless cells roundtrip EXACTLY.
            if matches!(cell.variant, SweepVariant::Lossless(_)) && decoded != img.rgb {
                hard_failures.push(format!(
                    "LOSSLESS ROUNDTRIP MISMATCH: {} on {}",
                    cell.id, img.name
                ));
            }
            measures.insert(
                (ci, ii),
                Measure {
                    bytes: webp.len(),
                    hash: fnv64(&webp),
                    ssim2: score,
                },
            );
        }
        println!("  done {}", img.name);
    }
    println!("encode+score: {:.1}s", t0.elapsed().as_secs_f64());

    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut tsv = std::fs::File::create(&out_path).expect("tsv create");
    writeln!(
        tsv,
        "# zenwebp sweep_validate: modes_full dev<=1, q={Q_GRID:?}\n# git_commit: {}\nimage\tcell\tlabel\tdeviations\tbytes\tssim2",
        std::env::var("GIT_COMMIT").unwrap_or_else(|_| "unknown".into()),
    )
    .unwrap();
    for (ii, img) in images.iter().enumerate() {
        for &ci in &subset {
            let Some(m) = measures.get(&(ci, ii)) else {
                continue;
            };
            let c = &plan.cells[ci];
            let base = c.id.rfind("_q").map(|at| &c.id[..at]).unwrap_or(&c.id);
            let (_, label) = parse_label(base);
            writeln!(
                tsv,
                "{}\t{}\t{label}\t{}\t{}\t{}",
                img.name,
                c.id,
                c.deviations,
                m.bytes,
                m.ssim2.map(|s| format!("{s:.3}")).unwrap_or_default()
            )
            .unwrap();
        }
    }
    println!("wrote {out_path}");

    // Per-label byte-diff rate vs the default stratum at the same q.
    let canon: HashMap<&str, usize> = plan
        .cells
        .iter()
        .enumerate()
        .map(|(i, c)| (c.id.as_str(), i))
        .collect();
    let baseline = |q: f32, ii: usize| -> Option<&Measure> {
        canon
            .get(format!("{DEFAULT_BASE}_q{q}").as_str())
            .and_then(|&ci| measures.get(&(ci, ii)))
    };
    let mut agg: HashMap<String, (usize, usize, f64)> = HashMap::new();
    for &ci in &subset {
        let c = &plan.cells[ci];
        if c.deviations != 1 {
            continue;
        }
        let base = c.id.rfind("_q").map(|at| &c.id[..at]).unwrap_or(&c.id);
        let (_, label) = parse_label(base);
        for (ii, _) in images.iter().enumerate() {
            let Some(m) = measures.get(&(ci, ii)) else {
                continue;
            };
            // Lossless cells compare against the lossless default m4.
            let b = match (&c.quality, canon.get("vp8l-m4")) {
                (Some(q), _) => baseline(*q, ii),
                (None, Some(&bi)) => measures.get(&(bi, ii)),
                _ => None,
            };
            let Some(b) = b else { continue };
            let e = agg.entry(label.clone()).or_insert((0, 0, 0.0));
            e.0 += 1;
            if m.hash != b.hash {
                e.1 += 1;
            }
            e.2 += (m.bytes as f64 - b.bytes as f64) / b.bytes as f64 * 100.0;
        }
    }
    let mut labels: Vec<_> = agg.into_iter().collect();
    labels.sort_by(|a, b| a.0.cmp(&b.0));
    println!(
        "\n{:<22} {:>4} {:>6} {:>9}",
        "label", "n", "diff%", "dsize%"
    );
    for (label, (n, d, ds)) in &labels {
        println!(
            "{label:<22} {n:>4} {:>5.0}% {:>8.2}%",
            *d as f64 / *n as f64 * 100.0,
            ds / *n as f64
        );
        if *d == 0 && label != "vp8l-m4" {
            // mpass is live at m4 only AND ~0.1% size — may legitimately
            // tie at q10 on tiny content; everything else must move bytes.
            if label == "mpass" {
                warnings.push(format!("{label}: byte-identical everywhere (gate-shadow?)"));
            } else {
                hard_failures.push(format!(
                    "INERT STEP: {label} never changed output bytes across {n} pairs"
                ));
            }
        }
    }

    // ssim2 sanity floor at q85 (content-relative).
    for (ii, img) in images.iter().enumerate() {
        let floor = if img.name.starts_with("noise") {
            15.0
        } else {
            30.0
        };
        for &ci in &subset {
            let c = &plan.cells[ci];
            if c.quality != Some(85.0) {
                continue;
            }
            if let Some(m) = measures.get(&(ci, ii))
                && let Some(s) = m.ssim2
                && s < floor
            {
                hard_failures.push(format!(
                    "ssim2 sanity: {} on {} scored {s:.1} (floor {floor})",
                    c.id, img.name
                ));
            }
        }
    }

    println!();
    for w in &warnings {
        println!("WARN {w}");
    }
    if hard_failures.is_empty() {
        println!("\nALL HARD CHECKS PASSED ({} warnings)", warnings.len());
    } else {
        println!("\n{} HARD FAILURES:", hard_failures.len());
        for f in &hard_failures {
            println!("FAIL {f}");
        }
        std::process::exit(1);
    }
}
