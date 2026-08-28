//! #71 probe: where do the bytes go when zenwebp's 16px-tile predictor
//! selection loses to libwebp on a screenshot?
//!
//! For each PNG × method, encodes losslessly (exact=true both sides) with
//! zenwebp and libwebp (webpx), then parses BOTH streams' transform lists
//! with zenwebp's decoder (`mode_debug` diagnostic) and reports, per
//! transform, the bits it cost on the wire, the predictor tile grid, the
//! per-mode histogram, the tile-by-tile agreement between the two mode
//! maps, and the bits left for the main image. That splits the size gap
//! into "mode image coding" vs "residual coding" — suspects (1) and (3) of
//! the issue.
//!
//! Usage:
//!   cargo run --release --features mode_debug --example issue71_probe -- \
//!       [--methods 4,5] <png> [<png> ...]

use std::path::Path;

fn load_png_rgba(path: &Path) -> (Vec<u8>, u32, u32) {
    let file = std::fs::File::open(path).expect("open png");
    let mut d = png::Decoder::new(std::io::BufReader::new(file));
    d.set_transformations(png::Transformations::normalize_to_color8());
    let mut r = d.read_info().expect("png info");
    let mut buf = vec![0u8; r.output_buffer_size().expect("png size")];
    let info = r.next_frame(&mut buf).expect("png frame");
    buf.truncate(info.buffer_size());
    let rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => buf
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        other => panic!("unsupported png color type {other:?}"),
    };
    (rgba, info.width, info.height)
}

const KIND: [&str; 4] = ["predictor", "color", "subtract-green", "color-indexing"];

/// Wrap a raw VP8L payload in a simple RIFF/WEBP container.
fn riff_vp8l(payload: &[u8]) -> Vec<u8> {
    let padded = payload.len() + (payload.len() & 1);
    let mut out = Vec::with_capacity(20 + padded);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((4 + 8 + padded) as u32).to_le_bytes());
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(b"VP8L");
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() & 1 == 1 {
        out.push(0);
    }
    out
}

struct Side {
    name: &'static str,
    bytes: Vec<u8>,
    dump: Vec<(u8, u8, u64, Vec<u8>)>,
    main_start: u64,
    total_bits: u64,
    main: zenwebp::decoder::Vp8lMainImageInfo,
}

fn side(name: &'static str, bytes: Vec<u8>) -> Side {
    let (dump, main_start, total_bits, main) =
        zenwebp::__test_helpers::vp8l_transform_dump(&bytes).expect("transform dump");
    Side {
        name,
        bytes,
        dump,
        main_start,
        total_bits,
        main,
    }
}

fn mode_histogram(data: &[u8]) -> [u32; 16] {
    let mut h = [0u32; 16];
    for tile in data.as_chunks::<4>().0 {
        h[(tile[1] & 15) as usize] += 1;
    }
    h
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut methods = vec![4u8, 5];
    // `--parity`: encode zenwebp's side with `Vp8lConfig::parity = true`
    // (libwebp's hash-chain iteration accounting instead of the tuned
    // default) to separate deliberate tuning from porting gaps.
    let parity = if let Some(i) = args.iter().position(|a| a == "--parity") {
        args.remove(i);
        true
    } else {
        false
    };
    if let Some(i) = args.iter().position(|a| a == "--methods") {
        methods = args[i + 1]
            .split(',')
            .map(|m| m.parse().expect("method"))
            .collect();
        args.drain(i..=i + 1);
    }
    if args.is_empty() {
        eprintln!("usage: issue71_probe [--methods 4,5] <png>...");
        std::process::exit(2);
    }

    for path in &args {
        let path = Path::new(path);
        let (rgba, w, h) = load_png_rgba(path);
        let name = path.file_stem().unwrap().to_string_lossy();
        // For dev/libwebp-histo-trace: dump the raw RGBA the encoders see.
        if let Ok(dir) = std::env::var("ISSUE71_DUMP_RGBA") {
            std::fs::create_dir_all(&dir).expect("dump dir");
            let out = Path::new(&dir).join(format!("{name}.rgba"));
            std::fs::write(&out, &rgba).expect("dump rgba");
            eprintln!("dumped {} ({w}x{h})", out.display());
        }
        for &m in &methods {
            let zen = if parity {
                let cfg = zenwebp::encoder::vp8l::Vp8lConfig {
                    quality: zenwebp::encoder::vp8l::Vp8lQuality {
                        quality: 75,
                        method: m,
                    },
                    exact: true,
                    parity: true,
                    ..zenwebp::encoder::vp8l::Vp8lConfig::default()
                };
                let payload = zenwebp::encoder::vp8l::encode_vp8l(
                    &rgba,
                    w,
                    h,
                    true,
                    &cfg,
                    &enough::Unstoppable,
                )
                .expect("zen parity encode");
                riff_vp8l(&payload)
            } else {
                let zen_cfg = zenwebp::LosslessConfig::new()
                    .with_method(m)
                    .with_exact(true);
                zenwebp::EncodeRequest::lossless(&zen_cfg, &rgba, zenwebp::PixelLayout::Rgba8, w, h)
                    .encode()
                    .expect("zen encode")
            };
            let lib = webpx::EncoderConfig::new_lossless()
                .method(m)
                .exact(true)
                .encode_rgba(&rgba, w, h, webpx::Unstoppable)
                .expect("libwebp encode");
            let sides = [side("zen", zen), side("lib", lib)];
            println!(
                "\n=== {name} {w}x{h} m{m}: zen {} B, lib {} B, zen/lib {:.4}",
                sides[0].bytes.len(),
                sides[1].bytes.len(),
                sides[0].bytes.len() as f64 / sides[1].bytes.len() as f64
            );
            for s in &sides {
                let tf_bits: u64 = s.dump.iter().map(|d| d.2).sum();
                println!(
                    "  {}: header+transforms {} bits ({} B), main image {} bits ({} B) of {} bits",
                    s.name,
                    s.main_start,
                    s.main_start / 8,
                    s.total_bits - s.main_start,
                    (s.total_bits - s.main_start) / 8,
                    s.total_bits
                );
                for (kind, size_bits, bits, data) in &s.dump {
                    let kind_name = KIND[(*kind & 3) as usize];
                    if *kind == 0 || *kind == 1 {
                        let tw = w.div_ceil(1 << size_bits);
                        let th = h.div_ceil(1 << size_bits);
                        print!(
                            "    {kind_name:15} bits={size_bits} tiles={tw}x{th} ({} tiles) cost={} bits = {:.2} bits/tile",
                            tw * th,
                            bits,
                            *bits as f64 / f64::from(tw * th)
                        );
                        if *kind == 0 {
                            let hist = mode_histogram(data);
                            print!("  modes=");
                            for (i, c) in hist.iter().enumerate().take(14) {
                                if *c > 0 {
                                    print!("{i}:{c} ");
                                }
                            }
                        }
                        println!();
                    } else {
                        println!("    {kind_name:15} cost={bits} bits");
                    }
                }
                let _ = tf_bits;
                println!(
                    "    main image: cache_bits={:?} histo_bits={} groups={} entropy_image={} bits ({} B) huffman_tables={} bits ({} B) pixels={} bits ({} B)",
                    s.main.cache_bits,
                    s.main.histo_bits,
                    s.main.num_groups,
                    s.main.entropy_image_bits,
                    s.main.entropy_image_bits / 8,
                    s.main.huffman_tables_bits,
                    s.main.huffman_tables_bits / 8,
                    s.main.pixel_data_bits,
                    s.main.pixel_data_bits / 8
                );
                let [lit, cache, copies, copied] = s.main.tokens;
                println!(
                    "    main tokens: literals={lit} cache_hits={cache} copies={copies} copied_pixels={copied} (avg copy len {:.1}, {:.1}% of pixels via copies)",
                    copied as f64 / copies.max(1) as f64,
                    100.0 * copied as f64 / f64::from(w * h)
                );
            }
            // Tile-by-tile predictor agreement when both streams use the same grid.
            let zp = sides[0].dump.iter().find(|d| d.0 == 0);
            let lp = sides[1].dump.iter().find(|d| d.0 == 0);
            if let (Some(z), Some(l)) = (zp, lp) {
                if z.1 == l.1 && z.3.len() == l.3.len() {
                    let n = z.3.len() / 4;
                    let same =
                        z.3.as_chunks::<4>()
                            .0
                            .iter()
                            .zip(l.3.as_chunks::<4>().0)
                            .filter(|(a, b)| a[1] & 15 == b[1] & 15)
                            .count();
                    // Confusion: for tiles that differ, which (zen, lib) pairs dominate.
                    let mut conf = std::collections::BTreeMap::<(u8, u8), u32>::new();
                    for (a, b) in z.3.as_chunks::<4>().0.iter().zip(l.3.as_chunks::<4>().0) {
                        if a[1] & 15 != b[1] & 15 {
                            *conf.entry((a[1] & 15, b[1] & 15)).or_default() += 1;
                        }
                    }
                    let mut conf: Vec<_> = conf.into_iter().collect();
                    conf.sort_by_key(|&(_, c)| core::cmp::Reverse(c));
                    println!(
                        "  predictor tiles identical: {same}/{n} ({:.1}%); top (zen→lib) disagreements: {}",
                        100.0 * same as f64 / n as f64,
                        conf.iter()
                            .take(8)
                            .map(|((a, b), c)| format!("{a}→{b}:{c}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                } else {
                    println!(
                        "  predictor grids differ: zen bits={} lib bits={}",
                        z.1, l.1
                    );
                }
            }
            let zc = sides[0].dump.iter().find(|d| d.0 == 1);
            let lc = sides[1].dump.iter().find(|d| d.0 == 1);
            if let (Some(z), Some(l)) = (zc, lc)
                && z.1 == l.1
                && z.3.len() == l.3.len()
            {
                let n = z.3.len() / 4;
                let same =
                    z.3.as_chunks::<4>()
                        .0
                        .iter()
                        .zip(l.3.as_chunks::<4>().0)
                        .filter(|(a, b)| a[..3] == b[..3])
                        .count();
                println!(
                    "  cross-color tiles identical: {same}/{n} ({:.1}%)",
                    100.0 * same as f64 / n as f64
                );
            }
        }
    }
}
