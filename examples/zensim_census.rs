// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! Target-zensim INSTRUMENT census (registration:
//! benchmarks/zensim_instrument_census_2026-08-27.md): drive the closed-loop
//! `ZensimTarget` encoder over the corpus9 instrument (9 refs × targets) and
//! judge DECODED pixels with the SAME published-zensim calls the loop itself
//! uses (`Zensim::new(latest())` + `compute_with_ref_and_diffmap`) — the
//! self-consistent `err_pub02` the phase-B gate reads. Decoded PNGs are saved
//! so the driver can add the fleet-standard judge column separately.
//!
//! Usage:
//!   cargo run --release --features target-zensim --example zensim_census -- \
//!     <corpus9.tsv> <targets-csv> <max_passes> <out-dir>
//! corpus TSV rows: abs_path\tname\tclass

use std::io::Write;

use zenwebp::{EncodeRequest, LossyConfig, PixelLayout, ZensimTarget};

fn load_rgb8(path: &str) -> (Vec<u8>, u32, u32) {
    let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).expect("open")));
    let mut reader = dec.read_info().expect("png info");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    buf.truncate(info.buffer_size());
    let (w, h) = (info.width, info.height);
    match info.color_type {
        png::ColorType::Rgb => (buf, w, h),
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity((w * h * 3) as usize);
            for px in buf.chunks_exact(4) {
                rgb.extend_from_slice(&px[..3]);
            }
            (rgb, w, h)
        }
        png::ColorType::Grayscale => {
            let mut rgb = Vec::with_capacity((w * h * 3) as usize);
            for &g in &buf {
                rgb.extend_from_slice(&[g, g, g]);
            }
            (rgb, w, h)
        }
        other => panic!("unsupported png color type {other:?} for {path}"),
    }
}

fn pub02_score(reference: &[u8], decoded: &[u8], w: u32, h: u32) -> f32 {
    let z = zensim::Zensim::new(zensim::ZensimProfile::latest());
    let rc: &[[u8; 3]] = bytemuck::cast_slice(&reference[..(w * h * 3) as usize]);
    let rs = zensim::RgbSlice::new(rc, w as usize, h as usize);
    let pre = z.precompute_reference(&rs).expect("precompute");
    let dc: &[[u8; 3]] = bytemuck::cast_slice(&decoded[..(w * h * 3) as usize]);
    let ds = zensim::RgbSlice::new(dc, w as usize, h as usize);
    let r = z
        .compute_with_ref_and_diffmap(&pre, &ds, zensim::DiffmapWeighting::Trained)
        .expect("compute");
    r.score() as f32
}

fn write_png(path: &std::path::Path, rgb: &[u8], w: u32, h: u32) {
    let f = std::fs::File::create(path).expect("create png");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().expect("png header");
    wr.write_image_data(rgb).expect("png data");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (corpus, targets_csv, passes, out_dir) = (&args[1], &args[2], &args[3], &args[4]);
    let max_passes: u8 = passes.parse().expect("max_passes");
    let targets: Vec<f32> = targets_csv
        .split(',')
        .map(|t| t.trim().parse().expect("target"))
        .collect();
    std::fs::create_dir_all(out_dir).expect("out dir");
    let mut tsv = std::fs::File::create(format!("{out_dir}/census_k{max_passes}.tsv")).unwrap();
    writeln!(
        tsv,
        "image\tclass\ttarget\tpasses_used\tachieved_inloop\terr_pub02\tscore_pub02\tbytes\ttargets_met\tencode_ms"
    )
    .unwrap();
    for line in std::fs::read_to_string(corpus).expect("corpus").lines() {
        let mut f = line.split('\t');
        let (path, name, class) = (
            f.next().expect("path"),
            f.next().expect("name"),
            f.next().unwrap_or("image"),
        );
        let (rgb, w, h) = load_rgb8(path);
        for &t in &targets {
            // SHIPPED band contract (default overshoot/undershoot): the band
            // is what TRIGGERS iteration — band=None ships pass 1 always
            // (measured: k2≡k3, med passes 1.0 — that run is kept as the
            // pass-1 anchor-accuracy row, it is not the census).
            let zt = ZensimTarget::new(t).with_max_passes(max_passes);
            let cfg = LossyConfig::new().with_target_zensim(zt);
            let t0 = std::time::Instant::now();
            let (bytes, metrics) =
                EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, w, h)
                    .encode_with_metrics()
                    .expect("encode");
            let encode_ms = t0.elapsed().as_secs_f64() * 1e3;
            let (dec, dw, dh) = zenwebp::decoder::decode_rgb(&bytes).expect("decode");
            assert_eq!((dw, dh), (w, h), "decode dims");
            let s = pub02_score(&rgb, &dec, w, h);
            write_png(
                &std::path::Path::new(out_dir).join(format!("{name}_t{t:.0}_k{max_passes}.png")),
                &dec,
                w,
                h,
            );
            writeln!(
                tsv,
                "{name}\t{class}\t{t:.0}\t{}\t{:.3}\t{:.3}\t{s:.3}\t{}\t{}\t{encode_ms:.1}",
                metrics.passes_used,
                metrics.achieved_score,
                (s - t).abs(),
                metrics.bytes,
                metrics.targets_met,
            )
            .unwrap();
            eprintln!("{name} t{t:.0} k{max_passes}: passes={} inloop={:.2} pub02={s:.2}", metrics.passes_used, metrics.achieved_score);
        }
    }
    println!("census written to {out_dir}");
}
