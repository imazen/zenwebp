//! SharpYUV port verification vs libwebp's `SharpYuvConvert` (#38).
//!
//! Compares `encoder::sharpyuv`'s output planes byte-for-byte against a
//! plane dump produced by the instrumented libwebp tree
//! (`~/work/zen/libwebp--zen38trace/dump_sharpyuv W H SEED`), using the SAME
//! deterministic synthetic generator on both sides.
//!
//! Usage:
//!   cargo run --release --features __expert --example sharpyuv_selftest -- \
//!       <planes.bin> <W> <H> <SEED>
//!   cargo run ... -- <planes.bin> <W> <H> --rgb <raw_rgb_file>
//!   cargo run ... -- --bench <W> <H> [iters=20]
//!
//! The second form compares against a real image dumped by the trace tree's
//! `dump_encoder_yuv` (webpx-flow ARGB import + WebPPictureSharpARGBToYUVA).
//! `--bench` times the port converter (min of N runs) for adoption calls.

fn synth(w: usize, h: usize, mut s: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let p = &mut rgb[(y * w + x) * 3..][..3];
            p[0] = ((x * 255 / (w - 1).max(1)) as u8) ^ (s & 15) as u8;
            p[1] = ((y * 255 / (h - 1).max(1)) as u8) ^ ((s >> 8) & 15) as u8;
            p[2] = (((x + y) * 127 / (w + h)) as u8) ^ ((s >> 16) & 15) as u8;
        }
    }
    rgb
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args[1] == "--bench" {
        let w: usize = args[2].parse().unwrap();
        let h: usize = args[3].parse().unwrap();
        let iters: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20);
        let rgb = synth(w, h, 99);
        let mut best = f64::MAX;
        for _ in 0..iters {
            let t = std::time::Instant::now();
            let (y, _, _) = zenwebp::__expert::sharpyuv_convert_rgb(&rgb, w as u16, h as u16);
            best = best.min(t.elapsed().as_secs_f64());
            std::hint::black_box(y);
        }
        println!(
            "port sharpyuv {w}x{h}: min {:.3} ms ({:.1} Mpix/s)",
            best * 1e3,
            (w * h) as f64 / best / 1e6
        );
        return;
    }
    let planes = std::fs::read(&args[1]).expect("planes.bin from dump_sharpyuv");
    let w: usize = args[2].parse().unwrap();
    let h: usize = args[3].parse().unwrap();
    let (uvw, uvh) = (w.div_ceil(2), h.div_ceil(2));
    assert_eq!(planes.len(), w * h + 2 * uvw * uvh, "plane dump size");

    let (rgb, seed) = if args[4] == "--rgb" {
        let raw = std::fs::read(&args[5]).expect("raw rgb file");
        assert_eq!(raw.len(), w * h * 3, "raw rgb size");
        (raw, 0)
    } else {
        let seed: u32 = args[4].parse().unwrap();
        (synth(w, h, seed), seed)
    };
    let (y, u, v) = zenwebp::__expert::sharpyuv_convert_rgb(&rgb, w as u16, h as u16);

    let (ly, rest) = planes.split_at(w * h);
    let (lu, lv) = rest.split_at(uvw * uvh);
    let count_diff = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(x, y)| x != y).count();
    let (dy, du, dv) = (count_diff(&y, ly), count_diff(&u, lu), count_diff(&v, lv));
    println!(
        "{w}x{h} seed{seed}: Y diff {dy}/{} U diff {du}/{} V diff {dv}/{}",
        y.len(),
        u.len(),
        v.len()
    );
    if dy + du + dv == 0 {
        println!("IDENTICAL");
        return;
    }
    // Locate the first differing sample per plane for tracing.
    for (name, ours, libs, pw) in [
        ("Y", &y, &ly.to_vec(), w),
        ("U", &u, &lu.to_vec(), uvw),
        ("V", &v, &lv.to_vec(), uvw),
    ] {
        if let Some(i) = ours.iter().zip(libs.iter()).position(|(a, b)| a != b) {
            println!(
                "  first {name} diff at ({}, {}): zen={} lib={}",
                i % pw,
                i / pw,
                ours[i],
                libs[i]
            );
        }
    }
    std::process::exit(1);
}
