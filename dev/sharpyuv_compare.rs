//! Sharp-YUV quality comparison: zenyuv's sharp converter vs libwebp's
//! SharpYUV library vs the zenwebp SharpYUV PORT (#38 "dig into yuv").
//!
//! zenyuv's sharp-YUV is a redesign (Newton step with the inverse-matrix
//! Jacobian, f32 math, 2 iterations, optional Y refinement), not a port of
//! libwebp's fixed-point 4-iteration forward-gradient loop — so byte parity
//! is impossible by construction and the meaningful question is QUALITY:
//! for the same encoder settings, which converter yields better decoded
//! output per byte?
//!
//! For each corpus PNG × quality: encode with sharp YUV enabled through
//! zenwebp (tuned default), through the `encoder::sharpyuv` port (its planes
//! fed back via `PixelLayout::Yuv420` so only the converter differs), and
//! through libwebp (webpx); decode, score zensim, print one TSV row.
//!
//! Usage:
//!   cargo run --release --features __expert --example sharpyuv_compare -- \
//!       <corpus_dir> [qs=50,75,90] [method=4]

use zensim::{RgbSlice, Zensim, ZensimProfile};
use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};

fn load_png(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
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

fn parse_u8_list(s: &str) -> Vec<u8> {
    s.split(',').filter_map(|v| v.trim().parse().ok()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = args.get(1).cloned().expect("corpus dir");
    let qs = parse_u8_list(&args.get(2).cloned().unwrap_or_else(|| "50,75,90".into()));
    let m: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);

    let mut images: Vec<std::path::PathBuf> = std::fs::read_dir(&corpus)
        .expect("corpus dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    images.sort();

    let z = Zensim::new(ZensimProfile::latest());
    println!(
        "image\tq\tzen_sharp_bytes\tzen_sharp_zsim\tport_sharp_bytes\tport_sharp_zsim\tlib_sharp_bytes\tlib_sharp_zsim\tzen_std_bytes\tzen_std_zsim"
    );
    for path in &images {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let Some((rgb, w, h)) = load_png(path) else {
            continue;
        };
        let src_chunks: &[[u8; 3]] = bytemuck::cast_slice(&rgb);
        let src_slice = RgbSlice::new(src_chunks, w as usize, h as usize);
        let pre = z.precompute_reference(&src_slice).expect("precompute");
        let score = |webp: &[u8]| -> f64 {
            let (dec, w2, h2) = zenwebp::oneshot::decode_rgb(webp).expect("decode");
            assert_eq!((w2, h2), (w, h));
            let n = (w as usize) * (h as usize) * 3;
            let dec_chunks: &[[u8; 3]] = bytemuck::cast_slice(&dec[..n]);
            let dec_slice = RgbSlice::new(dec_chunks, w as usize, h as usize);
            z.compute_with_ref(&pre, &dec_slice)
                .expect("zensim")
                .score()
        };

        for &q in &qs {
            let zen_sharp = EncodeRequest::lossy(
                &LossyConfig::new()
                    .with_quality(f32::from(q))
                    .with_method(m)
                    .with_sharp_yuv(true),
                &rgb,
                PixelLayout::Rgb8,
                w,
                h,
            )
            .encode()
            .expect("zen sharp encode");
            let zen_std = EncodeRequest::lossy(
                &LossyConfig::new().with_quality(f32::from(q)).with_method(m),
                &rgb,
                PixelLayout::Rgb8,
                w,
                h,
            )
            .encode()
            .expect("zen std encode");
            // The port's planes fed back through Yuv420 input: identical
            // tuned encoder, ONLY the RGB→YUV converter differs.
            let (py, pu, pv) = zenwebp::__expert::sharpyuv_convert_rgb(&rgb, w as u16, h as u16);
            let mut yuv = py;
            yuv.extend_from_slice(&pu);
            yuv.extend_from_slice(&pv);
            let port_sharp = EncodeRequest::lossy(
                &LossyConfig::new().with_quality(f32::from(q)).with_method(m),
                &yuv,
                PixelLayout::Yuv420,
                w,
                h,
            )
            .encode()
            .expect("port sharp encode");
            let lib_sharp = webpx::EncoderConfig::new()
                .quality(f32::from(q))
                .method(m)
                .sharp_yuv(true)
                .encode_rgb(&rgb, w, h, webpx::Unstoppable)
                .expect("lib sharp encode");

            println!(
                "{name}\t{q}\t{}\t{:.3}\t{}\t{:.3}\t{}\t{:.3}\t{}\t{:.3}",
                zen_sharp.len(),
                score(&zen_sharp),
                port_sharp.len(),
                score(&port_sharp),
                lib_sharp.len(),
                score(&lib_sharp),
                zen_std.len(),
                score(&zen_std),
            );
        }
        eprintln!("done: {name}");
    }
}
