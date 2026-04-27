//! Compare `CostModel::ZenwebpDefault` vs `CostModel::StrictLibwebpParity`
//! on a corpus sample. Outputs aggregate size + quality deltas.
//!
//! Usage:
//!   cargo run --release --example cost_model_sweep -- <out.tsv> <q_csv> <m_csv> <preset_csv> <sample> <corpus_label:path>
//!
//! E.g.:
//!   cargo run --release --example cost_model_sweep -- /tmp/cm.tsv 75 4 Default 25 \
//!       cid22:/home/lilith/work/codec-corpus/CID22/CID22-512/training

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use butteraugli::{ButteraugliParams, butteraugli};
use imgref::Img;
use rgb::RGB8;
use zenwebp::{CostModel, EncodeRequest, LossyConfig, PixelLayout, Preset};

struct Image {
    name: String,
    rgb: Vec<u8>,
    w: u32,
    h: u32,
}

fn load_png_rgb(path: &PathBuf) -> Option<Image> {
    let f = fs::File::open(path).ok()?;
    let dec = png::Decoder::new(std::io::BufReader::new(f));
    let mut r = dec.read_info().ok()?;
    let info = r.info();
    let (w, h) = (info.width, info.height);
    let mut buf = vec![0u8; r.output_buffer_size().unwrap_or(0)];
    r.next_frame(&mut buf).ok()?;
    let info = r.info();
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => buf
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        png::ColorType::GrayscaleAlpha => buf
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0]])
            .collect(),
        _ => return None,
    };
    Some(Image {
        name: path.file_name()?.to_string_lossy().into_owned(),
        rgb,
        w,
        h,
    })
}

fn parse_preset(s: &str) -> Preset {
    match s {
        "Default" => Preset::Default,
        "Photo" => Preset::Photo,
        "Drawing" => Preset::Drawing,
        "Picture" => Preset::Picture,
        "Auto" => Preset::Auto,
        _ => Preset::Default,
    }
}

fn encode(rgb: &[u8], w: u32, h: u32, p: Preset, q: f32, m: u8, cm: CostModel) -> Vec<u8> {
    let cfg = LossyConfig::with_preset(p, q)
        .with_method(m)
        .with_cost_model(cm);
    EncodeRequest::lossy(&cfg, rgb, PixelLayout::Rgb8, w, h)
        .encode()
        .unwrap()
}

fn decode_rgba(webp: &[u8]) -> Vec<u8> {
    zenwebp::oneshot::decode_rgba(webp).unwrap().0
}

fn rgba_to_rgb_vec(rgba: &[u8]) -> Vec<RGB8> {
    rgba.chunks_exact(4)
        .map(|p| RGB8 {
            r: p[0],
            g: p[1],
            b: p[2],
        })
        .collect()
}

fn rgb_to_rgb_vec(rgb: &[u8]) -> Vec<RGB8> {
    rgb.chunks_exact(3)
        .map(|p| RGB8 {
            r: p[0],
            g: p[1],
            b: p[2],
        })
        .collect()
}

fn butter_score(orig_rgb: &[u8], dec_rgba: &[u8], w: u32, h: u32) -> f64 {
    let orig = rgb_to_rgb_vec(orig_rgb);
    let dec = rgba_to_rgb_vec(dec_rgba);
    let params = ButteraugliParams::default();
    butteraugli(
        Img::new(orig, w as usize, h as usize).as_ref(),
        Img::new(dec, w as usize, h as usize).as_ref(),
        &params,
    )
    .map(|r| r.score)
    .unwrap_or(f64::NAN)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 7 {
        eprintln!(
            "usage: cost_model_sweep <out.tsv> <q_csv> <m_csv> <preset_csv> <sample> <corpus:path>..."
        );
        std::process::exit(1);
    }
    let out_path = &args[1];
    let qs: Vec<f32> = args[2].split(',').map(|s| s.parse().unwrap()).collect();
    let ms: Vec<u8> = args[3].split(',').map(|s| s.parse().unwrap()).collect();
    let presets: Vec<Preset> = args[4].split(',').map(parse_preset).collect();
    let sample: usize = args[5].parse().unwrap();

    let mut images = Vec::new();
    for arg in &args[6..] {
        let (label, path) = arg.split_once(':').unwrap();
        let mut entries: Vec<_> = fs::read_dir(path)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
            .collect();
        entries.sort();
        for p in entries.iter().take(sample) {
            if let Some(mut img) = load_png_rgb(p) {
                img.name = format!("{label}/{}", img.name);
                images.push(img);
            }
        }
    }
    eprintln!("Loaded {} images", images.len());

    let mut out = fs::File::create(out_path).unwrap();
    writeln!(
        out,
        "file\twidth\theight\tpreset\tq\tm\tzen_def_b\tzen_strict_b\tlib_b\tdef_butter\tstrict_butter\tlib_butter\tdef_size_ratio\tstrict_size_ratio\tdef_butter_delta\tstrict_butter_delta"
    )
    .unwrap();

    let total = images.len() * presets.len() * qs.len() * ms.len();
    let mut done = 0;
    let start = Instant::now();
    for img in &images {
        for &p in &presets {
            for &q in &qs {
                for &m in &ms {
                    done += 1;
                    let zen_def =
                        encode(&img.rgb, img.w, img.h, p, q, m, CostModel::ZenwebpDefault);
                    let zen_strict = encode(
                        &img.rgb,
                        img.w,
                        img.h,
                        p,
                        q,
                        m,
                        CostModel::StrictLibwebpParity,
                    );
                    // libwebp baseline via webpx
                    let lib_preset = match p {
                        Preset::Photo => webpx::Preset::Photo,
                        Preset::Drawing => webpx::Preset::Drawing,
                        _ => webpx::Preset::Default,
                    };
                    let lib = webpx::EncoderConfig::with_preset(lib_preset, q)
                        .method(m)
                        .encode_rgb(&img.rgb, img.w, img.h, webpx::Unstoppable)
                        .unwrap();

                    let dec_def = decode_rgba(&zen_def);
                    let dec_strict = decode_rgba(&zen_strict);
                    let dec_lib = decode_rgba(&lib);

                    let bd = butter_score(&img.rgb, &dec_def, img.w, img.h);
                    let bs = butter_score(&img.rgb, &dec_strict, img.w, img.h);
                    let bl = butter_score(&img.rgb, &dec_lib, img.w, img.h);

                    writeln!(
                        out,
                        "{}\t{}\t{}\t{:?}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:+.4}\t{:+.4}",
                        img.name,
                        img.w,
                        img.h,
                        p,
                        q,
                        m,
                        zen_def.len(),
                        zen_strict.len(),
                        lib.len(),
                        bd,
                        bs,
                        bl,
                        zen_def.len() as f64 / lib.len() as f64,
                        zen_strict.len() as f64 / lib.len() as f64,
                        bd - bl,
                        bs - bl,
                    )
                    .unwrap();
                }
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        eprintln!(
            "[{done}/{total}] {} ({:.0} cells/s)",
            img.name,
            done as f64 / elapsed
        );
    }
    eprintln!(
        "Done in {:.1}s -> {}",
        start.elapsed().as_secs_f64(),
        out_path
    );
}
