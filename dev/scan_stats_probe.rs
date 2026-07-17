//! Per-config I4/I16/skip stats probe on scan images (tuned vs parity).
#![forbid(unsafe_code)]
use zenwebp::{CostModel, EncodeRequest, LossyConfig, PixelLayout};

fn load_png(path: &str) -> (Vec<u8>, u32, u32) {
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
        _ => panic!("layout"),
    };
    (rgb, info.width, info.height)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    println!("image\tmodel\tm\tq\tsize\ti4\ti16\tskip\tsegs_used");
    for path in &args[1..] {
        let (rgb, w, h) = load_png(path);
        let name = std::path::Path::new(path)
            .file_stem()
            .unwrap()
            .to_string_lossy();
        for &m in &[4u8, 6] {
            for &q in &[25u8, 50, 75] {
                for (label, model) in [
                    ("tuned", CostModel::ZenwebpDefault),
                    ("parity", CostModel::StrictLibwebpParity),
                ] {
                    let cfg = LossyConfig::new()
                        .with_quality(f32::from(q))
                        .with_method(m)
                        .with_cost_model(model);
                    let (out, stats) = EncodeRequest::lossy(&cfg, &rgb, PixelLayout::Rgb8, w, h)
                        .encode_with_stats()
                        .unwrap();
                    println!(
                        "{name}\t{label}\t{m}\t{q}\t{}\t{}\t{}\t{}\t{:?}\t{:?}\t{}\t{}",
                        out.len(),
                        stats.block_count_i4,
                        stats.block_count_i16,
                        stats.block_count_skip,
                        stats.segment_quant,
                        stats.segment_size,
                        stats.header_bytes,
                        stats.mode_partition_bytes
                    );
                }
            }
        }
    }
}
