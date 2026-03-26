//! Verify lossless roundtrip pixel-exactness on real corpus images.
//! Checks all method levels and flags any where zenwebp < libwebp in size.

fn corpus_path(subdir: &str, filename: &str) -> Option<std::path::PathBuf> {
    let corpus = codec_corpus::Corpus::new().ok()?;
    let dir = corpus.get(subdir).ok()?;
    let path = dir.join(filename);
    if path.exists() { Some(path) } else { None }
}

fn load_png_rgba(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgb => buf[..info.buffer_size()]
            .chunks(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        _ => return None,
    };
    Some((rgba, info.width, info.height))
}

fn encode_zenwebp(rgba: &[u8], w: u32, h: u32, method: u8) -> Vec<u8> {
    let config = zenwebp::EncoderConfig::new_lossless().with_method(method);
    zenwebp::EncodeRequest::new(&config, rgba, zenwebp::PixelLayout::Rgba8, w, h)
        .encode()
        .unwrap()
}

fn encode_libwebp(rgba: &[u8], w: u32, h: u32, method: u8) -> Vec<u8> {
    webpx::EncoderConfig::new_lossless()
        .method(method)
        .encode_rgba(rgba, w, h, webpx::Unstoppable)
        .unwrap()
}

fn decode_zenwebp_rgba(webp: &[u8]) -> Vec<u8> {
    let config = zenwebp::DecodeConfig::default();
    {
        let (pixels, _w, _h) = zenwebp::DecodeRequest::new(&config, webp)
            .decode_rgba()
            .unwrap();
        pixels
    }
}

fn main() {
    let photos = [
        "792079.png",
        "750463.png",
        "725551.png",
        "890595.png",
        "816411.png",
    ];
    let screenshots = [
        "codec_wiki.png",
        "terminal.png",
        "windows.png",
        "imac_dark.png",
        "imac_g3.png",
    ];

    let mut images: Vec<(String, Vec<u8>, u32, u32)> = Vec::new();
    for f in &photos {
        if let Some(path) = corpus_path("CID22/CID22-512/validation", f) {
            if let Some((rgba, w, h)) = load_png_rgba(&path) {
                images.push((f.to_string(), rgba, w, h));
            }
        }
    }
    for f in &screenshots {
        if let Some(path) = corpus_path("gb82-sc", f) {
            if let Some((rgba, w, h)) = load_png_rgba(&path) {
                images.push((f.to_string(), rgba, w, h));
            }
        }
    }

    if images.is_empty() {
        eprintln!("No corpus images found");
        return;
    }

    println!(
        "{:<20} {:>6} {:>8} {:>8} {:>6} {:>12}",
        "image", "method", "zenwebp", "libwebp", "ratio", "rt_check"
    );
    println!("{:-<70}", "");

    let mut total_checked = 0u32;
    let mut total_failed = 0u32;
    let mut total_smaller = 0u32;

    for (name, rgba, w, h) in &images {
        for method in [0u8, 2, 4, 6] {
            let zen_webp = encode_zenwebp(rgba, *w, *h, method);
            let lib_webp = encode_libwebp(rgba, *w, *h, method);
            let zen_size = zen_webp.len();
            let lib_size = lib_webp.len();
            let ratio = zen_size as f64 / lib_size as f64;
            let smaller = zen_size <= lib_size;

            // Roundtrip check: decode zenwebp output and compare pixels
            let decoded = decode_zenwebp_rgba(&zen_webp);
            let expected_len = (*w as usize) * (*h as usize) * 4;

            let rt_ok = if decoded.len() != expected_len {
                false
            } else {
                decoded == *rgba
            };

            total_checked += 1;
            if !rt_ok {
                total_failed += 1;
            }
            if smaller {
                total_smaller += 1;
            }

            let rt_str = if rt_ok { "EXACT" } else { "MISMATCH!" };
            let flag = if smaller { " <<<" } else { "" };

            println!(
                "{:<20} {:>6} {:>8} {:>8} {:>5.3}x {:>10}{}",
                &name[..name.len().min(20)],
                format!("m{method}"),
                zen_size,
                lib_size,
                ratio,
                rt_str,
                flag
            );
        }
    }

    println!("{:-<70}", "");
    println!(
        "{} images × 4 methods = {} checks",
        images.len(),
        total_checked
    );
    println!(
        "{} pixel-exact, {} MISMATCH",
        total_checked - total_failed,
        total_failed
    );
    println!("{} cases where zenwebp <= libwebp", total_smaller);

    if total_failed > 0 {
        std::process::exit(1);
    }
}
