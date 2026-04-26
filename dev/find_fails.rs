use std::fs;
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout, Preset};
fn main() {
    let dir = std::path::Path::new("/tmp/imagemagick-sample");
    let mut entries: Vec<_> = fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    let mut fail_count = 0;
    for path in entries {
        let file = match fs::File::open(&path) { Ok(f) => f, Err(_) => continue };
        let decoder = png::Decoder::new(std::io::BufReader::new(file));
        let mut reader = match decoder.read_info() { Ok(r) => r, Err(_) => continue };
        let bs = match reader.output_buffer_size() { Some(s) => s, None => continue };
        let mut buf = vec![0u8; bs];
        let info = match reader.next_frame(&mut buf) { Ok(i) => i, Err(_) => continue };
        let (rgb, w, h) = match info.color_type {
            png::ColorType::Rgb => (buf[..info.buffer_size()].to_vec(), info.width, info.height),
            png::ColorType::Rgba => {
                let rgba = &buf[..info.buffer_size()];
                let mut r = Vec::with_capacity(rgba.len()*3/4);
                for c in rgba.chunks(4) { r.extend_from_slice(&c[..3]); }
                (r, info.width, info.height)
            }
            png::ColorType::Grayscale => {
                let mut r = Vec::with_capacity(buf.len()*3);
                for &g in &buf[..info.buffer_size()] { r.extend_from_slice(&[g,g,g]); }
                (r, info.width, info.height)
            }
            png::ColorType::GrayscaleAlpha => {
                let mut r = Vec::with_capacity(buf.len()*3/2);
                for c in buf[..info.buffer_size()].chunks(2) { r.extend_from_slice(&[c[0],c[0],c[0]]); }
                (r, info.width, info.height)
            }
            _ => continue,
        };
        let cfg = EncoderConfig::with_preset(Preset::Default, 75.0).with_method(4);
        let res = EncodeRequest::new(&cfg, &rgb, PixelLayout::Rgb8, w, h).encode();
        if res.is_err() {
            fail_count += 1;
            if fail_count <= 5 {
                eprintln!("FAIL: {} ({}x{}) ct={:?} err={:?}", path.display(), w, h, info.color_type, res.unwrap_err());
            }
        }
    }
    eprintln!("Total zen fails: {}", fail_count);
}
