//! Try decoding our VP8L output with zenwebp's decoder.

fn main() {
    // Try the new_vp8l image first, then gradient, then simple test
    let webp_data = if std::path::Path::new("/tmp/zenwebp_new_vp8l.webp").exists() {
        std::fs::read("/tmp/zenwebp_new_vp8l.webp").unwrap()
    } else if std::path::Path::new("/tmp/test_gradient_vp8l.webp").exists() {
        std::fs::read("/tmp/test_gradient_vp8l.webp").unwrap()
    } else {
        std::fs::read("/tmp/test_debug_vp8l.webp").unwrap()
    };
    println!("File size: {} bytes", webp_data.len());

    match zenwebp::decode_rgba(&webp_data) {
        Ok((data, width, height)) => {
            println!("Decoded successfully!");
            println!("  Width: {}", width);
            println!("  Height: {}", height);
            println!("  First pixel RGBA: {:?}", &data[0..4.min(data.len())]);
        }
        Err(e) => {
            println!("Decode FAILED: {:?}", e);
        }
    }
}
