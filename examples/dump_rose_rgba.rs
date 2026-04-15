use std::fs;
fn main() {
    let bytes = fs::read("/home/lilith/work/imageflow-zen-v3/.image-cache/sources/imageflow-resources/test_inputs/1_webp_ll.webp").unwrap();
    let (pixels, w, h) = zenwebp::oneshot::decode_rgba(&bytes).unwrap();
    fs::write("/tmp/zenwebp_oneshot_decoded.rgba", &pixels).unwrap();
    println!("Decoded {w}x{h}, {} bytes", pixels.len());
}
