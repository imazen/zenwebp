use std::time::Instant;
use zenwebp::EncoderConfig;

fn main() {
    // Load test image
    let img = image::open("/home/lilith/work/zenwebp/tests/out/gallery1/1.png")
        .expect("Failed to load image");
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let pixels = rgb.into_raw();
    
    // Warm up
    for _ in 0..5 {
        let _ = EncoderConfig::new()
            .quality(75.0)
            .method(4)
            .encode_rgb(&pixels, width, height)
            .unwrap();
    }
    
    // Time 50 encodes
    let start = Instant::now();
    for _ in 0..50 {
        let _ = EncoderConfig::new()
            .quality(75.0)
            .method(4)
            .encode_rgb(&pixels, width, height)
            .unwrap();
    }
    let elapsed = start.elapsed();
    println!("50 encodes: {:?}", elapsed);
    println!("Avg: {:?}", elapsed / 50);
}
