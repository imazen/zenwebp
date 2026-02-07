//! Test VP8L encoder with a two-color image (no LZ77 needed).
//! This should use simple 2-symbol trees.

use std::process::Command;

fn main() {
    // Create a 4x4 image with alternating red and blue pixels
    // Red: RGB(255, 0, 0), Blue: RGB(0, 0, 255)
    let width = 4u32;
    let height = 4u32;
    let mut rgb_pixels = Vec::with_capacity((width * height * 3) as usize);
    for i in 0..(width * height) {
        if i % 2 == 0 {
            rgb_pixels.extend_from_slice(&[255, 0, 0]); // Red
        } else {
            rgb_pixels.extend_from_slice(&[0, 0, 255]); // Blue
        }
    }

    println!("Testing {}x{} alternating red/blue image", width, height);

    // Encode with new VP8L encoder (no transforms)
    let mut config = zenwebp::encoder::vp8l::Vp8lConfig {
        use_predictor: false,
        use_subtract_green: false,
        use_palette: false,
        ..Default::default()
    };
    config.quality.quality = 0; // Force simple refs (no LZ77)

    let vp8l_data = zenwebp::encoder::vp8l::encode_vp8l(
        &rgb_pixels,
        width,
        height,
        false,
        &config,
        &enough::Unstoppable,
    )
    .unwrap();

    println!(
        "VP8L data ({} bytes): {:02x?}",
        vp8l_data.len(),
        &vp8l_data[..vp8l_data.len().min(32)]
    );

    // Wrap in RIFF container
    let mut webp = Vec::new();
    webp.extend_from_slice(b"RIFF");
    let riff_size = 4 + 8 + vp8l_data.len() + (vp8l_data.len() % 2);
    webp.extend_from_slice(&(riff_size as u32).to_le_bytes());
    webp.extend_from_slice(b"WEBP");
    webp.extend_from_slice(b"VP8L");
    webp.extend_from_slice(&(vp8l_data.len() as u32).to_le_bytes());
    webp.extend_from_slice(&vp8l_data);
    if !vp8l_data.len().is_multiple_of(2) {
        webp.push(0);
    }

    std::fs::write("/tmp/test_two_color.webp", &webp).unwrap();

    print!("Verifying... ");
    let verify = Command::new("/home/lilith/work/libwebp/examples/dwebp")
        .args(["/tmp/test_two_color.webp", "-o", "/tmp/test_two_color.ppm"])
        .output();
    match verify {
        Ok(out) if out.status.success() => println!("OK"),
        Ok(out) => println!("FAILED: {}", String::from_utf8_lossy(&out.stderr)),
        Err(e) => println!("Error: {}", e),
    }
}
