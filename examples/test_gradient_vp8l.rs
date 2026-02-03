//! Test VP8L encoder with a gradient image.

use std::process::Command;

fn main() {
    // Create a 16x16 gradient image
    let width = 16u32;
    let height = 16u32;
    let mut rgb_pixels = Vec::with_capacity((width * height * 3) as usize);

    for y in 0..height {
        for x in 0..width {
            let r = (x * 16) as u8;
            let g = (y * 16) as u8;
            let b = 128;
            rgb_pixels.push(r);
            rgb_pixels.push(g);
            rgb_pixels.push(b);
        }
    }

    println!("Testing {}x{} gradient image", width, height);

    // Encode with new VP8L encoder (no transforms for simplicity)
    let config = zenwebp::encoder::vp8l::Vp8lConfig {
        use_predictor: false,
        use_subtract_green: false,
        use_palette: false,
        ..Default::default()
    };

    let vp8l_data = zenwebp::encoder::vp8l::encode_vp8l(
        &rgb_pixels, width, height, false, &config
    ).unwrap();

    println!("VP8L data ({} bytes)", vp8l_data.len());

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

    std::fs::write("/tmp/test_gradient_vp8l.webp", &webp).unwrap();

    print!("Verifying... ");
    let verify = Command::new("/home/lilith/work/libwebp/examples/dwebp")
        .args(["/tmp/test_gradient_vp8l.webp", "-o", "/tmp/test_gradient.ppm"])
        .output();
    match verify {
        Ok(out) if out.status.success() => {
            println!("OK");
            // Verify pixel values
            let ppm = std::fs::read_to_string("/tmp/test_gradient.ppm").unwrap_or_default();
            println!("First 200 chars of decoded PPM:\n{}", &ppm[..200.min(ppm.len())]);
        }
        Ok(out) => println!("FAILED: {}", String::from_utf8_lossy(&out.stderr)),
        Err(e) => println!("Error: {}", e),
    }
}
