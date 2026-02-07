// Test to compare with/without trellis at method 4
use zenwebp::{PixelLayout, EncodeRequest};

fn main() {
    let path = "/tmp/CID22/original/792079.png";
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = buf[..info.buffer_size()].to_vec();
    let (w, h) = (info.width, info.height);

    println!("Comparing method 2 (no trellis) vs method 4 (trellis):");
    println!("Settings: SNS=0, filter=0, segments=1, Q75\n");

    for method in [2, 3, 4, 5, 6] {
        let config = zenwebp::EncoderConfig::new()
            .quality(75.0)
            .method(method)
            .sns_strength(0)
            .filter_strength(0)
            .segments(1);
        let output = EncodeRequest::new(&config, &rgb, PixelLayout::Rgb8, w, h)
            .encode()
            .unwrap();
        println!("Method {}: {} bytes", method, output.len());
    }

    println!("\nComparing to libwebp:");
    for method in [2, 3, 4, 5, 6] {
        let config = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
            .method(method)
            .sns_strength(0)
            .filter_strength(0)
            .filter_sharpness(0)
            .segments(1);
        let output = config.encode_rgb(&rgb, w, h, webpx::Unstoppable).unwrap();
        println!("libwebp m{}: {} bytes", method, output.len());
    }
}
