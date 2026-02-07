// Debug a specific I4 block to see all mode RD scores

use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout, Preset};

fn main() {
    let path = "/tmp/CID22/original/792079.png";
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = buf[..info.buffer_size()].to_vec();
    let (w, h) = (info.width, info.height);

    // Get debug target from command line or use default
    let args: Vec<String> = std::env::args().collect();
    let (mbx, mby, block) = if args.len() >= 4 {
        (
            args[1].parse().unwrap_or(21),
            args[2].parse().unwrap_or(1),
            args[3].parse().unwrap_or(4),
        )
    } else {
        (21, 1, 4) // Block where we chose DC but libwebp chose VE
    };

    println!("=== Debug Block MB({},{}) block {} ===\n", mbx, mby, block);
    println!("Mode order: DC=0, TM=1, VE=2, HE=3, LD=4, RD=5, VR=6, VL=7, HD=8, HU=9");
    println!();

    // Set debug env var
    std::env::set_var("BLOCK_DEBUG", format!("{},{},{}", mbx, mby, block));

    // Encode - this will print debug output for the specified block
    let _cfg = EncoderConfig::with_preset(Preset::Default, 75.0)
        .method(4)
        .sns_strength(0)
        .filter_strength(0)
        .segments(1);
    let _zen = EncodeRequest::new(&_cfg, &rgb, PixelLayout::Rgb8, w, h)
        .encode()
        .unwrap();

    println!("\nDone.");
}
