//! Emit timing output for an independent WebP decoder.
use zencodec::animation::FrameDuration;
use zencodec::encode::{AnimationFrameEncoder as _, EncodeJob as _, EncoderConfig as _};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os().nth(1).expect("output path required");
    let mut encoder = zenwebp::zencodec::WebpEncoderConfig::lossless()
        .job()
        .animation_frame_encoder()?;
    for index in 0..3 {
        let bytes = [index * 40, 20, 60].repeat(256);
        let pixels =
            zenpixels::PixelSlice::new(&bytes, 16, 16, 48, zenpixels::PixelDescriptor::RGB8_SRGB)?;
        encoder.push_frame_timed(pixels, FrameDuration::new(1, 30)?, None)?;
    }
    std::fs::write(path, encoder.finish(None)?.into_vec())?;
    Ok(())
}
