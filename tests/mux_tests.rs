//! Tests for the mux/demux/animation APIs.

use zenwebp::mux::{
    AnimationConfig, AnimationEncoder, BlendMethod, DisposeMethod, MuxFrame, WebPDemuxer, WebPMux,
};
use zenwebp::{ColorType, EncoderConfig, LoopCount, WebPDecoder};

/// Create a solid-color RGBA frame.
fn solid_rgba(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
    let pixel = [r, g, b, a];
    pixel
        .iter()
        .cycle()
        .take((width * height * 4) as usize)
        .copied()
        .collect()
}

/// Create a solid-color RGB frame.
fn solid_rgb(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
    let pixel = [r, g, b];
    pixel
        .iter()
        .cycle()
        .take((width * height * 3) as usize)
        .copied()
        .collect()
}

// ============================================================================
// Demux tests
// ============================================================================

#[test]
fn demux_lossy_animated() {
    let data = std::fs::read("tests/images/animated/random_lossy.webp").unwrap();
    let demuxer = WebPDemuxer::new(&data).unwrap();

    assert!(demuxer.is_animated());
    assert!(demuxer.num_frames() > 1);
    assert!(demuxer.canvas_width() > 0);
    assert!(demuxer.canvas_height() > 0);

    // Verify we can iterate all frames
    let frames: Vec<_> = demuxer.frames().collect();
    assert_eq!(frames.len(), demuxer.num_frames() as usize);

    // Verify frame metadata is reasonable
    for frame in &frames {
        assert!(frame.width > 0);
        assert!(frame.height > 0);
        assert!(!frame.bitstream.is_empty());
        assert!(frame.is_lossy);
    }

    // Verify 1-based indexing
    assert!(demuxer.frame(0).is_none());
    assert!(demuxer.frame(1).is_some());
    assert!(demuxer.frame(demuxer.num_frames()).is_some());
    assert!(demuxer.frame(demuxer.num_frames() + 1).is_none());
}

#[test]
fn demux_lossless_animated() {
    let data = std::fs::read("tests/images/animated/random_lossless.webp").unwrap();
    let demuxer = WebPDemuxer::new(&data).unwrap();

    assert!(demuxer.is_animated());
    assert!(demuxer.num_frames() > 1);

    let frames: Vec<_> = demuxer.frames().collect();
    assert_eq!(frames.len(), demuxer.num_frames() as usize);

    for frame in &frames {
        assert!(frame.width > 0);
        assert!(frame.height > 0);
        assert!(!frame.bitstream.is_empty());
        assert!(!frame.is_lossy);
    }
}

#[test]
fn demux_matches_decoder_metadata() {
    let data = std::fs::read("tests/images/animated/random_lossy.webp").unwrap();

    let demuxer = WebPDemuxer::new(&data).unwrap();
    let decoder = WebPDecoder::new(&data).unwrap();

    assert_eq!(demuxer.canvas_width(), decoder.dimensions().0);
    assert_eq!(demuxer.canvas_height(), decoder.dimensions().1);
    assert_eq!(demuxer.num_frames(), decoder.num_frames());
    assert_eq!(demuxer.loop_count(), decoder.loop_count());
}

#[test]
fn demux_simple_lossy() {
    // Encode a simple lossy image
    let pixels = solid_rgb(64, 64, 128, 64, 192);
    let config = EncoderConfig::new().quality(75.0);
    let webp = config.encode_rgb(&pixels, 64, 64).unwrap();

    let demuxer = WebPDemuxer::new(&webp).unwrap();
    assert!(!demuxer.is_animated());
    assert_eq!(demuxer.num_frames(), 1);
    assert_eq!(demuxer.canvas_width(), 64);
    assert_eq!(demuxer.canvas_height(), 64);

    let frame = demuxer.frame(1).unwrap();
    assert_eq!(frame.frame_num, 1);
    assert_eq!(frame.x_offset, 0);
    assert_eq!(frame.y_offset, 0);
    assert!(frame.is_lossy);
    assert!(!frame.bitstream.is_empty());
}

#[test]
fn demux_simple_lossless() {
    // Encode a simple lossless image
    let pixels = solid_rgba(32, 32, 128, 64, 192, 255);
    let config = EncoderConfig::new().quality(100.0).lossless(true);
    let webp = config.encode_rgba(&pixels, 32, 32).unwrap();

    let demuxer = WebPDemuxer::new(&webp).unwrap();
    assert!(!demuxer.is_animated());
    assert_eq!(demuxer.num_frames(), 1);
    assert_eq!(demuxer.canvas_width(), 32);
    assert_eq!(demuxer.canvas_height(), 32);

    let frame = demuxer.frame(1).unwrap();
    assert!(!frame.is_lossy);
    assert!(!frame.bitstream.is_empty());
}

// ============================================================================
// Mux tests
// ============================================================================

#[test]
fn mux_single_image_simple() {
    // Encode a frame, then wrap it in a mux container
    let pixels = solid_rgb(64, 64, 200, 100, 50);
    let config = EncoderConfig::new().quality(75.0);
    let webp = config.encode_rgb(&pixels, 64, 64).unwrap();

    // Demux to get raw bitstream
    let demuxer = WebPDemuxer::new(&webp).unwrap();
    let frame = demuxer.frame(1).unwrap();

    // Mux it back
    let mut mux = WebPMux::new(64, 64);
    mux.set_image(MuxFrame {
        x_offset: 0,
        y_offset: 0,
        width: 64,
        height: 64,
        duration_ms: 0,
        dispose: DisposeMethod::None,
        blend: BlendMethod::Overwrite,
        bitstream: frame.bitstream.to_vec(),
        alpha_data: None,
        is_lossless: false,
    });
    let assembled = mux.assemble().unwrap();

    // Verify it decodes
    let decoder = WebPDecoder::new(&assembled).unwrap();
    assert_eq!(decoder.dimensions(), (64, 64));
}

#[test]
fn mux_single_image_with_metadata() {
    let pixels = solid_rgb(32, 32, 100, 100, 100);
    let config = EncoderConfig::new().quality(75.0);
    let webp = config.encode_rgb(&pixels, 32, 32).unwrap();

    let demuxer = WebPDemuxer::new(&webp).unwrap();
    let frame = demuxer.frame(1).unwrap();

    let icc_data = vec![1, 2, 3, 4, 5]; // fake ICC

    let mut mux = WebPMux::new(32, 32);
    mux.set_icc_profile(icc_data.clone());
    mux.set_image(MuxFrame {
        x_offset: 0,
        y_offset: 0,
        width: 32,
        height: 32,
        duration_ms: 0,
        dispose: DisposeMethod::None,
        blend: BlendMethod::Overwrite,
        bitstream: frame.bitstream.to_vec(),
        alpha_data: None,
        is_lossless: false,
    });
    let assembled = mux.assemble().unwrap();

    // Verify ICC round-trips
    let demuxer2 = WebPDemuxer::new(&assembled).unwrap();
    assert_eq!(demuxer2.icc_profile().unwrap(), &icc_data[..]);

    // Verify it still decodes
    let decoder = WebPDecoder::new(&assembled).unwrap();
    assert_eq!(decoder.dimensions(), (32, 32));
}

#[test]
fn mux_from_data_roundtrip() {
    let data = std::fs::read("tests/images/animated/random_lossy.webp").unwrap();

    // Parse into mux
    let mux = WebPMux::from_data(&data).unwrap();

    // Reassemble
    let assembled = mux.assemble().unwrap();

    // Verify it decodes with same metadata
    let decoder = WebPDecoder::new(&assembled).unwrap();
    let original_decoder = WebPDecoder::new(&data).unwrap();
    assert_eq!(decoder.dimensions(), original_decoder.dimensions());
    assert_eq!(decoder.num_frames(), original_decoder.num_frames());
}

// ============================================================================
// Animation encoder tests
// ============================================================================

#[test]
fn animation_encode_decode_lossy_roundtrip() {
    let config = AnimationConfig {
        background_color: [0, 0, 0, 0],
        loop_count: LoopCount::Forever,
    };
    let mut anim = AnimationEncoder::new(64, 64, config).unwrap();

    let frame_config = EncoderConfig::new().quality(75.0).method(0);

    // 3 differently-colored frames
    let frame1 = solid_rgb(64, 64, 255, 0, 0);
    let frame2 = solid_rgb(64, 64, 0, 255, 0);
    let frame3 = solid_rgb(64, 64, 0, 0, 255);

    anim.add_frame(&frame1, ColorType::Rgb8, 0, &frame_config)
        .unwrap();
    anim.add_frame(&frame2, ColorType::Rgb8, 100, &frame_config)
        .unwrap();
    anim.add_frame(&frame3, ColorType::Rgb8, 200, &frame_config)
        .unwrap();

    let webp = anim.finalize(100).unwrap();

    // Decode and verify
    let mut decoder = WebPDecoder::new(&webp).unwrap();
    assert_eq!(decoder.dimensions(), (64, 64));
    assert!(decoder.is_animated());
    assert_eq!(decoder.num_frames(), 3);
    assert_eq!(decoder.loop_count(), LoopCount::Forever);

    // Read all frames
    let buf_size = decoder.output_buffer_size().unwrap();
    let mut buf = vec![0u8; buf_size];
    for _ in 0..3 {
        decoder.read_frame(&mut buf).unwrap();
    }
}

#[test]
fn animation_encode_decode_lossless_roundtrip() {
    let config = AnimationConfig {
        background_color: [0, 0, 0, 255],
        loop_count: LoopCount::Times(std::num::NonZeroU16::new(3).unwrap()),
    };
    let mut anim = AnimationEncoder::new(32, 32, config).unwrap();

    let frame_config = EncoderConfig::new().quality(100.0).lossless(true);

    let frame1 = solid_rgba(32, 32, 255, 0, 0, 255);
    let frame2 = solid_rgba(32, 32, 0, 255, 0, 255);

    anim.add_frame(&frame1, ColorType::Rgba8, 0, &frame_config)
        .unwrap();
    anim.add_frame(&frame2, ColorType::Rgba8, 200, &frame_config)
        .unwrap();

    let webp = anim.finalize(200).unwrap();

    let mut decoder = WebPDecoder::new(&webp).unwrap();
    assert_eq!(decoder.dimensions(), (32, 32));
    assert!(decoder.is_animated());
    assert_eq!(decoder.num_frames(), 2);
    assert_eq!(
        decoder.loop_count(),
        LoopCount::Times(std::num::NonZeroU16::new(3).unwrap())
    );

    let buf_size = decoder.output_buffer_size().unwrap();
    let mut buf = vec![0u8; buf_size];

    let d1 = decoder.read_frame(&mut buf).unwrap();
    assert_eq!(d1, 200); // first frame duration = timestamp[1] - timestamp[0]

    let d2 = decoder.read_frame(&mut buf).unwrap();
    assert_eq!(d2, 200); // last frame duration from finalize
}

#[test]
fn animation_frame_durations_from_timestamps() {
    let config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(16, 16, config).unwrap();

    let frame_config = EncoderConfig::new().quality(50.0).method(0);
    let pixels = solid_rgb(16, 16, 128, 128, 128);

    // timestamps: 0, 50, 200
    // durations should be: 50, 150, last_frame_duration
    anim.add_frame(&pixels, ColorType::Rgb8, 0, &frame_config)
        .unwrap();
    anim.add_frame(&pixels, ColorType::Rgb8, 50, &frame_config)
        .unwrap();
    anim.add_frame(&pixels, ColorType::Rgb8, 200, &frame_config)
        .unwrap();

    let webp = anim.finalize(300).unwrap();

    let mut decoder = WebPDecoder::new(&webp).unwrap();
    assert_eq!(decoder.num_frames(), 3);

    let buf_size = decoder.output_buffer_size().unwrap();
    let mut buf = vec![0u8; buf_size];

    let d1 = decoder.read_frame(&mut buf).unwrap();
    assert_eq!(d1, 50);

    let d2 = decoder.read_frame(&mut buf).unwrap();
    assert_eq!(d2, 150);

    let d3 = decoder.read_frame(&mut buf).unwrap();
    assert_eq!(d3, 300);
}

#[test]
fn animation_with_metadata() {
    let config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(16, 16, config).unwrap();

    let icc_data = vec![10, 20, 30, 40];
    let exif_data = vec![50, 60, 70];
    anim.set_icc_profile(icc_data.clone());
    anim.set_exif(exif_data.clone());

    let frame_config = EncoderConfig::new().quality(50.0).method(0);
    let pixels = solid_rgb(16, 16, 100, 100, 100);
    anim.add_frame(&pixels, ColorType::Rgb8, 0, &frame_config)
        .unwrap();

    let webp = anim.finalize(100).unwrap();

    let demuxer = WebPDemuxer::new(&webp).unwrap();
    assert_eq!(demuxer.icc_profile().unwrap(), &icc_data[..]);
    assert_eq!(demuxer.exif().unwrap(), &exif_data[..]);
    assert!(demuxer.xmp().is_none());
}

// ============================================================================
// Demux -> Mux roundtrip
// ============================================================================

#[test]
fn demux_mux_roundtrip_lossy() {
    let data = std::fs::read("tests/images/animated/random_lossy.webp").unwrap();

    // Demux
    let demuxer = WebPDemuxer::new(&data).unwrap();
    let original_frames: Vec<_> = demuxer.frames().collect();

    // Mux
    let mut mux = WebPMux::new(demuxer.canvas_width(), demuxer.canvas_height());
    mux.set_animation(demuxer.background_color(), demuxer.loop_count());

    for frame in &original_frames {
        mux.push_frame(MuxFrame {
            x_offset: frame.x_offset,
            y_offset: frame.y_offset,
            width: frame.width,
            height: frame.height,
            duration_ms: frame.duration_ms,
            dispose: frame.dispose,
            blend: frame.blend,
            bitstream: frame.bitstream.to_vec(),
            alpha_data: frame.alpha_data.map(|d| d.to_vec()),
            is_lossless: !frame.is_lossy,
        })
        .unwrap();
    }

    let assembled = mux.assemble().unwrap();

    // Verify roundtrip
    let demuxer2 = WebPDemuxer::new(&assembled).unwrap();
    assert_eq!(demuxer2.canvas_width(), demuxer.canvas_width());
    assert_eq!(demuxer2.canvas_height(), demuxer.canvas_height());
    assert_eq!(demuxer2.num_frames(), demuxer.num_frames());
    assert_eq!(demuxer2.loop_count(), demuxer.loop_count());

    // Verify each frame's metadata matches
    for (orig, new) in original_frames.iter().zip(demuxer2.frames()) {
        assert_eq!(orig.x_offset, new.x_offset);
        assert_eq!(orig.y_offset, new.y_offset);
        assert_eq!(orig.width, new.width);
        assert_eq!(orig.height, new.height);
        assert_eq!(orig.duration_ms, new.duration_ms);
        assert_eq!(orig.dispose, new.dispose);
        assert_eq!(orig.blend, new.blend);
        assert_eq!(orig.bitstream.len(), new.bitstream.len());
    }

    // Verify it decodes
    let mut decoder = WebPDecoder::new(&assembled).unwrap();
    let buf_size = decoder.output_buffer_size().unwrap();
    let mut buf = vec![0u8; buf_size];
    for _ in 0..demuxer2.num_frames() {
        decoder.read_frame(&mut buf).unwrap();
    }
}

#[test]
fn demux_mux_roundtrip_lossless() {
    let data = std::fs::read("tests/images/animated/random_lossless.webp").unwrap();

    let demuxer = WebPDemuxer::new(&data).unwrap();
    let original_frames: Vec<_> = demuxer.frames().collect();

    let mut mux = WebPMux::new(demuxer.canvas_width(), demuxer.canvas_height());
    mux.set_animation(demuxer.background_color(), demuxer.loop_count());

    for frame in &original_frames {
        mux.push_frame(MuxFrame {
            x_offset: frame.x_offset,
            y_offset: frame.y_offset,
            width: frame.width,
            height: frame.height,
            duration_ms: frame.duration_ms,
            dispose: frame.dispose,
            blend: frame.blend,
            bitstream: frame.bitstream.to_vec(),
            alpha_data: frame.alpha_data.map(|d| d.to_vec()),
            is_lossless: !frame.is_lossy,
        })
        .unwrap();
    }

    let assembled = mux.assemble().unwrap();

    let demuxer2 = WebPDemuxer::new(&assembled).unwrap();
    assert_eq!(demuxer2.num_frames(), demuxer.num_frames());

    // Verify it decodes
    let mut decoder = WebPDecoder::new(&assembled).unwrap();
    let buf_size = decoder.output_buffer_size().unwrap();
    let mut buf = vec![0u8; buf_size];
    for _ in 0..demuxer2.num_frames() {
        decoder.read_frame(&mut buf).unwrap();
    }
}

// ============================================================================
// Error handling tests
// ============================================================================

#[test]
fn mux_no_frames_error() {
    let mux = WebPMux::new(100, 100);
    assert!(mux.assemble().is_err());
}

#[test]
fn mux_animated_no_frames_error() {
    let mut mux = WebPMux::new(100, 100);
    mux.set_animation([0; 4], LoopCount::Forever);
    assert!(mux.assemble().is_err());
}

#[test]
fn mux_frame_outside_canvas_error() {
    let mut mux = WebPMux::new(100, 100);
    mux.set_animation([0; 4], LoopCount::Forever);

    let result = mux.push_frame(MuxFrame {
        x_offset: 90,
        y_offset: 0,
        width: 20, // 90 + 20 = 110 > 100
        height: 10,
        duration_ms: 100,
        dispose: DisposeMethod::None,
        blend: BlendMethod::Overwrite,
        bitstream: vec![0],
        alpha_data: None,
        is_lossless: true,
    });
    assert!(result.is_err());
}

#[test]
fn mux_odd_offset_error() {
    let mut mux = WebPMux::new(100, 100);
    mux.set_animation([0; 4], LoopCount::Forever);

    let result = mux.push_frame(MuxFrame {
        x_offset: 3, // odd!
        y_offset: 0,
        width: 10,
        height: 10,
        duration_ms: 100,
        dispose: DisposeMethod::None,
        blend: BlendMethod::Overwrite,
        bitstream: vec![0],
        alpha_data: None,
        is_lossless: true,
    });
    assert!(result.is_err());
}

#[test]
fn animation_encoder_invalid_dimensions() {
    let config = AnimationConfig::default();
    assert!(AnimationEncoder::new(0, 100, config.clone()).is_err());
    assert!(AnimationEncoder::new(100, 0, config.clone()).is_err());
    assert!(AnimationEncoder::new(20000, 100, config).is_err());
}

#[test]
fn demux_invalid_data() {
    assert!(WebPDemuxer::new(&[]).is_err());
    assert!(WebPDemuxer::new(&[0; 12]).is_err());
    assert!(WebPDemuxer::new(b"not a webp file at all!!").is_err());
}
