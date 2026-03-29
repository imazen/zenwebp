//! Tests for v2 decoder animation support.
//!
//! Verifies that the v2 DecoderContext produces correct output when
//! decoding animated WebP files, both for lossy and mixed-codec animations.

use std::io::Cursor;
use std::path::PathBuf;

use zenwebp::{
    AnimationConfig, AnimationDecoder, AnimationEncoder, BlendMethod, DecodeConfig, DecoderContext,
    DisposeMethod, EncoderConfig, PixelLayout, WebPDecoder,
};

/// Generate a gradient image with a unique pattern per frame index.
fn frame_rgb(w: u32, h: u32, frame_idx: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    let offset = frame_idx * 37; // unique offset per frame
    for y in 0..h {
        for x in 0..w {
            let r = ((x.wrapping_add(offset) * 255) / w.max(1)) as u8;
            let g = ((y.wrapping_add(offset) * 255) / h.max(1)) as u8;
            let b = (((x + y + offset) * 128) / (w + h).max(1)) as u8;
            rgb.extend_from_slice(&[r, g, b]);
        }
    }
    rgb
}

/// Create a multi-frame lossy animation, decode all frames, and verify
/// that each frame decodes without error and produces non-zero pixel data.
#[test]
fn lossy_animation_decodes_all_frames() {
    let w = 64u32;
    let h = 64u32;
    let num_frames = 5u32;

    let config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(w, h, config).unwrap();

    let enc_config = EncoderConfig::new_lossy().with_quality(75.0).with_method(4);
    for i in 0..num_frames {
        let pixels = frame_rgb(w, h, i);
        anim.add_frame(&pixels, PixelLayout::Rgb8, i * 100, &enc_config)
            .unwrap();
    }

    let webp_data = anim.finalize(num_frames * 100).unwrap();

    // Decode all frames using AnimationDecoder (which now uses v2 internally)
    let mut decoder = AnimationDecoder::new(&webp_data).unwrap();
    let info = decoder.info();
    assert_eq!(info.canvas_width, w);
    assert_eq!(info.canvas_height, h);
    assert_eq!(info.frame_count, num_frames);

    let mut frames_decoded = 0u32;
    while let Some(frame) = decoder.next_frame().unwrap() {
        assert_eq!(frame.width, w);
        assert_eq!(frame.height, h);
        // Verify frame has actual pixel data (not all zeros)
        let non_zero = frame.data.iter().filter(|&&b| b != 0).count();
        assert!(non_zero > 0, "frame {} is all zeros", frames_decoded + 1);
        frames_decoded += 1;
    }
    assert_eq!(frames_decoded, num_frames);
}

/// Decode the test animated lossy file and compare against reference PNGs.
/// This exercises the full v2 pipeline through the WebPDecoder::read_frame path.
#[test]
fn animated_lossy_matches_reference() {
    let path = "tests/images/animated/random_lossy.webp";
    let contents = std::fs::read(path).unwrap();
    let mut decoder = WebPDecoder::new(&contents).unwrap();
    let config = DecodeConfig::default().with_dithering_strength(0);
    decoder.set_lossy_upsampling(config.upsampling);

    assert!(decoder.is_animated());
    let (width, height) = decoder.dimensions();
    let num_frames = decoder.num_frames();
    assert!(num_frames > 0, "expected animated file to have frames");

    // Decode first frame via read_image
    let bpp = if decoder.has_alpha() { 4 } else { 3 };
    let mut data = vec![0u8; width as usize * height as usize * bpp];
    decoder.read_image(&mut data).unwrap();

    // Check first frame against reference
    let ref_path = PathBuf::from(format!("tests/reference/animated/random_lossy-1.png"));
    if ref_path.exists() {
        let ref_contents = std::fs::read(&ref_path).unwrap();
        let mut ref_decoder = png::Decoder::new(Cursor::new(ref_contents))
            .read_info()
            .unwrap();
        let mut ref_data = vec![0; ref_decoder.output_buffer_size().unwrap()];
        ref_decoder.next_frame(&mut ref_data).unwrap();

        let diff_count = data
            .iter()
            .zip(ref_data.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diff_count, 0,
            "frame 1 has {diff_count} byte differences vs reference"
        );
    }

    // Decode remaining frames
    for i in 1..=num_frames {
        let mut frame_data = vec![0u8; width as usize * height as usize * bpp];
        let duration = decoder.read_frame(&mut frame_data).unwrap();
        assert!(
            duration <= 10000,
            "suspicious frame duration: {}ms",
            duration
        );

        let ref_path = PathBuf::from(format!("tests/reference/animated/random_lossy-{i}.png"));
        if ref_path.exists() {
            let ref_contents = std::fs::read(&ref_path).unwrap();
            let mut ref_decoder = png::Decoder::new(Cursor::new(ref_contents))
                .read_info()
                .unwrap();
            let mut ref_data = vec![0; ref_decoder.output_buffer_size().unwrap()];
            ref_decoder.next_frame(&mut ref_data).unwrap();

            let diff_count = frame_data
                .iter()
                .zip(ref_data.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                diff_count, 0,
                "frame {i} has {diff_count} byte differences vs reference"
            );
        }
    }
}

/// Verify AnimationDecoder produces correct output for lossy animation.
#[test]
fn animation_decoder_lossy_correctness() {
    let path = "tests/images/animated/random_lossy.webp";
    let contents = std::fs::read(path).unwrap();

    let config = DecodeConfig::default().with_dithering_strength(0);
    let mut decoder = AnimationDecoder::new_with_config(&contents, &config).unwrap();
    let info = decoder.info();

    let mut frames_decoded = 0u32;
    let mut cumulative_ms = 0u32;

    while let Some(frame) = decoder.next_frame().unwrap() {
        assert_eq!(frame.width, info.canvas_width);
        assert_eq!(frame.height, info.canvas_height);
        assert_eq!(frame.timestamp_ms, cumulative_ms);
        cumulative_ms += frame.duration_ms;

        // Verify against reference PNG
        let i = frames_decoded + 1;
        let ref_path = PathBuf::from(format!("tests/reference/animated/random_lossy-{i}.png"));
        if ref_path.exists() {
            let ref_contents = std::fs::read(&ref_path).unwrap();
            let mut ref_decoder = png::Decoder::new(Cursor::new(ref_contents))
                .read_info()
                .unwrap();
            let mut ref_data = vec![0; ref_decoder.output_buffer_size().unwrap()];
            ref_decoder.next_frame(&mut ref_data).unwrap();

            // The AnimationDecoder may output RGB or RGBA depending on has_alpha.
            // The reference PNGs should match the decoder's output format.
            let diff_count = frame
                .data
                .iter()
                .zip(ref_data.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                diff_count, 0,
                "AnimationDecoder frame {i} has {diff_count} byte differences vs reference"
            );
        }

        frames_decoded += 1;
    }

    assert!(frames_decoded > 0, "no frames decoded");
}

/// Encode a lossy animation with varying frame sizes, then decode with v2.
/// This tests DecoderContext buffer reuse across differently-sized frames.
#[test]
fn v2_reuse_across_different_frame_sizes() {
    let canvas_w = 128u32;
    let canvas_h = 128u32;

    // Create frames at different sizes within the canvas
    let anim_config = AnimationConfig {
        minimize_size: false,
        ..Default::default()
    };
    let mut anim = AnimationEncoder::new(canvas_w, canvas_h, anim_config).unwrap();
    let enc_config = EncoderConfig::new_lossy().with_quality(75.0).with_method(4);

    // Frame 1: full canvas
    let frame1 = frame_rgb(canvas_w, canvas_h, 0);
    anim.add_frame(&frame1, PixelLayout::Rgb8, 0, &enc_config)
        .unwrap();

    // Frame 2: smaller sub-frame at offset
    let sub_w = 64u32;
    let sub_h = 48u32;
    let frame2 = frame_rgb(sub_w, sub_h, 1);
    anim.add_frame_advanced(
        &frame2,
        PixelLayout::Rgb8,
        sub_w,
        sub_h,
        16,
        16,
        100,
        &enc_config,
        DisposeMethod::None,
        BlendMethod::Overwrite,
    )
    .unwrap();

    // Frame 3: different size sub-frame
    let sub2_w = 96u32;
    let sub2_h = 32u32;
    let frame3 = frame_rgb(sub2_w, sub2_h, 2);
    anim.add_frame_advanced(
        &frame3,
        PixelLayout::Rgb8,
        sub2_w,
        sub2_h,
        0,
        0,
        200,
        &enc_config,
        DisposeMethod::None,
        BlendMethod::Overwrite,
    )
    .unwrap();

    let webp_data = anim.finalize(300).unwrap();

    // Decode all frames — exercises v2 DecoderContext buffer reuse
    let mut decoder = AnimationDecoder::new(&webp_data).unwrap();
    let info = decoder.info();
    assert_eq!(info.frame_count, 3);

    let mut frames_decoded = 0;
    while let Some(frame) = decoder.next_frame().unwrap() {
        assert_eq!(frame.width, canvas_w);
        assert_eq!(frame.height, canvas_h);
        frames_decoded += 1;
    }
    assert_eq!(frames_decoded, 3);
}

/// Encode and decode a lossy animation, then reset and decode again.
/// Verifies that frame-by-frame output is identical across reset cycles,
/// confirming proper DecoderContext state cleanup.
#[test]
fn v2_animation_reset_produces_identical_output() {
    let w = 48u32;
    let h = 48u32;

    let config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(w, h, config).unwrap();

    let enc_config = EncoderConfig::new_lossy().with_quality(75.0).with_method(4);
    for i in 0..3u32 {
        let pixels = frame_rgb(w, h, i);
        anim.add_frame(&pixels, PixelLayout::Rgb8, i * 100, &enc_config)
            .unwrap();
    }

    let webp_data = anim.finalize(300).unwrap();

    let mut decoder = AnimationDecoder::new(&webp_data).unwrap();

    // First pass: collect all frame data
    let first_pass: Vec<Vec<u8>> = decoder
        .decode_all()
        .unwrap()
        .into_iter()
        .map(|f| f.data)
        .collect();

    // Reset and decode again
    decoder.reset().unwrap();

    let second_pass: Vec<Vec<u8>> = decoder
        .decode_all()
        .unwrap()
        .into_iter()
        .map(|f| f.data)
        .collect();

    assert_eq!(first_pass.len(), second_pass.len());
    for (i, (a, b)) in first_pass.iter().zip(second_pass.iter()).enumerate() {
        assert_eq!(
            a,
            b,
            "frame {} differs between first and second decode pass",
            i + 1
        );
    }
}

/// Decode the test animated lossless file and compare against reference PNGs.
/// Lossless frames use VP8L, not v2, but this verifies the mixed pipeline works.
#[test]
fn animated_lossless_matches_reference() {
    let path = "tests/images/animated/random_lossless.webp";
    let contents = std::fs::read(path).unwrap();
    let mut decoder = WebPDecoder::new(&contents).unwrap();

    assert!(decoder.is_animated());
    let (width, height) = decoder.dimensions();
    let num_frames = decoder.num_frames();

    let bpp = if decoder.has_alpha() { 4 } else { 3 };
    let mut data = vec![0u8; width as usize * height as usize * bpp];
    decoder.read_image(&mut data).unwrap();

    for i in 1..=num_frames {
        let mut frame_data = vec![0u8; width as usize * height as usize * bpp];
        let _duration = decoder.read_frame(&mut frame_data).unwrap();

        let ref_path = PathBuf::from(format!("tests/reference/animated/random_lossless-{i}.png"));
        if ref_path.exists() {
            let ref_contents = std::fs::read(&ref_path).unwrap();
            let mut ref_decoder = png::Decoder::new(Cursor::new(ref_contents))
                .read_info()
                .unwrap();
            let mut ref_data = vec![0; ref_decoder.output_buffer_size().unwrap()];
            ref_decoder.next_frame(&mut ref_data).unwrap();

            assert_eq!(
                frame_data, ref_data,
                "lossless frame {i} doesn't match reference"
            );
        }
    }
}

/// Encode a lossy animation with the AnimationEncoder, decode with
/// AnimationDecoder twice, verify the frames are identical both times.
/// This confirms v2 DecoderContext reuse produces consistent results.
#[test]
fn lossy_animation_roundtrip_pixel_exact() {
    let w = 32u32;
    let h = 32u32;

    let anim_config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(w, h, anim_config).unwrap();

    let enc_config = EncoderConfig::new_lossy().with_quality(75.0).with_method(4);
    for i in 0..4u32 {
        let pixels = frame_rgb(w, h, i);
        anim.add_frame(&pixels, PixelLayout::Rgb8, i * 50, &enc_config)
            .unwrap();
    }

    let webp_data = anim.finalize(200).unwrap();

    // Decode all frames with AnimationDecoder (uses v2 for VP8 frames)
    let mut decoder1 = AnimationDecoder::new(&webp_data).unwrap();
    let frames1 = decoder1.decode_all().unwrap();
    assert_eq!(frames1.len(), 4);

    // Decode again with a fresh AnimationDecoder
    let mut decoder2 = AnimationDecoder::new(&webp_data).unwrap();
    let frames2 = decoder2.decode_all().unwrap();
    assert_eq!(frames2.len(), 4);

    for (i, (f1, f2)) in frames1.iter().zip(frames2.iter()).enumerate() {
        assert_eq!(
            f1.data.len(),
            f2.data.len(),
            "frame {} buffer size mismatch",
            i + 1
        );
        let diff_count = f1
            .data
            .iter()
            .zip(f2.data.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diff_count,
            0,
            "frame {} has {diff_count} differences between two decode passes",
            i + 1
        );
    }

    // Also verify each frame has actual content
    for (i, frame) in frames1.iter().enumerate() {
        let non_zero = frame.data.iter().filter(|&&b| b != 0).count();
        assert!(non_zero > 0, "frame {} is all zeros", i + 1);
    }
}

/// Test the DecoderContext::decode_animation API directly.
/// Encodes a multi-frame lossy animation and decodes it through the
/// callback-based API, verifying frame metadata and pixel content.
#[test]
fn decode_animation_api_lossy() {
    let w = 64u32;
    let h = 64u32;
    let num_frames = 4u32;

    let config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(w, h, config).unwrap();

    let enc_config = EncoderConfig::new_lossy().with_quality(75.0).with_method(4);
    for i in 0..num_frames {
        let pixels = frame_rgb(w, h, i);
        anim.add_frame(&pixels, PixelLayout::Rgb8, i * 100, &enc_config)
            .unwrap();
    }

    let webp_data = anim.finalize(num_frames * 100).unwrap();

    // Decode via DecoderContext::decode_animation
    let mut ctx = DecoderContext::new();
    let mut frame_count = 0u32;
    let mut timestamps = Vec::new();
    let mut durations = Vec::new();

    ctx.decode_animation(&webp_data, |frame| {
        assert_eq!(frame.width, w);
        assert_eq!(frame.height, h);
        assert_eq!(frame.frame_num, frame_count + 1);

        // Canvas should have content
        let non_zero = frame.pixels.iter().filter(|&&b| b != 0).count();
        assert!(
            non_zero > 0,
            "frame {} canvas is all zeros",
            frame.frame_num
        );

        timestamps.push(frame.timestamp_ms);
        durations.push(frame.duration_ms);
        frame_count += 1;
        true // continue
    })
    .unwrap();

    assert_eq!(frame_count, num_frames);

    // Verify timestamps are monotonically increasing
    for i in 1..timestamps.len() {
        assert!(
            timestamps[i] >= timestamps[i - 1],
            "timestamps not monotonic: {} >= {} failed",
            timestamps[i],
            timestamps[i - 1]
        );
    }
}

/// Test early termination of decode_animation via callback returning false.
#[test]
fn decode_animation_api_early_stop() {
    let w = 32u32;
    let h = 32u32;

    let config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(w, h, config).unwrap();

    let enc_config = EncoderConfig::new_lossy().with_quality(75.0).with_method(0);
    for i in 0..5u32 {
        let pixels = frame_rgb(w, h, i);
        anim.add_frame(&pixels, PixelLayout::Rgb8, i * 50, &enc_config)
            .unwrap();
    }

    let webp_data = anim.finalize(250).unwrap();

    let mut ctx = DecoderContext::new();
    let mut frame_count = 0u32;

    ctx.decode_animation(&webp_data, |_frame| {
        frame_count += 1;
        frame_count < 3 // stop after 3 frames
    })
    .unwrap();

    assert_eq!(frame_count, 3, "expected early stop after 3 frames");
}

/// Compare decode_animation output against AnimationDecoder output
/// to verify both paths produce identical composited frames.
#[test]
fn decode_animation_api_matches_animation_decoder() {
    let path = "tests/images/animated/random_lossy.webp";
    let contents = std::fs::read(path).unwrap();

    // Decode with AnimationDecoder
    let config = DecodeConfig::default().with_dithering_strength(0);
    let mut anim_decoder = AnimationDecoder::new_with_config(&contents, &config).unwrap();
    let anim_frames = anim_decoder.decode_all().unwrap();

    // Decode with DecoderContext::decode_animation
    let mut ctx = DecoderContext::new();
    let mut api_frames: Vec<Vec<u8>> = Vec::new();

    ctx.decode_animation(&contents, |frame| {
        api_frames.push(frame.pixels.to_vec());
        true
    })
    .unwrap();

    assert_eq!(anim_frames.len(), api_frames.len(), "frame count mismatch");

    for (i, (anim, api)) in anim_frames.iter().zip(api_frames.iter()).enumerate() {
        // AnimationDecoder may return RGB (if no alpha) while decode_animation
        // always returns RGBA canvas. Handle both cases.
        if anim.data.len() == api.len() {
            let diff_count = anim
                .data
                .iter()
                .zip(api.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                diff_count,
                0,
                "frame {} has {diff_count} pixel differences",
                i + 1
            );
        } else {
            // AnimationDecoder returned RGB, decode_animation returned RGBA
            // Compare RGB channels only
            let rgb_len = (anim.width as usize) * (anim.height as usize) * 3;
            assert_eq!(
                anim.data.len(),
                rgb_len,
                "unexpected AnimationDecoder format"
            );

            for pixel_idx in 0..(anim.width as usize * anim.height as usize) {
                let rgb_base = pixel_idx * 3;
                let rgba_base = pixel_idx * 4;
                for c in 0..3 {
                    assert_eq!(
                        anim.data[rgb_base + c],
                        api[rgba_base + c],
                        "frame {} pixel {} channel {} differs",
                        i + 1,
                        pixel_idx,
                        c,
                    );
                }
            }
        }
    }
}
