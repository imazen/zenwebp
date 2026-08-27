//! `WebPDecoder::reset_animation` / `AnimationDecoder::reset` must restart
//! compositing from a clean canvas.
//!
//! Regression for #78-B: `reset_animation` rewound the frame cursor and set
//! `dispose_next_frame = true` (the fresh-decoder state) but kept the
//! composited canvas and the previous-frame rectangle from the LAST frame.
//! On the next frame 1 the dispose step therefore cleared only that stale
//! rectangle and painted frame 1 over whatever else the last pass left on
//! the canvas — so `decode_all()` twice (it resets first) returned a
//! different first frame the second time, and `reset()` mid-stream
//! corrupted every sub-canvas frame after it.

use zenwebp::mux::{
    AnimationConfig, AnimationDecoder, AnimationEncoder, BlendMethod, DisposeMethod,
};
use zenwebp::{EncoderConfig, PixelLayout};

/// 8x8 transparent canvas; three opaque 4x4 sub-frames at (0,0), (4,0),
/// (0,4), no disposal, overwrite blending. After frame 3 the canvas holds
/// all three squares; a correct restart of frame 1 shows ONLY the first.
fn three_subframe_animation() -> Vec<u8> {
    let mut anim = AnimationEncoder::new(
        8,
        8,
        AnimationConfig {
            background_color: [0, 0, 0, 0],
            minimize_size: false,
            ..Default::default()
        },
    )
    .unwrap();
    let cfg = EncoderConfig::new_lossless();
    for (i, (x, y, color)) in [
        (0, 0, [255, 0, 0, 255]),
        (4, 0, [0, 255, 0, 255]),
        (0, 4, [0, 0, 255, 255]),
    ]
    .into_iter()
    .enumerate()
    {
        let rgba: Vec<u8> = color.iter().cycle().take(4 * 4 * 4).copied().collect();
        anim.add_frame_advanced(
            &rgba,
            PixelLayout::Rgba8,
            4,
            4,
            x,
            y,
            i as u32 * 100,
            &cfg,
            DisposeMethod::None,
            BlendMethod::Overwrite,
        )
        .unwrap();
    }
    anim.finalize(100).unwrap()
}

#[test]
fn decode_all_twice_returns_identical_frames() {
    let webp = three_subframe_animation();
    let mut dec = AnimationDecoder::new(&webp).unwrap();
    let pass1 = dec.decode_all().unwrap();
    let pass2 = dec.decode_all().unwrap();
    assert_eq!(pass1.len(), 3);
    assert_eq!(pass2.len(), 3);
    for (i, (a, b)) in pass1.iter().zip(&pass2).enumerate() {
        assert!(
            a.data == b.data,
            "frame {} differs between the first and second decode_all() pass",
            i + 1
        );
    }
    // And the first frame really is "only the first square": the pixel at
    // (5, 1) (inside frame 2's rectangle) must be transparent on both passes.
    let px = |f: &zenwebp::mux::AnimFrame, x: usize, y: usize| {
        let i = (y * 8 + x) * 4;
        [f.data[i], f.data[i + 1], f.data[i + 2], f.data[i + 3]]
    };
    assert_eq!(px(&pass2[0], 1, 1), [255, 0, 0, 255]);
    assert_eq!(
        px(&pass2[0], 5, 1),
        [0, 0, 0, 0],
        "stale frame-2 pixels survived the reset"
    );
    assert_eq!(
        px(&pass2[0], 1, 5),
        [0, 0, 0, 0],
        "stale frame-3 pixels survived the reset"
    );
}

#[test]
fn reset_mid_stream_restarts_from_a_clean_canvas() {
    let webp = three_subframe_animation();
    let mut dec = AnimationDecoder::new(&webp).unwrap();
    let first = dec.next_frame().unwrap().unwrap();
    let _second = dec.next_frame().unwrap().unwrap();
    dec.reset().unwrap();
    let first_again = dec.next_frame().unwrap().unwrap();
    assert!(
        first.data == first_again.data,
        "frame 1 after a mid-stream reset must equal the original frame 1"
    );
}
