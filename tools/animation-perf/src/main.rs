use std::{path::PathBuf, sync::Arc, time::Duration};
use zenbench::prelude::*;
use zenpixels::{PixelBuffer, PixelDescriptor};
mod main_arm {
    use zenwebp_main as codec;
    include!("arm.rs");
}
mod pr_arm {
    use zenwebp_pr as codec;
    include!("arm.rs");
}
fn settings(group: &mut BenchGroup) {
    group
        .config()
        .warmup_time(Duration::from_millis(200))
        .min_rounds(20)
        .max_rounds(80)
        .max_time(Duration::from_secs(3));
}
fn bench(suite: &mut Suite) {
    let root = PathBuf::from(
        std::env::var_os("ZENWEBP_PERF_FIXTURES")
            .expect("set ZENWEBP_PERF_FIXTURES to the common repository tests directory"),
    );
    let png = std::fs::read(root.join("reference/gallery1/1.png")).unwrap();
    let mut reader = png::Decoder::new(std::io::Cursor::new(png))
        .read_info()
        .unwrap();
    let mut photo = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut photo).unwrap();
    assert_eq!(info.color_type, png::ColorType::Rgb);
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    let frames: Arc<Vec<_>> = Arc::new(
        (0..3usize)
            .map(|i| {
                let mut data = Vec::with_capacity(256 * 256 * 3);
                for row in 0..256 {
                    let offset = ((row + i * 8) * info.width as usize + i * 16) * 3;
                    data.extend_from_slice(&photo[offset..offset + 256 * 3]);
                }
                PixelBuffer::from_vec(data, 256, 256, PixelDescriptor::RGB8_SRGB).unwrap()
            })
            .collect(),
    );

    for lossy in [false, true] {
        for shared in [false, true] {
            let name = format!(
                "{}_anim_encode_{}",
                if shared { "shared" } else { "native" },
                if lossy { "lossy" } else { "lossless" }
            );
            let frames = frames.clone();
            suite.compare(name, move |g| {
                settings(g);
                for (name, work) in [
                    (
                        "main",
                        if shared {
                            main_arm::shared_encode
                        } else {
                            main_arm::native_encode
                        } as fn(&[PixelBuffer], bool) -> usize,
                    ),
                    (
                        "pr",
                        if shared {
                            pr_arm::shared_encode
                        } else {
                            pr_arm::native_encode
                        } as fn(&[PixelBuffer], bool) -> usize,
                    ),
                ] {
                    let frames = frames.clone();
                    g.bench(name, move |b| {
                        let frames = frames.clone();
                        b.iter(move || black_box(work(&frames, lossy)));
                    });
                }
            });
        }
        let bytes: Arc<[u8]> = std::fs::read(root.join(if lossy {
            "images/animated/random_lossy.webp"
        } else {
            "images/animated/random_lossless.webp"
        }))
        .unwrap()
        .into();
        for shared in [false, true] {
            let name = format!(
                "{}_anim_decode_{}",
                if shared { "shared" } else { "native" },
                if lossy { "lossy" } else { "lossless" }
            );
            let bytes = bytes.clone();
            suite.compare(name, move |g| {
                settings(g);
                for (name, work) in [
                    (
                        "main",
                        if shared {
                            main_arm::shared_decode
                        } else {
                            main_arm::native_decode
                        } as fn(&[u8]) -> usize,
                    ),
                    (
                        "pr",
                        if shared {
                            pr_arm::shared_decode
                        } else {
                            pr_arm::native_decode
                        } as fn(&[u8]) -> usize,
                    ),
                ] {
                    let bytes = bytes.clone();
                    g.bench(name, move |b| {
                        let bytes = bytes.clone();
                        b.iter(move || black_box(work(&bytes)));
                    });
                }
            });
        }
    }
    let bytes: Arc<[u8]> = std::fs::read(root.join("images/gallery1/1.webp"))
        .unwrap()
        .into();
    suite.compare("shared_still_decode_lossy", move |g| {
        settings(g);
        for (name, work) in [
            ("main", main_arm::still_decode as fn(&[u8]) -> usize),
            ("pr", pr_arm::still_decode as fn(&[u8]) -> usize),
        ] {
            let bytes = bytes.clone();
            g.bench(name, move |b| {
                let bytes = bytes.clone();
                b.iter(move || black_box(work(&bytes)));
            });
        }
    });
}
zenbench::main!(bench);
