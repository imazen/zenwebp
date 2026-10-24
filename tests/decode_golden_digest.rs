//! Cross-target decode gate: every gallery image must decode to the same
//! pixels on every target and SIMD tier, including wasm32 with and without
//! `+simd128` (run under wasmtime in CI).
//!
//! On wasm32 `+simd128`, 0.4.4 decoded every lossy file wrong: the fused
//! YUV->RGB kernel discarded every sample (solid green, dc41263 on main) and
//! the loop filter's base delta was wrong (3a7e50a on main). `cargo check`
//! never executed either kernel.
//!
//! The digests are FNV-1a over `oneshot::decode_rgb` / `decode_rgba` output.
//! On native targets the test also requires zenwebp to match libwebp (webpx)
//! byte-for-byte, so the table can only hold libwebp's pixels.

const GOLDEN: &[(&str, u64, u64)] = &[
    (
        "tests/images/gallery1/1.webp",
        0xf569a1750c4854c7,
        0x77d2378da785e5e3,
    ),
    (
        "tests/images/gallery1/2.webp",
        0xed94b4f03953e270,
        0xb44f64d33ee3c7f0,
    ),
    (
        "tests/images/gallery1/3.webp",
        0x22b956eac0f9633f,
        0x44669a4308036a57,
    ),
    (
        "tests/images/gallery1/4.webp",
        0xb9a80f24d9095835,
        0xc9c8537bcf09927f,
    ),
    (
        "tests/images/gallery1/5.webp",
        0x156f486cf607d937,
        0xf1f0d90e3d164e43,
    ),
    (
        "tests/images/gallery2/1_webp_a.webp",
        0x7fb1cbb1d6b5fc29,
        0x35d0a6ac1b9ca811,
    ),
    (
        "tests/images/gallery2/1_webp_ll.webp",
        0x21f1bd0fc0e730ff,
        0x308ed50c436491fd,
    ),
    (
        "tests/images/gallery2/2_webp_a.webp",
        0x4c71fae351e8a4f0,
        0xd7ff39a55c9251f1,
    ),
    (
        "tests/images/gallery2/2_webp_ll.webp",
        0xe2efc50a0914e515,
        0xcad9b3d7d846d372,
    ),
    (
        "tests/images/gallery2/3_webp_a.webp",
        0xee730e167c7b2134,
        0xca60bf1408ecb028,
    ),
    (
        "tests/images/gallery2/3_webp_ll.webp",
        0x329c00d2157a2754,
        0x0647ab51b6644328,
    ),
    (
        "tests/images/gallery2/4_webp_a.webp",
        0x23e135b6b5b903d8,
        0xc7120cbfc0b9e945,
    ),
    (
        "tests/images/gallery2/4_webp_ll.webp",
        0x2795f9406668e6f4,
        0x848816c9c2999827,
    ),
    (
        "tests/images/gallery2/5_webp_a.webp",
        0xa9aff2b6cbaa287a,
        0xf7d987ee0bfe2a1c,
    ),
    (
        "tests/images/gallery2/5_webp_ll.webp",
        0xa3a1b8174cfe0e6a,
        0x353e16c56407d06e,
    ),
];

fn fnv(d: &[u8]) -> u64 {
    d.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

#[test]
fn gallery_decodes_match_golden_digests() {
    // Relative to the package root (cargo's cwd for tests): wasmtime only
    // exposes the directory granted with `--dir=.`, not absolute host paths.
    let mut bad = Vec::new();
    for &(name, rgb_digest, rgba_digest) in GOLDEN {
        let data = std::fs::read(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (rgb, ..) = zenwebp::oneshot::decode_rgb(&data).unwrap();
        let (rgba, ..) = zenwebp::oneshot::decode_rgba(&data).unwrap();
        if fnv(&rgb) != rgb_digest || fnv(&rgba) != rgba_digest {
            bad.push(format!(
                "{name}: rgb {:016x} rgba {:016x}",
                fnv(&rgb),
                fnv(&rgba)
            ));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (lib_rgb, ..) = webpx::decode_rgb(&data).unwrap();
            let (lib_rgba, ..) = webpx::decode_rgba(&data).unwrap();
            assert!(
                lib_rgb == rgb && lib_rgba == rgba,
                "{name}: zenwebp != libwebp"
            );
        }
    }
    assert!(
        bad.is_empty(),
        "decode digests changed:\n{}",
        bad.join("\n")
    );
}
