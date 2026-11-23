//! Explicit, corpus-free photographic decode fixtures for before/after runs.
//! Select with `ZENWEBP_BENCH_FIXTURES=gallery`; no files are skipped.

use zenbench::{Throughput, black_box};

const LOSSY: [&[u8]; 5] = [
    include_bytes!("../../tests/images/gallery1/1.webp"),
    include_bytes!("../../tests/images/gallery1/2.webp"),
    include_bytes!("../../tests/images/gallery1/3.webp"),
    include_bytes!("../../tests/images/gallery1/4.webp"),
    include_bytes!("../../tests/images/gallery1/5.webp"),
];
const LOSSLESS: [&[u8]; 5] = [
    include_bytes!("../../tests/images/gallery2/1_webp_ll.webp"),
    include_bytes!("../../tests/images/gallery2/2_webp_ll.webp"),
    include_bytes!("../../tests/images/gallery2/3_webp_ll.webp"),
    include_bytes!("../../tests/images/gallery2/4_webp_ll.webp"),
    include_bytes!("../../tests/images/gallery2/5_webp_ll.webp"),
];

pub fn decode(suite: &mut zenbench::Suite, lossless: bool) {
    let fixtures = if lossless { LOSSLESS } else { LOSSY };
    let kind = if lossless { "vp8l" } else { "vp8" };
    for (i, bytes) in fixtures.into_iter().enumerate() {
        let (rgba, w, h) = zenwebp::oneshot::decode_rgba(bytes).unwrap();
        let oracle = webp::Decoder::new(bytes).decode().unwrap();
        assert_eq!((oracle.width(), oracle.height()), (w, h));
        if oracle.is_alpha() {
            assert_eq!(rgba.as_slice(), &*oracle);
        } else {
            for (actual, expected) in rgba
                .as_chunks::<4>()
                .0
                .iter()
                .zip(oracle.as_chunks::<3>().0)
            {
                assert_eq!(&actual[..3], expected);
                assert_eq!(actual[3], 255);
            }
        }
        eprintln!(
            "gallery/{kind}/{}: {w}x{h}, {} bytes, libwebp exact",
            i + 1,
            bytes.len()
        );
        suite.compare(format!("gallery/{kind}/{}", i + 1), |group| {
            group.throughput(Throughput::Elements(u64::from(w) * u64::from(h)));
            group.throughput_unit("pixels");
            group.bench("zenwebp", move |b| {
                b.iter(move || black_box(zenwebp::oneshot::decode_rgba(black_box(bytes)).unwrap()))
            });
            group.bench("libwebp", move |b| {
                b.iter(move || black_box(webp::Decoder::new(black_box(bytes)).decode().unwrap()))
            });
        });
    }
}
