#[path = "../../../tests/support/icc.rs"]
mod icc;

use std::sync::Arc;

use zencodec::encode::{AnimationFrameEncoder, EncodeJob, Encoder, EncoderConfig};
use zencodec::{CodecErrorExt, ErrorCategory, Metadata, MetadataPolicy, RequestError};
use zenpixels::{AlphaMode, ColorSpace, ImageSpec, PixelView, RenderingIntent, SampleLayout};
use zenwebp::zencodec::WebpEncoderConfig;

const FIRST: [u8; 18] = [
    0, 64, 255, 192, 128, 8, 16, 211, 96, 48, 128, 224, 255, 133, 0, 5, 19, 77,
];
const SECOND: [u8; 18] = [
    255, 2, 22, 17, 34, 51, 187, 33, 54, 29, 19, 255, 55, 211, 180, 255, 1, 193,
];

fn rgb(bytes: &[u8], color: ColorSpace) -> PixelView<'_> {
    PixelView::from_packed(bytes, 3, 2, ImageSpec::new(SampleLayout::RGB8, color)).unwrap()
}

fn cms_required(error: &(dyn std::error::Error + 'static)) {
    assert_eq!(
        error.error_category(),
        Some(ErrorCategory::Request(RequestError::CmsRequired)),
        "{error}"
    );
}

#[test]
fn p3_carrier_is_available_in_the_isolated_consumer() {
    let enc = WebpEncoderConfig::lossless().job().encoder().unwrap();
    let out = enc.encode(rgb(&FIRST, ColorSpace::DISPLAY_P3)).unwrap();
    icc::assert_p3(&webpx::get_icc_profile(out.data()).unwrap().unwrap(), false);
    assert_eq!(webpx::decode_rgb(out.data()).unwrap().0, FIRST);
}

#[test]
#[allow(deprecated)] // Exercise the actual writer and its at-finish row path.
fn pq_still_uses_the_production_database_feature() {
    let enc = WebpEncoderConfig::lossless().job().encoder().unwrap();
    let mut bytes = Vec::new();
    let result = zencodec_io::encode_to_writer(enc, rgb(&FIRST, ColorSpace::BT2100_PQ), &mut bytes);
    if cfg!(feature = "cms") {
        let report = result.unwrap();
        assert_eq!(report.bytes, bytes.len() as u64);
        assert_eq!(webpx::decode_rgb(&bytes).unwrap().0, FIRST);
        icc::assert_pq(&webpx::get_icc_profile(&bytes).unwrap().unwrap());
    } else {
        let Err(zencodec_io::EncodeWriteError::Encoder(err)) = result else {
            panic!("expected an encoder error");
        };
        cms_required(&err);
        assert_eq!(bytes.len(), 0);
    }
}

#[test]
#[allow(deprecated)] // Exercise the writer and documented metadata retention path.
fn hlg_requires_rendering_and_preserves_an_explicit_rendered_profile() {
    let mut bytes = Vec::new();
    let enc = WebpEncoderConfig::lossless().job().encoder().unwrap();
    let scene = rgb(&FIRST, ColorSpace::BT2100_HLG_SCENE);
    let Err(zencodec_io::EncodeWriteError::Encoder(err)) =
        zencodec_io::encode_to_writer(enc, scene, &mut bytes)
    else {
        panic!("bare scene HLG must refuse still authoring");
    };
    cms_required(&err);
    assert_eq!(bytes.len(), 0);

    let mut anim = WebpEncoderConfig::lossless()
        .job()
        .animation_frame_encoder()
        .unwrap();
    let err = anim
        .push_frame(rgb(&FIRST, ColorSpace::BT2100_HLG_SCENE), 17, None)
        .unwrap_err();
    cms_required(&err);
    assert_eq!(
        zencodec_io::write_pending_animation(&mut anim, &mut bytes).unwrap(),
        0
    );
    assert!(anim.finish(None).is_err());
    assert_eq!(bytes.len(), 0);

    let profile: Arc<[u8]> = rendered_hlg_profile().into();
    icc::assert_rgb_display(&profile);
    let table = icc::tag(&profile, b"A2B0").unwrap();
    assert_eq!(&table[..12], b"mft2\0\0\0\0\x03\x03\x11\0");
    let at = |i| {
        let offset = 52 + 12 + 6 * i + 2;
        u16::from_be_bytes(table[offset..offset + 2].try_into().unwrap())
    };
    assert_eq!(at(0), 0);
    assert_eq!(at(17usize.pow(3) - 1), 32768);
    // For neutral HLG 0.5, scene Y = 1/12 and display Y = (1/12)^1.2.
    assert_eq!(at((8 * 17 + 8) * 17 + 8), 1661);
    assert_eq!(
        &icc::tag(&profile, b"lumi").unwrap()[12..16],
        &(1000 * 65536u32).to_be_bytes()
    );
    let rendered = ColorSpace::Icc {
        profile: profile.clone(),
        channels: 3,
        alpha: AlphaMode::None,
        intent: RenderingIntent::RelativeColorimetric,
    };
    let job = || {
        WebpEncoderConfig::lossless().job().with_metadata_policy(
            Metadata::none().with_icc(profile.clone()),
            MetadataPolicy::PreserveExact,
        )
    };
    let enc = job().encoder().unwrap();
    zencodec_io::encode_to_writer(enc, rgb(&FIRST, rendered.clone()), &mut bytes).unwrap();
    assert_eq!(webpx::decode_rgb(&bytes).unwrap().0, FIRST);
    assert_eq!(
        webpx::get_icc_profile(&bytes).unwrap().as_deref(),
        Some(profile.as_ref())
    );

    bytes.clear();
    let mut anim = job().animation_frame_encoder().unwrap();
    anim.push_frame(rgb(&FIRST, rendered.clone()), 17, None)
        .unwrap();
    anim.push_frame(rgb(&SECOND, rendered), 29, None).unwrap();
    assert_eq!(
        zencodec_io::write_pending_animation(&mut anim, &mut bytes).unwrap(),
        0
    );
    let out = anim.finish(None).unwrap();
    assert_eq!(
        webpx::get_icc_profile(out.data()).unwrap().as_deref(),
        Some(profile.as_ref())
    );
    let frames = webpx::AnimationDecoder::new(out.data())
        .unwrap()
        .decode_all()
        .unwrap();
    assert_eq!(frames.len(), 2);
    for (frame, expected) in frames.iter().zip([FIRST, SECOND]) {
        let rgba: Vec<u8> = expected
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[r, g, b]| [r, g, b, 255])
            .collect();
        assert_eq!(frame.data, rgba);
    }
}

fn fixed(value: f64) -> [u8; 4] {
    ((value * 65536.0).round() as i32).to_be_bytes()
}

fn xyz_tag(xyz: [f64; 3]) -> Vec<u8> {
    let mut tag = b"XYZ \0\0\0\0".to_vec();
    tag.extend(xyz.into_iter().flat_map(fixed));
    tag
}

fn text_tag(text: &str) -> Vec<u8> {
    let utf16: Vec<u8> = text.encode_utf16().flat_map(u16::to_be_bytes).collect();
    let mut tag = b"mluc\0\0\0\0\0\0\0\x01\0\0\0\x0cenUS".to_vec();
    tag.extend_from_slice(&(utf16.len() as u32).to_be_bytes());
    tag.extend_from_slice(&28u32.to_be_bytes());
    tag.extend(utf16);
    tag
}

/// BT.2100 HLG for a 1000 cd/m² display with zero black. A 3D A2B0
/// applies the luminance-coupled OOTF and lands in D50 XYZ. No synthesis
/// or color interpretation from the crate under test is used.
fn rendered_hlg_profile() -> Vec<u8> {
    let mut lut = b"mft2\0\0\0\0\x03\x03\x11\0".to_vec();
    for i in 0..9 {
        lut.extend(fixed(if i % 4 == 0 { 1.0 } else { 0.0 }));
    }
    lut.extend([0, 2, 0, 2]);
    lut.extend([0, 0, 255, 255].repeat(3));
    let scene = |code: u8| {
        let x = f64::from(code) / 16.0;
        if x <= 0.5 {
            x * x / 3.0
        } else {
            (((x - 0.55991073) / 0.17883277).exp() + 0.28466892) / 12.0
        }
    };
    for r in 0..17 {
        for g in 0..17 {
            for b in 0..17 {
                let [r, g, b] = [r, g, b].map(scene);
                let ootf = (0.2627 * r + 0.6780 * g + 0.0593 * b).powf(0.2);
                let xyz = [
                    0.67348 * r + 0.16566 * g + 0.12506 * b,
                    0.27904 * r + 0.67534 * g + 0.04562 * b,
                    -0.00194 * r + 0.02998 * g + 0.79686 * b,
                ];
                lut.extend(
                    xyz.map(|v| ((v * ootf * 32768.0).round().max(0.0) as u16).to_be_bytes())
                        .concat(),
                );
            }
        }
    }
    lut.extend([0, 0, 255, 255].repeat(3));
    let d50 = [0.9642, 1.0, 0.8249];
    let tags = [
        (b"desc", text_tag("BT.2100 HLG display, 1000 cd/m2")),
        (b"cprt", text_tag("Test fixture")),
        (b"wtpt", xyz_tag(d50)),
        (b"lumi", xyz_tag([0.0, 1000.0, 0.0])),
        (b"cicp", b"cicp\0\0\0\0\x09\x12\0\x01".to_vec()),
        (b"A2B0", lut),
    ];
    let mut profile = vec![0; 132 + 12 * tags.len()];
    profile[8..12].copy_from_slice(&[4, 0x30, 0, 0]);
    profile[12..24].copy_from_slice(b"mntrRGB XYZ ");
    profile[24..36].copy_from_slice(&[0x07, 0xea, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]);
    profile[36..40].copy_from_slice(b"acsp");
    profile[67] = 1; // relative colorimetric
    for (i, value) in d50.into_iter().enumerate() {
        profile[68 + i * 4..72 + i * 4].copy_from_slice(&fixed(value));
    }
    profile[128..132].copy_from_slice(&(tags.len() as u32).to_be_bytes());
    for (i, (name, data)) in tags.into_iter().enumerate() {
        let entry = 132 + 12 * i;
        let offset = (profile.len() as u32).to_be_bytes();
        profile[entry..entry + 4].copy_from_slice(name);
        profile[entry + 4..entry + 8].copy_from_slice(&offset);
        profile[entry + 8..entry + 12].copy_from_slice(&(data.len() as u32).to_be_bytes());
        profile.extend(data);
        profile.resize(profile.len().next_multiple_of(4), 0);
    }
    let len = (profile.len() as u32).to_be_bytes();
    profile[..4].copy_from_slice(&len);
    profile
}
