#![cfg(not(target_arch = "wasm32"))]

#[path = "support/icc.rs"]
mod icc;

use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
use zencodec::{Metadata, MetadataPolicy};
use zenpixels::{Cicp, ColorSpace, ImageSpec, PixelView};
use zenwebp::zencodec::WebpEncoderConfig;

#[test]
#[allow(deprecated)] // Exercise the documented legacy metadata path.
fn still_p3_carrier_matches_colorants_and_trc_and_preserves_authored_icc() {
    let authored = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    let codes = [0, 32, 255, 192, 128, 8];
    for explicit in [false, true] {
        let metadata = if explicit {
            Metadata::none().with_icc(authored)
        } else {
            Metadata::none().with_cicp(Cicp::DISPLAY_P3)
        };
        let spec = ImageSpec::new(zenpixels::SampleLayout::RGB8, ColorSpace::DISPLAY_P3);
        let out = WebpEncoderConfig::lossless()
            .job()
            .with_metadata_policy(metadata, MetadataPolicy::PreserveExact)
            .encoder()
            .unwrap()
            .encode(PixelView::from_bytes(&codes, 2, 1, &spec).unwrap())
            .unwrap();
        let embedded = webpx::get_icc_profile(out.data()).unwrap().unwrap();
        icc::assert_p3(&embedded, explicit);
        if explicit {
            assert_eq!(embedded, authored);
        }
        assert_eq!(webpx::decode_rgb(out.data()).unwrap().0, codes);
    }
}

#[test]
fn unified_dev_graph_encodes_pq_labeled_samples_with_bundled_database() {
    // zenpng enables icc-db in this graph, even when zenwebp's cms is off.
    let codes = [0, 64, 255, 192, 128, 8];
    let spec = ImageSpec::new(zenpixels::SampleLayout::RGB8, ColorSpace::BT2100_PQ);
    let out = WebpEncoderConfig::lossless()
        .job()
        .encoder()
        .unwrap()
        .encode(PixelView::from_bytes(&codes, 2, 1, &spec).unwrap())
        .unwrap();
    icc::assert_pq(&webpx::get_icc_profile(out.data()).unwrap().unwrap());
    assert_eq!(webpx::decode_rgb(out.data()).unwrap().0, codes);
}
