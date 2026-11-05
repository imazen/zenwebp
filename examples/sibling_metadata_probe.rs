//! Diagnostic for the shared-core checked metadata boundary; does not encode.
use zencodec::encode::{EncodeJob, EncoderConfig};
use zencodec::{Metadata, MetadataPolicy};
use zenwebp::zencodec::WebpEncoderConfig;

fn main() {
    let cases = [
        ("empty", Metadata::none()),
        (
            "icc",
            Metadata::none().with_icc(zenpixels_convert::icc_profiles::DISPLAY_P3_V4),
        ),
        (
            "cicp",
            Metadata::none().with_cicp(zenpixels::Cicp::DISPLAY_P3),
        ),
    ];
    let mut rejected = 0;
    for (name, metadata) in cases {
        let mut job = WebpEncoderConfig::lossless().job();
        match job.try_set_metadata_policy(metadata, MetadataPolicy::PreserveExact) {
            Ok(()) => println!("CHECKED_METADATA {name}: accepted"),
            Err(error) => {
                rejected += 1;
                println!("CHECKED_METADATA {name}: rejected: {error}");
            }
        }
    }
    println!("CHECKED_METADATA rejected={rejected}/3");
}
