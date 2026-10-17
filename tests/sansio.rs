//! Incremental (sans-I/O) decode through `dyn_incremental_decoder`: the
//! testkit's incremental contract on lossy and lossless encodes.

use zencodec::decode::DecoderConfig;
use zencodec::encode::{EncodeJob, EncoderConfig};
use zencodec_testkit as tk;
use zenwebp::zencodec::{WebpDecoderConfig as D, WebpEncoderConfig as E};

#[test]
fn incremental_contract_holds() {
    assert!(D::capabilities().incremental());
    for enc in [E::lossy(), E::lossless()] {
        for img in [
            tk::TestImage::rgb8_gradient(37, 21),
            tk::TestImage::rgba8_gradient(64, 48),
        ] {
            let bytes = enc.clone().job().encode(img.as_slice()).unwrap();
            tk::check_incremental_capability_honesty(D::new(), bytes.data()).unwrap();
        }
    }
}
