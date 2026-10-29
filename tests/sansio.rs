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

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use zencodec::decode::{
    AnimationFrameDecoder, Decode, DecodeJob, Event, Input, StepError, StreamingDecode,
};
use zencodec::source::AccessPattern;
use zencodec::{CodecErrorExt, LimitExceeded, ResourceLimits};
use zencodec_io::{CollectSink, drive_from_slice};

fn fixtures(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            fixtures(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "webp") {
            out.push(path);
        }
    }
}

fn assert_parity(data: &[u8]) {
    let expected = D::new()
        .job()
        .decoder(Cow::Borrowed(data), &[])
        .and_then(|dec| dec.decode());
    let mut inc = D::new().job().dyn_incremental_decoder(&[]).unwrap();
    assert_eq!(inc.access_pattern(), AccessPattern::Sequential);
    let mut sink = CollectSink::new();
    let got = drive_from_slice(&mut *inc, data, &mut sink, None, 7);
    match (expected, got) {
        (Ok(expected), Ok((info, _))) => {
            let pixels = expected.into_buffer();
            assert_eq!((info.width, info.height), (pixels.width(), pixels.height()));
            assert_eq!(sink.descriptor(), Some(pixels.descriptor()));
            assert_eq!(sink.bytes(), pixels.into_parts().data);
            assert!(sink.finished());
        }
        (Err(_), Err(zencodec_io::DriveError::Step(StepError::Codec(_)))) => {}
        (expected, got) => panic!("one-shot/incremental outcome mismatch: {expected:?} / {got:?}"),
    }
}

#[test]
fn every_fixture_and_truncation_matches_one_shot() {
    let mut paths = Vec::new();
    fixtures(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/images"),
        &mut paths,
    );
    paths.sort();
    assert!(!paths.is_empty());
    for path in &paths {
        eprintln!("fixture {}", path.display());
        let data = std::fs::read(path).unwrap();
        tk::check_incremental_capability_honesty(D::new(), &data).unwrap();
        tk::check_seekable_capability_honesty(D::new(), &data).unwrap();
        let header = match &data[12..16] {
            b"VP8 " | b"VP8X" => 30,
            b"VP8L" => 25,
            magic => panic!("unexpected first chunk {magic:?}"),
        };
        for cut in 0..header {
            assert!(
                D::new().job().probe(&data[..cut]).is_err(),
                "{}: accepted incomplete header at {cut}",
                path.display()
            );
        }
        D::new().job().probe(&data[..header]).unwrap();
        assert_other_decode_paths(&data);
        // Header boundary, middle of the compressed image, and final byte.
        for cut in [
            0,
            header.saturating_sub(1),
            header,
            data.len() / 2,
            data.len() - 1,
            data.len(),
        ] {
            assert_parity(&data[..cut]);
        }
        let mut inc = D::new().job().dyn_incremental_decoder(&[]).unwrap();
        let mut sink = CollectSink::new();
        let event = inc
            .step(
                Input::new(0, &data[..header]).with_total_len(data.len() as u64),
                &mut sink,
                &zencodec::NoPulse,
            )
            .unwrap();
        assert!(matches!(event.event, Event::Metadata(_)));
        assert_eq!(sink.descriptor(), None, "rows before complete input");
    }
    eprintln!("verified {} WebP fixtures", paths.len());
}

#[test]
fn input_limit_rejects_known_length_before_requesting_file() {
    let mut inc = D::new()
        .job()
        .with_limits(ResourceLimits::none().with_max_input_bytes(32))
        .dyn_incremental_decoder(&[])
        .unwrap();
    let mut sink = CollectSink::new();
    let result = inc.step(
        Input::empty().with_total_len(1024 * 1024),
        &mut sink,
        &zencodec::NoPulse,
    );
    let Err(StepError::Codec(error)) = result else {
        panic!("must reject before requesting the oversized file: {result:?}");
    };
    assert_eq!(
        error.limit_exceeded(),
        Some(&LimitExceeded::InputSize {
            actual: 1024 * 1024,
            max: 32
        })
    );
    assert!(sink.descriptor().is_none());
}

#[test]
fn known_length_limit_is_independent_of_the_offered_offset() {
    let mut inc = D::new()
        .job()
        .with_limits(ResourceLimits::none().with_max_input_bytes(32))
        .dyn_incremental_decoder(&[])
        .unwrap();
    let mut sink = CollectSink::new();
    let result = inc.step(
        Input::new(17, &[0]).with_total_len(1024 * 1024),
        &mut sink,
        &zencodec::NoPulse,
    );
    let Err(StepError::Codec(error)) = result else {
        panic!("known oversized input must fail before any Need: {result:?}");
    };
    assert_eq!(
        error.limit_exceeded(),
        Some(&LimitExceeded::InputSize {
            actual: 1024 * 1024,
            max: 32,
        })
    );
    assert!(sink.descriptor().is_none());
    assert!(matches!(
        inc.step(Input::empty(), &mut sink, &zencodec::NoPulse),
        Err(StepError::Finished)
    ));
}
#[test]
fn shared_check_all() {
    tk::check_all(E::lossless(), D::new()).unwrap();
}

#[test]
fn unknown_length_input_requests_stay_bounded() {
    for limits in [
        ResourceLimits::none().with_max_input_bytes(32),
        ResourceLimits::none().with_max_memory(32),
    ] {
        let mut inc = D::new()
            .job()
            .with_limits(limits)
            .dyn_incremental_decoder(&[])
            .unwrap();
        let mut sink = CollectSink::new();
        let result = inc
            .step(Input::empty(), &mut sink, &zencodec::NoPulse)
            .unwrap();
        assert_eq!(result.event, Event::Need);
        assert!(matches!(
            inc.requested(),
            [zencodec::decode::InputRequest::AtLeast {
                offset: 0,
                min_len: 1..=33
            }]
        ));
        let bytes = [0u8; 33];
        assert!(matches!(
            inc.step(Input::new(0, &bytes), &mut sink, &zencodec::NoPulse),
            Err(StepError::Codec(_))
        ));
        assert!(matches!(
            inc.step(Input::empty(), &mut sink, &zencodec::NoPulse),
            Err(StepError::Finished)
        ));
    }
}

#[test]
fn header_limits_fail_before_the_body_is_requested() {
    let data = include_bytes!("images/gallery1/1.webp");
    let header = (0..=data.len())
        .find(|&n| D::new().job().probe(&data[..n]).is_ok())
        .unwrap();
    let mut inc = D::new()
        .job()
        .with_limits(ResourceLimits::none().with_max_width(1))
        .dyn_incremental_decoder(&[])
        .unwrap();
    let mut sink = CollectSink::new();
    assert!(matches!(
        inc.step(
            Input::new(0, &data[..header]),
            &mut sink,
            &zencodec::NoPulse
        ),
        Err(StepError::Codec(_))
    ));
    assert!(sink.descriptor().is_none());
}

#[test]
fn exact_input_cap_allows_eof_without_early_rows() {
    let data = include_bytes!("images/regression/tiny.webp");
    let mut inc = D::new()
        .job()
        .with_limits(ResourceLimits::none().with_max_input_bytes(data.len() as u64))
        .dyn_incremental_decoder(&[])
        .unwrap();
    let mut sink = CollectSink::new();
    // Even a complete-looking payload is not EOF until the source says so.
    for _ in 0..2 {
        let p = inc
            .step(Input::new(0, data), &mut sink, &zencodec::NoPulse)
            .unwrap();
        assert!(matches!(p.event, Event::Metadata(_) | Event::Need));
        assert_eq!(p.consumed, 0);
        assert!(sink.descriptor().is_none());
    }
    let p = inc
        .step(
            Input::new(0, data).with_total_len(data.len() as u64),
            &mut sink,
            &zencodec::NoPulse,
        )
        .unwrap();
    assert!(matches!(p.event, Event::Rows { .. }));
    assert!(sink.finished());
    assert!(matches!(
        inc.step(Input::empty(), &mut sink, &zencodec::NoPulse)
            .unwrap()
            .event,
        Event::Done(_)
    ));
}

// A real driver-owned window: deliberately do not expose SliceReadAt::as_slice.
struct WindowSource<'a>(zencodec::source::SliceReadAt<'a>);
impl zencodec::source::ReadAt for WindowSource<'_> {
    type Error = zencodec::source::SourceError;
    fn len(&mut self) -> Result<u64, Self::Error> {
        self.0.len()
    }
    fn len_hint(&self) -> Option<u64> {
        self.0.len_hint()
    }
    fn read_exact_at(&mut self, offset: u64, out: &mut [u8]) -> Result<(), Self::Error> {
        self.0.read_exact_at(offset, out)
    }
    fn read_at(&mut self, offset: u64, out: &mut [u8]) -> Result<usize, Self::Error> {
        self.0.read_at(offset, out)
    }
}

fn assert_other_decode_paths(data: &[u8]) {
    use zenpixels::PixelDescriptor;
    for preferred in [
        &[][..],
        &[PixelDescriptor::BGRA8_SRGB][..],
        &[PixelDescriptor::RGBA8_SRGB][..],
    ] {
        let reference = D::new()
            .job()
            .decoder(Cow::Borrowed(data), preferred)
            .unwrap()
            .decode()
            .unwrap()
            .into_buffer();
        let mut stream = D::new()
            .job()
            .streaming_decoder(Cow::Borrowed(data), preferred)
            .unwrap();
        let mut actual = Vec::new();
        let mut row = 0;
        while let Some((y, strip)) = stream.next_batch().unwrap() {
            assert_eq!(y, row);
            assert_eq!(strip.descriptor(), reference.descriptor());
            for r in 0..strip.rows() {
                actual.extend_from_slice(strip.row(r));
            }
            row += strip.rows();
        }
        assert_eq!(row, reference.height());
        let mut expected = Vec::new();
        for row in 0..reference.height() {
            expected.extend_from_slice(reference.as_slice().row(row));
        }
        assert_eq!(actual, expected);
        let mut animation = D::new()
            .job()
            .animation_frame_decoder(Cow::Borrowed(data), preferred)
            .unwrap();
        let frame = animation.render_next_frame_owned(None).unwrap().unwrap();
        assert_eq!(frame.pixels().descriptor(), reference.descriptor());
        for r in 0..reference.height() {
            assert_eq!(frame.pixels().row(r), reference.as_slice().row(r));
        }
        tk::check_read_at_contract(
            &mut zencodec::source::ReadAtShim(WindowSource(zencodec::source::SliceReadAt::new(
                data,
            ))),
            data,
        )
        .unwrap();
        let mut source = WindowSource(zencodec::source::SliceReadAt::new(data));
        let mut inc = D::new().job().dyn_incremental_decoder(preferred).unwrap();
        let mut sink = CollectSink::new();
        let (_, stats) = zencodec_io::drive_from_read_at(
            &mut *inc,
            &mut source,
            &mut sink,
            None,
            zencodec_io::DriveOptions::new(0, 7),
        )
        .unwrap();
        assert!(stats.reads > 0);
        assert_eq!(sink.descriptor(), Some(reference.descriptor()));
        assert_eq!(sink.bytes(), reference.into_parts().data);
    }
}

#[test]
fn complete_animation_limits_are_checked_after_early_header() {
    let data = include_bytes!("images/animated/random_lossless.webp");
    let count = D::new().job().probe(data).unwrap().frame_count().unwrap();
    assert!(count > 1);
    let mut inc = D::new()
        .job()
        .with_limits(ResourceLimits::none().with_max_frames(1))
        .dyn_incremental_decoder(&[])
        .unwrap();
    let mut sink = CollectSink::new();
    let early = inc
        .step(Input::new(0, &data[..30]), &mut sink, &zencodec::NoPulse)
        .unwrap();
    assert!(matches!(early.event, Event::Metadata(_)));
    let Err(StepError::Codec(error)) = inc.step(
        Input::new(0, data).with_total_len(data.len() as u64),
        &mut sink,
        &zencodec::NoPulse,
    ) else {
        panic!("full animation must be refused before emitting rows");
    };
    assert_eq!(
        error.limit_exceeded(),
        Some(&LimitExceeded::Frames {
            actual: count,
            max: 1
        })
    );
    assert!(sink.descriptor().is_none());
}

#[test]
fn structured_orientation_uses_exif_without_rotating_encoded_pixels() {
    use zencodec::{Metadata, MetadataPolicy, Orientation};
    let image = tk::TestImage::rgb8_gradient(19, 11);
    let bytes = E::lossless()
        .job()
        .with_metadata_policy(
            Metadata::none().with_orientation(Orientation::Rotate90),
            MetadataPolicy::Web,
        )
        .encode(image.as_slice())
        .unwrap();
    let info = D::new().job().probe(bytes.data()).unwrap();
    assert_eq!(info.orientation, Orientation::Rotate90);
    // Independent libwebp readback: the EXIF tag carries the rotation, while
    // the codec's stored pixels remain the caller's original pixels.
    let oracle = webp::Decoder::new(bytes.data()).decode().unwrap();
    assert_eq!((oracle.width(), oracle.height()), (19, 11));
    let mut expected = Vec::new();
    for row in 0..11 {
        expected.extend_from_slice(image.as_slice().row(row));
    }
    assert_eq!(&*oracle, expected);
}

#[test]
fn complete_invalid_extended_file_is_not_an_incomplete_header() {
    let image = tk::TestImage::rgb8_gradient(32, 24);
    let data = E::lossy().job().encode(image.as_slice()).unwrap();
    assert_eq!(&data.data()[12..16], b"VP8 ");
    let mut extended = data.data()[..12].to_vec();
    extended.extend_from_slice(b"VP8X\x0a\0\0\0");
    // Metadata flags require absent ICC and EXIF chunks in this complete file.
    extended.extend_from_slice(&[0x28, 0, 0, 0, 31, 0, 0, 23, 0, 0]);
    extended.extend_from_slice(&data.data()[12..]);
    let riff_size = u32::try_from(extended.len() - 8).unwrap();
    extended[4..8].copy_from_slice(&riff_size.to_le_bytes());
    D::new().job().probe(&extended[..30]).unwrap();
    assert!(D::new().job().probe(&extended).is_err());
}
